import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from tqdm import tqdm


SEED = 2027
TARGET_SIZE = 500
DATASET_NAME = "FaiyazAbdullah114708/BanglaVerse"

SPLITS = {
    "en": "english",
    "bn": "pure_bn",
    "hi": "hindi",
    "ur": "urdu",
}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "banglaverse"
IMAGE_DIR = RAW_DIR / "images"
OUTPUT_DIR = ROOT / "data" / "processed" / "cultural"
MANIFEST_DIR = ROOT / "data" / "manifests"

PARQUET_PATH = OUTPUT_DIR / "banglaverse_500.parquet"
CSV_PATH = OUTPUT_DIR / "banglaverse_500.csv"
REPORT_PATH = MANIFEST_DIR / "banglaverse_selection_report.json"


def make_directories():
    for folder in [
        RAW_DIR,
        IMAGE_DIR,
        OUTPUT_DIR,
        MANIFEST_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def list_value(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [
            clean_value(item)
            for item in value
            if clean_value(item)
        ]

    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()

        if isinstance(converted, list):
            return [
                clean_value(item)
                for item in converted
                if clean_value(item)
            ]

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    clean_value(item)
                    for item in parsed
                    if clean_value(item)
                ]
        except json.JSONDecodeError:
            pass

        return [text]

    text = clean_value(value)
    return [text] if text else []


def find_column(columns, names):
    lookup = {
        column.lower().strip(): column
        for column in columns
    }

    for name in names:
        column = lookup.get(name.lower().strip())

        if column is not None:
            return column

    raise ValueError(
        f"Could not find any of these columns: {names}. "
        f"Available columns: {columns}"
    )


def safe_filename(value):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    return name.strip("_") or "image"


def split_filename(split_name):
    return f"data/{split_name}-00000-of-00001.parquet"


def load_one_split(split_name):
    from datasets import load_dataset

    print(f"Loading BanglaVerse split: {split_name}")

    parquet_path = hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=split_filename(split_name),
    )

    load_options = {
        "path": "parquet",
        "data_files": {split_name: parquet_path},
        "split": split_name,
    }

    try:
        dataset = load_dataset(**load_options)
    except OSError as error:
        if "Consistency check failed" not in str(error):
            raise

        parquet_path = hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=split_filename(split_name),
            force_download=True,
        )

        load_options["data_files"] = {
            split_name: parquet_path
        }
        dataset = load_dataset(
            **load_options,
            download_mode="force_redownload",
        )

    return dataset


def load_splits():
    return {
        language: load_one_split(split_name)
        for language, split_name in SPLITS.items()
    }


def prepare_split(dataset):
    columns = dataset.column_names

    image_column = find_column(columns, ["image"])
    image_id_column = find_column(
        columns,
        ["image_id", "id"],
    )
    question_column = find_column(
        columns,
        ["question"],
    )
    answer_column = find_column(
        columns,
        ["answer"],
    )
    options_column = find_column(
        columns,
        ["options"],
    )
    domain_column = find_column(
        columns,
        ["domain", "category"],
    )

    metadata = dataset.remove_columns([image_column])
    row_index = {}

    for row_number, image_id in enumerate(
        metadata[image_id_column]
    ):
        image_id = clean_value(image_id)

        if image_id and image_id not in row_index:
            row_index[image_id] = row_number

    return {
        "dataset": dataset,
        "metadata": metadata,
        "index": row_index,
        "image_column": image_column,
        "image_id_column": image_id_column,
        "question_column": question_column,
        "answer_column": answer_column,
        "options_column": options_column,
        "domain_column": domain_column,
    }


def read_language_record(split_info, image_id):
    row_number = split_info["index"][image_id]
    row = split_info["metadata"][row_number]

    return {
        "question": clean_value(
            row[split_info["question_column"]]
        ),
        "answer": clean_value(
            row[split_info["answer_column"]]
        ),
        "options": list_value(
            row[split_info["options_column"]]
        ),
    }


def find_complete_ids(split_details):
    common_ids = set(split_details["en"]["index"])

    for language in ["bn", "hi", "ur"]:
        common_ids &= set(split_details[language]["index"])

    complete_ids = []
    incomplete_counts = {
        language: {
            "question": 0,
            "answer": 0,
            "options": 0,
        }
        for language in SPLITS
    }

    for image_id in sorted(common_ids):
        complete = True

        for language in SPLITS:
            record = read_language_record(
                split_details[language],
                image_id,
            )

            if not record["question"]:
                incomplete_counts[language]["question"] += 1
                complete = False

            if not record["answer"]:
                incomplete_counts[language]["answer"] += 1
                complete = False

            if len(record["options"]) < 2:
                incomplete_counts[language]["options"] += 1
                complete = False

        if complete:
            complete_ids.append(image_id)

    return complete_ids, len(common_ids), incomplete_counts


def balanced_selection(split_info, eligible_ids):
    grouped = {}
    metadata = split_info["metadata"]

    for image_id in eligible_ids:
        row_number = split_info["index"][image_id]
        row = metadata[row_number]

        domain = clean_value(
            row[split_info["domain_column"]]
        )
        domain = domain or "other"

        grouped.setdefault(domain, []).append(image_id)

    randomizer = random.Random(SEED)

    for image_ids in grouped.values():
        randomizer.shuffle(image_ids)

    selected = []
    domains = sorted(grouped)

    while len(selected) < TARGET_SIZE:
        added = False

        for domain in domains:
            if grouped[domain]:
                selected.append(grouped[domain].pop())
                added = True

                if len(selected) == TARGET_SIZE:
                    break

        if not added:
            break

    if len(selected) < TARGET_SIZE:
        raise ValueError(
            f"Only {len(selected)} complete cultural images "
            f"were available. {TARGET_SIZE} are required."
        )

    return selected


def valid_image(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def save_image(image_value, path):
    if valid_image(path):
        return

    path.unlink(missing_ok=True)

    if isinstance(image_value, Image.Image):
        image = image_value.copy()
    elif isinstance(image_value, dict):
        if image_value.get("bytes"):
            from io import BytesIO
            image = Image.open(BytesIO(image_value["bytes"]))
        elif image_value.get("path"):
            image = Image.open(image_value["path"])
        else:
            raise ValueError("Unsupported image dictionary")
    else:
        raise ValueError(
            f"Unsupported image type: {type(image_value)}"
        )

    image.load()
    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(path, format="JPEG", quality=95)

    if not valid_image(path):
        raise ValueError(f"Invalid saved image: {path}")


def build_table(split_details, selected_ids):
    rows = []
    english_info = split_details["en"]

    for number, image_id in enumerate(
        tqdm(selected_ids, desc="Cultural images"),
        start=1,
    ):
        english_row_number = english_info["index"][image_id]
        english_row = english_info["dataset"][
            english_row_number
        ]

        image_name = f"{safe_filename(image_id)}.jpg"
        image_path = IMAGE_DIR / image_name

        save_image(
            english_row[english_info["image_column"]],
            image_path,
        )

        records = {
            language: read_language_record(
                split_details[language],
                image_id,
            )
            for language in SPLITS
        }

        domain = clean_value(
            english_info["metadata"][english_row_number][
                english_info["domain_column"]
            ]
        )

        rows.append(
            {
                "benchmark_id": f"culture_{number:04d}",
                "source": "banglaverse",
                "source_image_id": image_id,
                "domain": domain,
                "image_path": (
                    image_path.relative_to(ROOT).as_posix()
                ),
                "question_en": records["en"]["question"],
                "question_bn": records["bn"]["question"],
                "question_hi": records["hi"]["question"],
                "question_ur": records["ur"]["question"],
                "answer_en": records["en"]["answer"],
                "answer_bn": records["bn"]["answer"],
                "answer_hi": records["hi"]["answer"],
                "answer_ur": records["ur"]["answer"],
                "options_en_json": json.dumps(
                    records["en"]["options"],
                    ensure_ascii=False,
                ),
                "options_bn_json": json.dumps(
                    records["bn"]["options"],
                    ensure_ascii=False,
                ),
                "options_hi_json": json.dumps(
                    records["hi"]["options"],
                    ensure_ascii=False,
                ),
                "options_ur_json": json.dumps(
                    records["ur"]["options"],
                    ensure_ascii=False,
                ),
            }
        )

    return pd.DataFrame(rows)


def verify_table(table):
    if len(table) != TARGET_SIZE:
        raise ValueError(
            f"Expected {TARGET_SIZE} cultural examples, "
            f"found {len(table)}"
        )

    if table["benchmark_id"].duplicated().any():
        raise ValueError("Duplicate cultural benchmark IDs found")

    if table["source_image_id"].nunique() != TARGET_SIZE:
        raise ValueError("Cultural image IDs are not unique")

    required_columns = [
        "question_en",
        "question_bn",
        "question_hi",
        "question_ur",
        "answer_en",
        "answer_bn",
        "answer_hi",
        "answer_ur",
    ]

    for column in required_columns:
        missing = int(
            table[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if missing:
            raise ValueError(
                f"{column} has {missing} empty values"
            )

    missing_images = []

    for relative_path in table["image_path"]:
        image_path = ROOT / relative_path

        if not valid_image(image_path):
            missing_images.append(relative_path)

    if missing_images:
        raise ValueError(
            f"{len(missing_images)} cultural images are missing"
        )


def main():
    make_directories()

    datasets = load_splits()
    split_details = {
        language: prepare_split(dataset)
        for language, dataset in datasets.items()
    }

    complete_ids, common_count, incomplete_counts = (
        find_complete_ids(split_details)
    )

    print()
    print("BanglaVerse alignment")
    print(f"Common images across four languages: {common_count}")
    print(f"Complete multilingual images: {len(complete_ids)}")

    for language in ["en", "bn", "hi", "ur"]:
        counts = incomplete_counts[language]
        print(
            f"{language}: "
            f"{counts['question']} missing questions, "
            f"{counts['answer']} missing answers, "
            f"{counts['options']} missing option sets"
        )

    selected_ids = balanced_selection(
        split_details["en"],
        complete_ids,
    )
    table = build_table(split_details, selected_ids)
    verify_table(table)

    table.to_parquet(PARQUET_PATH, index=False)
    table.to_csv(CSV_PATH, index=False, encoding="utf-8")

    report = {
        "source": DATASET_NAME,
        "seed": SEED,
        "common_images_across_four_languages": common_count,
        "complete_multilingual_images": len(complete_ids),
        "selected_images": len(table),
        "unique_selected_images": int(
            table["source_image_id"].nunique()
        ),
        "incomplete_counts": incomplete_counts,
        "domain_counts": {
            str(key): int(value)
            for key, value in (
                table["domain"].value_counts().to_dict().items()
            )
        },
        "languages": [
            "English",
            "Bangla",
            "Hindi",
            "Urdu",
        ],
        "parquet_path": PARQUET_PATH.relative_to(ROOT).as_posix(),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "image_directory": IMAGE_DIR.relative_to(ROOT).as_posix(),
        "usage": "Separate external cultural stress test",
        "license": "Apache-2.0",
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print("BanglaVerse cultural selection")
    print(f"Selected images: {len(table)}")
    print(f"Unique images: {table['source_image_id'].nunique()}")
    print(f"Domains: {table['domain'].nunique()}")
    print(f"Parquet: {PARQUET_PATH.relative_to(ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()