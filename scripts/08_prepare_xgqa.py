import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image as PilImage


dataset_name = "floschne/xgqa"

reasoning_terms = [
    "left",
    "right",
    "behind",
    "front",
    "above",
    "below",
    "under",
    "between",
    "near",
    "next to",
    "holding",
    "wearing",
    "standing",
    "sitting",
    "same color",
    "different",
    "both",
    "either",
    "or",
    "around",
    "inside",
    "outside",
    "beside",
    "on top of",
    "in front of",
]


def load_language_split(language, cache_dir):
    from datasets import load_dataset

    print(f"Loading xGQA split: {language}")

    dataset = load_dataset(
        dataset_name,
        split=language,
        cache_dir=str(cache_dir),
    )

    needed_columns = [
        "question",
        "answer",
        "full_answer",
        "image_id",
        "image",
    ]

    missing_columns = [
        column
        for column in needed_columns
        if column not in dataset.column_names
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            f"Missing columns in {language}: {missing_text}"
        )

    return dataset


def metadata_table(dataset, language):
    metadata = dataset.select_columns(
        [
            "question",
            "answer",
            "full_answer",
            "image_id",
        ]
    ).to_pandas()

    metadata = metadata.rename(
        columns={
            "question": f"question_{language}",
            "answer": f"answer_{language}",
            "full_answer": f"full_answer_{language}",
        }
    )

    metadata["source_row"] = range(len(metadata))
    metadata["image_id"] = metadata["image_id"].astype(str)

    return metadata


def clean_text(text):
    return " ".join(str(text).lower().strip().split())


def check_language_alignment(english_table, bangla_table):
    if len(english_table) != len(bangla_table):
        raise ValueError(
            f"Split lengths differ: "
            f"English={len(english_table)}, Bangla={len(bangla_table)}"
        )

    image_matches = (
        english_table["image_id"].reset_index(drop=True)
        == bangla_table["image_id"].reset_index(drop=True)
    )

    answer_matches = [
        clean_text(english_answer) == clean_text(bangla_answer)
        for english_answer, bangla_answer in zip(
            english_table["answer_en"],
            bangla_table["answer_bn"],
        )
    ]

    image_match_count = int(image_matches.sum())
    answer_match_count = int(sum(answer_matches))

    print()
    print("xGQA language alignment")
    print(f"English rows: {len(english_table)}")
    print(f"Bangla rows: {len(bangla_table)}")
    print(f"Matching image positions: {image_match_count}")
    print(f"Matching answer positions: {answer_match_count}")
    print(f"Unique images: {english_table['image_id'].nunique()}")

    if image_match_count != len(english_table):
        raise ValueError(
            "English and Bangla image ordering does not match."
        )

    if answer_match_count != len(english_table):
        raise ValueError(
            "English and Bangla answer ordering does not match."
        )


def combine_languages(english_table, bangla_table):
    bangla_columns = bangla_table[
        [
            "question_bn",
            "answer_bn",
            "full_answer_bn",
            "image_id",
            "source_row",
        ]
    ].rename(
        columns={
            "image_id": "bangla_image_id",
            "source_row": "bangla_source_row",
        }
    )

    combined = pd.concat(
        [
            english_table.reset_index(drop=True),
            bangla_columns.reset_index(drop=True),
        ],
        axis=1,
    )

    combined["image_id_match"] = (
        combined["image_id"] == combined["bangla_image_id"]
    )

    combined["answer_match"] = [
        clean_text(english_answer) == clean_text(bangla_answer)
        for english_answer, bangla_answer in zip(
            combined["answer_en"],
            combined["answer_bn"],
        )
    ]

    return combined


def reasoning_score(question):
    text = clean_text(question)
    word_count = len(text.split())
    term_count = sum(term in text for term in reasoning_terms)

    score = word_count + (term_count * 4)

    if " and " in text:
        score += 2

    if " or " in text:
        score += 2

    if text.startswith("why"):
        score += 4

    if text.startswith("where"):
        score += 3

    if text.startswith("which"):
        score += 2

    if re.search(r"\b(left|right)\b", text):
        score += 3

    return score


def select_one_per_image(combined, target_images, seed):
    candidates = combined.copy()
    candidates["reasoning_score"] = (
        candidates["question_en"].map(reasoning_score)
    )
    candidates["question_length"] = (
        candidates["question_en"]
        .fillna("")
        .str.split()
        .str.len()
    )

    candidates = candidates.sample(
        frac=1,
        random_state=seed,
    )

    candidates = candidates.sort_values(
        by=[
            "image_id",
            "reasoning_score",
            "question_length",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        kind="stable",
    )

    selected = candidates.drop_duplicates(
        subset=["image_id"],
        keep="first",
    ).copy()

    selected = selected.sort_values(
        by=[
            "reasoning_score",
            "question_length",
            "image_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    )

    if len(selected) < target_images:
        raise ValueError(
            f"Only {len(selected)} unique images are available, "
            f"but {target_images} were requested."
        )

    selected = selected.head(target_images).copy()
    selected = selected.sort_values(
        by="image_id",
        kind="stable",
    ).reset_index(drop=True)

    selected["benchmark_group_id"] = [
        f"xgqa_{row_number:04d}"
        for row_number in range(len(selected))
    ]
    selected["benchmark_source"] = "xgqa"
    selected["reasoning_category"] = "compositional_reasoning"
    selected["canonical_answer"] = selected["answer_en"]
    selected["translation_verified"] = False
    selected["selection_seed"] = seed

    return selected


def decode_image_value(image_value):
    from datasets import Image as DatasetImage

    if isinstance(image_value, PilImage.Image):
        return image_value.convert("RGB")

    possible_values = (
        image_value
        if isinstance(image_value, list)
        else [image_value]
    )

    decoder = DatasetImage()

    for value in possible_values:
        if isinstance(value, PilImage.Image):
            return value.convert("RGB")

        if isinstance(value, dict):
            try:
                decoded = decoder.decode_example(value)
                return decoded.convert("RGB")
            except Exception:
                continue

    raise ValueError("Could not decode the xGQA image.")


def save_selected_images(english_dataset, selected, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_images = {}

    print()
    print("Saving selected xGQA images")

    for row_number, row in enumerate(
        selected.itertuples(index=False),
        start=1,
    ):
        image_id = str(row.image_id)
        output_path = image_dir / f"{image_id}.jpg"

        if not output_path.exists():
            source_example = english_dataset[int(row.source_row)]
            image = decode_image_value(source_example["image"])
            image.save(
                output_path,
                format="JPEG",
                quality=95,
            )

        saved_images[image_id] = str(output_path)

        if row_number % 50 == 0 or row_number == len(selected):
            print(f"Saved {row_number}/{len(selected)} images")

    selected = selected.copy()
    selected["image_path"] = selected["image_id"].map(saved_images)

    return selected


def validate_selection(selected, image_dir, target_images):
    if len(selected) != target_images:
        raise ValueError(
            f"Expected {target_images} rows, found {len(selected)}"
        )

    if selected["image_id"].duplicated().any():
        raise ValueError("Repeated images were found.")

    if selected["benchmark_group_id"].duplicated().any():
        raise ValueError("Repeated benchmark group IDs were found.")

    if not selected["image_id_match"].all():
        raise ValueError("English and Bangla image IDs do not match.")

    if not selected["answer_match"].all():
        raise ValueError("English and Bangla answers do not match.")

    empty_english = int(
        selected["question_en"].fillna("").str.strip().eq("").sum()
    )
    empty_bangla = int(
        selected["question_bn"].fillna("").str.strip().eq("").sum()
    )

    if empty_english or empty_bangla:
        raise ValueError("Empty questions were found.")

    missing_images = [
        image_id
        for image_id in selected["image_id"]
        if not (image_dir / f"{image_id}.jpg").exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} selected images are missing."
        )


def save_selected_data(selected, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    selected.to_csv(csv_path, index=False, encoding="utf-8")

    print()
    print(f"Selected Parquet: {output_path}")
    print(f"Selected CSV: {csv_path}")


def selection_checksum(selected):
    records = [
        f"{row.benchmark_group_id}|{row.image_id}|{row.question_en}"
        for row in selected.itertuples(index=False)
    ]
    checksum_text = "\n".join(sorted(records))
    return hashlib.sha256(
        checksum_text.encode("utf-8")
    ).hexdigest()


def make_report(
    english_table,
    bangla_table,
    selected,
    image_dir,
    seed,
):
    from datetime import datetime, timezone

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "english_rows": len(english_table),
        "bangla_rows": len(bangla_table),
        "english_unique_images": int(
            english_table["image_id"].nunique()
        ),
        "bangla_unique_images": int(
            bangla_table["image_id"].nunique()
        ),
        "selected_groups": len(selected),
        "selected_unique_images": int(
            selected["image_id"].nunique()
        ),
        "saved_images": len(list(image_dir.glob("*.jpg"))),
        "matching_image_ids": int(
            selected["image_id_match"].sum()
        ),
        "matching_answers": int(
            selected["answer_match"].sum()
        ),
        "minimum_reasoning_score": int(
            selected["reasoning_score"].min()
        ),
        "maximum_reasoning_score": int(
            selected["reasoning_score"].max()
        ),
        "mean_reasoning_score": float(
            selected["reasoning_score"].mean()
        ),
        "selection_seed": seed,
        "selection_sha256": selection_checksum(selected),
        "translation_verified": False,
    }


def save_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(f"Report: {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/huggingface_cache"),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/xgqa/images"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/benchmark/xgqa_300.parquet"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/manifests/xgqa_selection_report.json"
        ),
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    english_dataset = load_language_split(
        "en",
        arguments.cache_dir,
    )
    bangla_dataset = load_language_split(
        "bn",
        arguments.cache_dir,
    )

    english_table = metadata_table(
        english_dataset,
        "en",
    )
    bangla_table = metadata_table(
        bangla_dataset,
        "bn",
    )

    check_language_alignment(
        english_table,
        bangla_table,
    )

    combined = combine_languages(
        english_table,
        bangla_table,
    )

    selected = select_one_per_image(
        combined,
        arguments.target_images,
        arguments.seed,
    )

    selected = save_selected_images(
        english_dataset,
        selected,
        arguments.image_dir,
    )

    validate_selection(
        selected,
        arguments.image_dir,
        arguments.target_images,
    )

    save_selected_data(
        selected,
        arguments.output,
    )

    report = make_report(
        english_table,
        bangla_table,
        selected,
        arguments.image_dir,
        arguments.seed,
    )
    save_report(
        report,
        arguments.report,
    )

    print()
    print("xGQA selection")
    print(f"English rows: {len(english_table)}")
    print(f"Bangla rows: {len(bangla_table)}")
    print(f"Selected groups: {len(selected)}")
    print(f"Unique images: {selected['image_id'].nunique()}")
    print(
        f"Reasoning score range: "
        f"{selected['reasoning_score'].min()}–"
        f"{selected['reasoning_score'].max()}"
    )
    print("xGQA preparation completed.")


if __name__ == "__main__":
    main()