import hashlib
import json
import os
import random
import re
import string
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps
from tqdm import tqdm


SEED = 2027
TARGET_SIZE = 400

DATA_URL = (
    "https://prior-datasets.s3.us-east-2.amazonaws.com/"
    "aokvqa/aokvqa_v1p0.tar.gz"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "aokvqa"
IMAGE_DIR = RAW_DIR / "images"
BENCHMARK_DIR = ROOT / "data" / "processed" / "benchmark"
MANIFEST_DIR = ROOT / "data" / "manifests"

ARCHIVE_PATH = RAW_DIR / "aokvqa_v1p0.tar.gz"
VQA_CORE_PATH = BENCHMARK_DIR / "vqa_core_4000.parquet"
PARQUET_PATH = BENCHMARK_DIR / "aokvqa_400.parquet"
CSV_PATH = BENCHMARK_DIR / "aokvqa_400.csv"
REPORT_PATH = MANIFEST_DIR / "aokvqa_selection_report.json"

NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ARTICLES = {"a", "an", "the"}


def make_directories():
    for folder in [
        RAW_DIR,
        IMAGE_DIR,
        BENCHMARK_DIR,
        MANIFEST_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def download_archive():
    if ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 100000:
        print(f"Using existing archive: {ARCHIVE_PATH}")
        return

    temporary_path = ARCHIVE_PATH.with_suffix(".tar.gz.part")
    temporary_path.unlink(missing_ok=True)

    print("Downloading A-OKVQA annotations")

    with requests.get(
        DATA_URL,
        stream=True,
        timeout=(30, 180),
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with temporary_path.open("wb") as file:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="A-OKVQA",
            )

            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    file.write(block)
                    progress.update(len(block))

            progress.close()

    temporary_path.replace(ARCHIVE_PATH)


def safe_extract_archive():
    existing_files = list(RAW_DIR.rglob("aokvqa_v1p0_val.json"))
    if existing_files:
        return existing_files[0]

    print("Extracting A-OKVQA annotations")

    raw_location = RAW_DIR.resolve()

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        for member in archive.getmembers():
            member_location = (RAW_DIR / member.name).resolve()

            if os.path.commonpath(
                [str(raw_location), str(member_location)]
            ) != str(raw_location):
                raise ValueError(
                    f"Unsafe path found in archive: {member.name}"
                )

        archive.extractall(RAW_DIR)

    extracted_files = list(RAW_DIR.rglob("aokvqa_v1p0_val.json"))

    if not extracted_files:
        raise FileNotFoundError(
            "The validation annotation was not found after extraction"
        )

    return extracted_files[0]


def read_annotations(annotation_path):
    with annotation_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("The A-OKVQA validation annotation is invalid")

    return records


def normalize_answer(answer):
    answer = str(answer).lower().strip()
    answer = answer.replace("\n", " ").replace("\t", " ")
    answer = answer.translate(
        str.maketrans({character: " " for character in string.punctuation})
    )

    cleaned_words = []

    for word in answer.split():
        if word in ARTICLES:
            continue
        cleaned_words.append(NUMBER_WORDS.get(word, word))

    return " ".join(cleaned_words)


def answer_information(record):
    choices = record.get("choices") or []
    correct_index = record.get("correct_choice_idx")

    if not isinstance(correct_index, int):
        return "", [], [], 0

    if correct_index < 0 or correct_index >= len(choices):
        return "", [], [], 0

    correct_answer = str(choices[correct_index]).strip()
    direct_answers = []

    for answer in record.get("direct_answers") or []:
        normalized = normalize_answer(answer)
        if normalized:
            direct_answers.append(normalized)

    normalized_correct = normalize_answer(correct_answer)
    accepted_answers = []

    for answer in [normalized_correct] + direct_answers:
        if answer and answer not in accepted_answers:
            accepted_answers.append(answer)

    counts = Counter(direct_answers)
    agreement = max(counts.values()) if counts else 0

    return correct_answer, accepted_answers, direct_answers, agreement


def extract_image_number(value):
    if value is None:
        return None

    text = str(value).strip()
    numbers = re.findall(r"\d+", text)

    if not numbers:
        return None

    try:
        return int(numbers[-1])
    except ValueError:
        return None


def find_vqa_image_ids():
    if not VQA_CORE_PATH.exists():
        raise FileNotFoundError(
            f"Required VQA core file was not found: {VQA_CORE_PATH}"
        )

    table = pd.read_parquet(VQA_CORE_PATH)
    image_columns = [
        column
        for column in table.columns
        if "image" in column.lower() and "id" in column.lower()
    ]

    if not image_columns:
        raise ValueError(
            "No image ID columns were found in the VQA core file"
        )

    image_ids = set()

    for column in image_columns:
        for value in table[column].dropna():
            image_id = extract_image_number(value)
            if image_id is not None:
                image_ids.add(image_id)

    if not image_ids:
        raise ValueError(
            "No VQA image IDs could be read for overlap checking"
        )

    return image_ids, image_columns


def build_candidates(records, blocked_image_ids):
    candidates = []
    excluded_overlap = 0
    invalid_records = 0
    seen_images = set()

    for record in records:
        image_id = extract_image_number(record.get("image_id"))
        question_id = str(record.get("question_id", "")).strip()
        question = str(record.get("question", "")).strip()

        correct_answer, accepted_answers, direct_answers, agreement = (
            answer_information(record)
        )

        if (
            image_id is None
            or not question_id
            or not question
            or not correct_answer
            or not accepted_answers
        ):
            invalid_records += 1
            continue

        if image_id in blocked_image_ids:
            excluded_overlap += 1
            continue

        if image_id in seen_images:
            continue

        choices = [
            str(choice).strip()
            for choice in record.get("choices") or []
        ]
        rationales = [
            str(rationale).strip()
            for rationale in record.get("rationales") or []
            if str(rationale).strip()
        ]

        candidates.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
                "correct_answer": correct_answer,
                "accepted_answers": accepted_answers,
                "direct_answers": direct_answers,
                "direct_answer_agreement": agreement,
                "choices": choices,
                "correct_choice_idx": int(
                    record.get("correct_choice_idx")
                ),
                "rationales": rationales,
            }
        )
        seen_images.add(image_id)

    randomizer = random.Random(SEED)
    randomizer.shuffle(candidates)

    counts = {
        "validation_records": len(records),
        "valid_unique_candidates": len(candidates),
        "excluded_vqa_image_overlap": excluded_overlap,
        "invalid_records": invalid_records,
    }

    return candidates, counts


def coco_image_url(image_id):
    filename = f"{image_id:012d}.jpg"
    return f"https://images.cocodataset.org/val2017/{filename}"


def valid_image(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_image(session, image_id, destination):
    if valid_image(destination):
        return True

    destination.unlink(missing_ok=True)
    image_url = coco_image_url(image_id)

    try:
        response = session.get(
            image_url,
            timeout=(20, 120),
            allow_redirects=True,
        )
        response.raise_for_status()

        if len(response.content) < 1000:
            return False

        with Image.open(BytesIO(response.content)) as image:
            image.load()
            image = ImageOps.exif_transpose(image)

            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(destination, format="JPEG", quality=95)

        return valid_image(destination)

    except Exception:
        destination.unlink(missing_ok=True)
        return False


def select_examples(candidates):
    selected = []
    failed_image_ids = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            )
        }
    )

    progress = tqdm(total=TARGET_SIZE, desc="Selected images")

    for candidate in candidates:
        if len(selected) >= TARGET_SIZE:
            break

        image_id = candidate["image_id"]
        image_name = f"{image_id:012d}.jpg"
        image_path = IMAGE_DIR / image_name

        if not download_image(session, image_id, image_path):
            failed_image_ids.append(image_id)
            continue

        selected.append(
            {
                "dataset": "aokvqa",
                "source_version": "v1p0",
                "source_split": "val",
                "question_id": candidate["question_id"],
                "image_id": image_id,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "question_en": candidate["question"],
                "answer_en": candidate["correct_answer"],
                "accepted_answers_json": json.dumps(
                    candidate["accepted_answers"],
                    ensure_ascii=False,
                ),
                "direct_answers_json": json.dumps(
                    candidate["direct_answers"],
                    ensure_ascii=False,
                ),
                "direct_answer_agreement": (
                    candidate["direct_answer_agreement"]
                ),
                "choices_json": json.dumps(
                    candidate["choices"],
                    ensure_ascii=False,
                ),
                "correct_choice_idx": candidate["correct_choice_idx"],
                "rationales_json": json.dumps(
                    candidate["rationales"],
                    ensure_ascii=False,
                ),
                "reasoning_category": "visual_commonsense",
                "image_source_url": coco_image_url(image_id),
            }
        )
        progress.update(1)

    progress.close()
    session.close()

    if len(selected) < TARGET_SIZE:
        raise RuntimeError(
            f"Only {len(selected)} A-OKVQA examples were prepared. "
            f"{TARGET_SIZE} are required. Check the internet connection "
            "and run the script again."
        )

    return selected, failed_image_ids


def save_selection(selected):
    table = pd.DataFrame(selected)

    table.insert(
        0,
        "benchmark_id",
        [
            f"aokvqa_{number:04d}"
            for number in range(1, len(table) + 1)
        ],
    )

    table.to_parquet(PARQUET_PATH, index=False)
    table.to_csv(CSV_PATH, index=False, encoding="utf-8")

    return table


def verify_selection(table, blocked_image_ids):
    if len(table) != TARGET_SIZE:
        raise ValueError(
            f"Expected {TARGET_SIZE} rows but found {len(table)}"
        )

    if table["question_id"].nunique() != TARGET_SIZE:
        raise ValueError("Repeated A-OKVQA question IDs were found")

    if table["image_id"].nunique() != TARGET_SIZE:
        raise ValueError("Repeated A-OKVQA images were found")

    overlap = set(table["image_id"]).intersection(blocked_image_ids)

    if overlap:
        raise ValueError(
            f"{len(overlap)} images overlap with the VQA core"
        )

    missing_images = []

    for relative_path in table["image_path"]:
        image_path = ROOT / relative_path

        if not valid_image(image_path):
            missing_images.append(relative_path)

    if missing_images:
        raise ValueError(
            f"{len(missing_images)} selected images are missing or invalid"
        )


def save_report(
    annotation_path,
    table,
    selection_counts,
    failed_image_ids,
    vqa_image_columns,
):
    agreement_counts = (
        table["direct_answer_agreement"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report = {
        "dataset": "A-OKVQA",
        "source_version": "v1p0",
        "source_split": "val",
        "source_url": DATA_URL,
        "annotation_path": annotation_path.relative_to(ROOT).as_posix(),
        "annotation_sha256": file_hash(annotation_path),
        "seed": SEED,
        "target_size": TARGET_SIZE,
        "selected_questions": int(len(table)),
        "selected_images": int(table["image_id"].nunique()),
        "reasoning_category": "visual_commonsense",
        "answer_language": "English",
        "failed_image_downloads": len(failed_image_ids),
        "failed_image_ids": failed_image_ids,
        "direct_answer_agreement_counts": {
            str(key): int(value)
            for key, value in agreement_counts.items()
        },
        "vqa_overlap_columns": vqa_image_columns,
        "selection_counts": selection_counts,
        "parquet_path": PARQUET_PATH.relative_to(ROOT).as_posix(),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "image_directory": IMAGE_DIR.relative_to(ROOT).as_posix(),
        "license": "See the official A-OKVQA and COCO licenses",
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report


def main():
    make_directories()
    download_archive()
    annotation_path = safe_extract_archive()
    records = read_annotations(annotation_path)

    blocked_image_ids, image_columns = find_vqa_image_ids()
    candidates, selection_counts = build_candidates(
        records,
        blocked_image_ids,
    )

    print()
    print("A-OKVQA validation")
    print(f"Questions: {len(records)}")
    print(
        "VQA core image IDs blocked: "
        f"{len(blocked_image_ids)}"
    )
    print(
        "A-OKVQA records excluded for overlap: "
        f"{selection_counts['excluded_vqa_image_overlap']}"
    )
    print(
        "Unique eligible candidates: "
        f"{selection_counts['valid_unique_candidates']}"
    )
    print("Downloading the selected image subset")

    selected, failed_image_ids = select_examples(candidates)
    table = save_selection(selected)
    verify_selection(table, blocked_image_ids)

    report = save_report(
        annotation_path,
        table,
        selection_counts,
        failed_image_ids,
        image_columns,
    )

    print()
    print("A-OKVQA selection")
    print(f"Questions: {report['selected_questions']}")
    print(f"Unique images: {report['selected_images']}")
    print(
        "VQA image overlap: "
        f"{len(set(table['image_id']).intersection(blocked_image_ids))}"
    )
    print(f"Failed image attempts: {report['failed_image_downloads']}")
    print(f"Parquet: {PARQUET_PATH.relative_to(ROOT)}")
    print(f"Images: {IMAGE_DIR.relative_to(ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()