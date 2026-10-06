import hashlib
import json
import random
import re
import shutil
import string
from collections import Counter
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageOps
from tqdm import tqdm


SEED = 2027
TARGET_SIZE = 300
MINIMUM_ANSWER_VOTES = 3

ANNOTATION_URL = (
    "https://dl.fbaipublicfiles.com/textvqa/data/"
    "TextVQA_0.5.1_val.json"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "textvqa"
IMAGE_DIR = RAW_DIR / "images"
BENCHMARK_DIR = ROOT / "data" / "processed" / "benchmark"
MANIFEST_DIR = ROOT / "data" / "manifests"

ANNOTATION_PATH = RAW_DIR / "TextVQA_0.5.1_val.json"
PARQUET_PATH = BENCHMARK_DIR / "textvqa_300.parquet"
CSV_PATH = BENCHMARK_DIR / "textvqa_300.csv"
REPORT_PATH = MANIFEST_DIR / "textvqa_selection_report.json"

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

OCR_TERMS = {
    "say",
    "says",
    "said",
    "sign",
    "signed",
    "written",
    "write",
    "word",
    "words",
    "letter",
    "letters",
    "number",
    "numbers",
    "name",
    "named",
    "brand",
    "label",
    "logo",
    "text",
    "title",
    "price",
    "phone",
    "website",
    "store",
    "shirt",
    "book",
    "building",
    "advertisement",
    "poster",
    "license",
    "plate",
}


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


def download_annotation():
    if ANNOTATION_PATH.exists() and ANNOTATION_PATH.stat().st_size > 100000:
        print(f"Using existing annotation: {ANNOTATION_PATH}")
        return

    temporary_path = ANNOTATION_PATH.with_suffix(".json.part")
    headers = {"User-Agent": "Mozilla/5.0"}

    print("Downloading TextVQA validation annotations")
    with requests.get(
        ANNOTATION_URL,
        stream=True,
        timeout=(30, 120),
        headers=headers,
    ) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with temporary_path.open("wb") as file:
            progress = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc="Annotations",
            )
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    file.write(block)
                    progress.update(len(block))
            progress.close()

    temporary_path.replace(ANNOTATION_PATH)


def read_annotation():
    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise ValueError("The TextVQA annotation format was not recognized")

    if not records:
        raise ValueError("The TextVQA validation annotation is empty")

    return records, payload


def normalize_answer(answer):
    answer = str(answer).lower().strip()
    answer = answer.replace("\n", " ").replace("\t", " ")
    answer = answer.translate(
        str.maketrans({character: " " for character in string.punctuation})
    )
    words = []

    for word in answer.split():
        if word in ARTICLES:
            continue
        words.append(NUMBER_WORDS.get(word, word))

    return " ".join(words)


def collect_answers(record):
    raw_answers = record.get("answers") or []
    answers = []

    for answer in raw_answers:
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        normalized = normalize_answer(answer)
        if normalized:
            answers.append(normalized)

    return answers


def summarize_answers(record):
    answers = collect_answers(record)

    if not answers:
        return "", 0, []

    counts = Counter(answers)
    majority_answer, votes = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]

    return majority_answer, votes, answers


def text_value(record, key):
    value = record.get(key)
    if value is None:
        return ""
    return str(value).strip()


def image_urls(record):
    urls = []

    for key in ["flickr_300k_url", "flickr_original_url"]:
        value = text_value(record, key)
        if value.startswith(("http://", "https://")) and value not in urls:
            urls.append(value)

    return urls


def question_score(record, votes, answer):
    question = text_value(record, "question").lower()
    words = set(re.findall(r"[a-z0-9]+", question))
    ocr_hits = len(words.intersection(OCR_TERMS))

    score = votes * 20
    score += min(ocr_hits, 4) * 10
    score += min(len(question.split()), 20)
    score += min(len(answer.split()), 4) * 2

    if question.startswith(("what", "which", "where", "who", "how")):
        score += 4
    if any(term in question for term in ["what does", "what is written", "what word"]):
        score += 12

    return score


def build_candidates(records):
    candidates = []
    rejected_without_answers = 0
    rejected_low_agreement = 0
    rejected_without_urls = 0

    for record in records:
        answer, votes, answers = summarize_answers(record)

        if not answer:
            rejected_without_answers += 1
            continue

        if votes < MINIMUM_ANSWER_VOTES:
            rejected_low_agreement += 1
            continue

        urls = image_urls(record)
        if not urls:
            rejected_without_urls += 1
            continue

        image_id = text_value(record, "image_id")
        question = text_value(record, "question")
        question_id = text_value(record, "question_id")

        if not image_id or not question or not question_id:
            continue

        candidates.append(
            {
                "record": record,
                "image_id": image_id,
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "answer_votes": votes,
                "answers": answers,
                "urls": urls,
                "score": question_score(record, votes, answer),
            }
        )

    randomizer = random.Random(SEED)
    randomizer.shuffle(candidates)
    candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["answer_votes"],
            item["question_id"],
        )
    )

    best_per_image = {}
    for candidate in candidates:
        if candidate["image_id"] not in best_per_image:
            best_per_image[candidate["image_id"]] = candidate

    unique_candidates = list(best_per_image.values())
    unique_candidates.sort(
        key=lambda item: (
            -item["score"],
            -item["answer_votes"],
            item["question_id"],
        )
    )

    preparation_counts = {
        "usable_questions": len(candidates),
        "unique_candidate_images": len(unique_candidates),
        "rejected_without_answers": rejected_without_answers,
        "rejected_low_agreement": rejected_low_agreement,
        "rejected_without_urls": rejected_without_urls,
    }

    return unique_candidates, preparation_counts


def valid_existing_image(path):
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def fetch_image(session, urls, destination):
    if valid_existing_image(destination):
        return True, "existing"

    destination.unlink(missing_ok=True)

    for url in urls:
        try:
            response = session.get(
                url,
                timeout=(20, 90),
                allow_redirects=True,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if content_type and "image" not in content_type:
                continue

            if len(response.content) < 1000:
                continue

            with Image.open(BytesIO(response.content)) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(destination, format="JPEG", quality=95)

            if valid_existing_image(destination):
                return True, url
        except Exception:
            destination.unlink(missing_ok=True)

    return False, ""


def select_and_download(candidates):
    selected = []
    failed_images = []
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

        image_name = f"{candidate['image_id']}.jpg"
        image_path = IMAGE_DIR / image_name
        success, used_url = fetch_image(
            session,
            candidate["urls"],
            image_path,
        )

        if not success:
            failed_images.append(candidate["image_id"])
            continue

        record = candidate["record"]
        relative_image_path = image_path.relative_to(ROOT).as_posix()

        selected.append(
            {
                "dataset": "textvqa",
                "source_version": "0.5.1",
                "source_split": "validation",
                "question_id": candidate["question_id"],
                "image_id": candidate["image_id"],
                "image_path": relative_image_path,
                "question_en": candidate["question"],
                "answer_en": candidate["answer"],
                "answer_votes": candidate["answer_votes"],
                "answers_json": json.dumps(
                    candidate["answers"],
                    ensure_ascii=False,
                ),
                "reasoning_category": "ocr",
                "selection_score": candidate["score"],
                "image_source_url": used_url,
                "flickr_300k_url": text_value(
                    record,
                    "flickr_300k_url",
                ),
                "flickr_original_url": text_value(
                    record,
                    "flickr_original_url",
                ),
            }
        )
        progress.update(1)

    progress.close()
    session.close()

    if len(selected) < TARGET_SIZE:
        raise RuntimeError(
            f"Only {len(selected)} TextVQA images could be downloaded. "
            f"{TARGET_SIZE} are required. Check the internet connection "
            "and run the script again."
        )

    return selected, failed_images


def save_selection(selected):
    table = pd.DataFrame(selected)
    table = table.sort_values(
        by=["selection_score", "question_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    table.insert(
        0,
        "benchmark_id",
        [f"textvqa_{number:04d}" for number in range(1, len(table) + 1)],
    )

    table.to_parquet(PARQUET_PATH, index=False)
    table.to_csv(CSV_PATH, index=False, encoding="utf-8")

    return table


def remove_unused_images(table):
    selected_names = {
        Path(path).name
        for path in table["image_path"].tolist()
    }

    removed = 0
    for image_path in IMAGE_DIR.glob("*.jpg"):
        if image_path.name not in selected_names:
            image_path.unlink()
            removed += 1

    return removed


def save_report(records, table, preparation_counts, failed_images, removed):
    answer_vote_counts = (
        table["answer_votes"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    report = {
        "dataset": "TextVQA",
        "source_version": "0.5.1",
        "source_split": "validation",
        "source_annotation_url": ANNOTATION_URL,
        "annotation_path": ANNOTATION_PATH.relative_to(ROOT).as_posix(),
        "annotation_sha256": file_hash(ANNOTATION_PATH),
        "annotation_questions": len(records),
        "seed": SEED,
        "target_size": TARGET_SIZE,
        "minimum_answer_votes": MINIMUM_ANSWER_VOTES,
        "selected_questions": int(len(table)),
        "selected_images": int(table["image_id"].nunique()),
        "answer_vote_counts": {
            str(key): int(value)
            for key, value in answer_vote_counts.items()
        },
        "failed_image_downloads": len(failed_images),
        "failed_image_ids": failed_images,
        "unused_images_removed": removed,
        "preparation_counts": preparation_counts,
        "parquet_path": PARQUET_PATH.relative_to(ROOT).as_posix(),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "image_directory": IMAGE_DIR.relative_to(ROOT).as_posix(),
        "answer_language": "English",
        "reasoning_category": "ocr",
        "license": "CC BY 4.0",
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return report


def verify_selection(table):
    if len(table) != TARGET_SIZE:
        raise ValueError(
            f"Expected {TARGET_SIZE} rows but found {len(table)}"
        )

    if table["image_id"].nunique() != TARGET_SIZE:
        raise ValueError("The TextVQA selection is not image-disjoint")

    if table["question_id"].nunique() != TARGET_SIZE:
        raise ValueError("Repeated TextVQA question IDs were found")

    missing_images = []
    for relative_path in table["image_path"]:
        image_path = ROOT / relative_path
        if not valid_existing_image(image_path):
            missing_images.append(relative_path)

    if missing_images:
        raise ValueError(
            f"{len(missing_images)} selected images are missing or invalid"
        )


def main():
    make_directories()
    download_annotation()
    records, _ = read_annotation()

    print()
    print("TextVQA validation")
    print(f"Questions: {len(records)}")

    candidates, preparation_counts = build_candidates(records)

    print(f"Usable questions: {preparation_counts['usable_questions']}")
    print(
        "Unique candidate images: "
        f"{preparation_counts['unique_candidate_images']}"
    )
    print("Downloading the selected image subset")

    selected, failed_images = select_and_download(candidates)
    table = save_selection(selected)
    removed = remove_unused_images(table)
    verify_selection(table)
    report = save_report(
        records,
        table,
        preparation_counts,
        failed_images,
        removed,
    )

    print()
    print("TextVQA selection")
    print(f"Questions: {report['selected_questions']}")
    print(f"Unique images: {report['selected_images']}")
    print(f"Failed image attempts: {report['failed_image_downloads']}")
    print(f"Parquet: {PARQUET_PATH.relative_to(ROOT)}")
    print(f"Images: {IMAGE_DIR.relative_to(ROOT)}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()