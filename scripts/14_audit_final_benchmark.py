import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "processed" / "benchmark"
CULTURAL_DIR = ROOT / "data" / "processed" / "cultural"
MANIFEST_DIR = ROOT / "data" / "manifests"

GENERAL_PATH = BENCHMARK_DIR / "general_benchmark_multilingual.parquet"
CULTURAL_PATH = CULTURAL_DIR / "cultural_benchmark_multilingual.parquet"
REPORT_PATH = MANIFEST_DIR / "final_benchmark_audit.json"

EXPECTED_GENERAL_GROUPS = {
    "vqa_v2": 4000,
    "xgqa": 300,
    "textvqa": 300,
    "aokvqa": 400,
}

EXPECTED_GENERAL_STIMULI = {
    "vqa_v2": 8000,
    "xgqa": 300,
    "textvqa": 300,
    "aokvqa": 400,
}

LANGUAGES = ["en", "bn", "hi", "ur", "en_bn_cs"]


def contains_bangla(text):
    return bool(re.search(r"[\u0980-\u09FF]", str(text)))


def contains_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", str(text)))


def contains_urdu(text):
    return bool(re.search(r"[\u0600-\u06FF]", str(text)))


def contains_latin(text):
    return bool(re.search(r"[A-Za-z]", str(text)))


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def count_missing_images(table):
    missing_by_source = {}
    available_by_source = {}

    for source, group in table.groupby("source"):
        missing = 0
        available = 0
        paths = group.drop_duplicates("stimulus_id")["image_path"]

        for value in paths:
            path_text = str(value).strip()

            if not path_text:
                missing += 1
                continue

            image_path = ROOT / path_text

            if image_path.exists() and image_path.stat().st_size > 0:
                available += 1
            else:
                missing += 1

        missing_by_source[source] = missing
        available_by_source[source] = available

    return missing_by_source, available_by_source


def language_failures(table):
    failures = {}

    checks = {
        "en": lambda text: contains_latin(text),
        "bn": lambda text: contains_bangla(text),
        "hi": lambda text: contains_devanagari(text),
        "ur": lambda text: contains_urdu(text),
        "en_bn_cs": lambda text: (
            contains_bangla(text) and contains_latin(text)
        ),
    }

    for language, check in checks.items():
        values = table.loc[
            table["language"] == language,
            "question",
        ]

        failures[language] = int(
            sum(not check(value) for value in values)
        )

    return failures


def audit_general(table):
    problems = []

    source_groups = (
        table.groupby("source")["group_id"].nunique().to_dict()
    )
    source_stimuli = (
        table.groupby("source")["stimulus_id"].nunique().to_dict()
    )

    if source_groups != EXPECTED_GENERAL_GROUPS:
        problems.append(
            f"Unexpected source group counts: {source_groups}"
        )

    if source_stimuli != EXPECTED_GENERAL_STIMULI:
        problems.append(
            f"Unexpected source stimulus counts: {source_stimuli}"
        )

    if table["group_id"].nunique() != 5000:
        problems.append("General benchmark does not contain 5,000 groups")

    if table["stimulus_id"].nunique() != 9000:
        problems.append(
            "General benchmark does not contain 9,000 stimuli"
        )

    if len(table) != 45000:
        problems.append(
            f"Expected 45,000 general rows, found {len(table)}"
        )

    if table["variant_id"].duplicated().any():
        problems.append("Duplicate general variant IDs were found")

    expected_language_counts = {
        language: 9000 for language in LANGUAGES
    }
    actual_language_counts = table["language"].value_counts().to_dict()

    if actual_language_counts != expected_language_counts:
        problems.append(
            f"Unexpected language counts: {actual_language_counts}"
        )

    for column in ["question", "answer_en", "image_path"]:
        missing = int(
            table[column].fillna("").astype(str).str.strip().eq("").sum()
        )
        if missing:
            problems.append(f"{column} has {missing} empty values")

    script_failures = language_failures(table)

    for language, count in script_failures.items():
        if count:
            problems.append(
                f"{language} has {count} script-check failures"
            )

    return problems, source_groups, source_stimuli, script_failures


def audit_cultural(table):
    problems = []

    if table["group_id"].nunique() != 500:
        problems.append(
            "Cultural benchmark does not contain 500 groups"
        )

    if len(table) != 2500:
        problems.append(
            f"Expected 2,500 cultural rows, found {len(table)}"
        )

    if table["variant_id"].duplicated().any():
        problems.append("Duplicate cultural variant IDs were found")

    expected_language_counts = {
        language: 500 for language in LANGUAGES
    }
    actual_language_counts = table["language"].value_counts().to_dict()

    if actual_language_counts != expected_language_counts:
        problems.append(
            f"Unexpected cultural language counts: "
            f"{actual_language_counts}"
        )

    for column in ["question", "answer_en", "image_path"]:
        missing = int(
            table[column].fillna("").astype(str).str.strip().eq("").sum()
        )
        if missing:
            problems.append(f"Cultural {column} has {missing} empty values")

    script_failures = language_failures(table)

    for language, count in script_failures.items():
        if count:
            problems.append(
                f"Cultural {language} has "
                f"{count} script-check failures"
            )

    return problems, script_failures


def audit_pair_structure(table):
    problems = []
    vqa = table[table["source"] == "vqa_v2"]
    group_sizes = vqa.groupby("group_id")["stimulus_id"].nunique()

    incorrect = int((group_sizes != 2).sum())

    if incorrect:
        problems.append(
            f"{incorrect} VQA groups do not have exactly two stimuli"
        )

    return problems, incorrect


def main():
    if not GENERAL_PATH.exists() or not CULTURAL_PATH.exists():
        raise FileNotFoundError(
            "Run scripts 12 and 13 before the final audit"
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    general = pd.read_parquet(GENERAL_PATH)
    cultural = pd.read_parquet(CULTURAL_PATH)

    general_problems, source_groups, source_stimuli, general_scripts = (
        audit_general(general)
    )
    cultural_problems, cultural_scripts = audit_cultural(cultural)
    pair_problems, incorrect_pairs = audit_pair_structure(general)

    general_missing, general_available = count_missing_images(general)
    cultural_missing, cultural_available = count_missing_images(
        cultural.assign(source="banglaverse")
    )

    critical_problems = (
        general_problems
        + cultural_problems
        + pair_problems
    )

    non_vqa_missing = {
        source: count
        for source, count in general_missing.items()
        if source != "vqa_v2" and count > 0
    }

    if non_vqa_missing:
        critical_problems.append(
            f"Missing downloaded images: {non_vqa_missing}"
        )

    if cultural_missing.get("banglaverse", 0):
        critical_problems.append(
            "Some BanglaVerse images are missing"
        )

    report = {
        "status": "passed" if not critical_problems else "failed",
        "general_groups": int(general["group_id"].nunique()),
        "general_stimuli": int(general["stimulus_id"].nunique()),
        "general_variants": int(len(general)),
        "cultural_groups": int(cultural["group_id"].nunique()),
        "cultural_variants": int(len(cultural)),
        "source_group_counts": source_groups,
        "source_stimulus_counts": source_stimuli,
        "general_language_counts": (
            general["language"].value_counts().to_dict()
        ),
        "cultural_language_counts": (
            cultural["language"].value_counts().to_dict()
        ),
        "general_script_failures": general_scripts,
        "cultural_script_failures": cultural_scripts,
        "incorrect_vqa_pair_groups": incorrect_pairs,
        "available_images_by_source": general_available,
        "missing_images_by_source": general_missing,
        "available_cultural_images": cultural_available,
        "missing_cultural_images": cultural_missing,
        "general_sha256": file_hash(GENERAL_PATH),
        "cultural_sha256": file_hash(CULTURAL_PATH),
        "critical_problems": critical_problems,
        "notes": [
            "Missing local VQA v2 images are allowed during local preparation",
            "VQA images must be attached or downloaded in the GPU environment",
            "Code-switched questions require human review before experiments",
            "BanglaVerse remains a separate external cultural stress test",
        ],
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Final benchmark audit")
    print(f"Status: {report['status'].upper()}")
    print(f"General groups: {report['general_groups']}")
    print(f"General stimuli: {report['general_stimuli']}")
    print(f"General variants: {report['general_variants']}")
    print(f"Cultural groups: {report['cultural_groups']}")
    print(f"Cultural variants: {report['cultural_variants']}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")

    if critical_problems:
        print()
        print("Problems")
        for problem in critical_problems:
            print(f"- {problem}")
        raise ValueError("The final benchmark audit failed")


if __name__ == "__main__":
    main()