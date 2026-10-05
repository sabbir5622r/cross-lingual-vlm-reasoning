import argparse
import json
import re
from pathlib import Path

import pandas as pd


required_columns = [
    "pair_id",
    "question_id_a",
    "question_id_b",
    "image_id_a",
    "image_id_b",
    "question",
    "answer_a",
    "answer_b",
    "answer_type_a",
    "bangla_question",
]


def load_aligned_pairs(input_path):
    if not input_path.exists():
        raise FileNotFoundError(f"Aligned data not found: {input_path}")

    pair_table = pd.read_parquet(input_path)

    missing_columns = [
        column
        for column in required_columns
        if column not in pair_table.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing columns: {missing_text}")

    return pair_table


def inspect_pair_pool(pair_table):
    all_question_ids = pd.concat(
        [
            pair_table["question_id_a"],
            pair_table["question_id_b"],
        ],
        ignore_index=True,
    )

    all_image_ids = pd.concat(
        [
            pair_table["image_id_a"],
            pair_table["image_id_b"],
        ],
        ignore_index=True,
    )

    duplicate_pair_ids = int(pair_table["pair_id"].duplicated().sum())
    repeated_question_uses = len(all_question_ids) - all_question_ids.nunique()
    repeated_image_uses = len(all_image_ids) - all_image_ids.nunique()

    empty_english = int(
        pair_table["question"].fillna("").str.strip().eq("").sum()
    )
    empty_bangla = int(
        pair_table["bangla_question"].fillna("").str.strip().eq("").sum()
    )

    summary = {
        "pairs": len(pair_table),
        "unique_pair_ids": int(pair_table["pair_id"].nunique()),
        "unique_question_ids": int(all_question_ids.nunique()),
        "unique_image_ids": int(all_image_ids.nunique()),
        "duplicate_pair_ids": duplicate_pair_ids,
        "repeated_question_uses": int(repeated_question_uses),
        "repeated_image_uses": int(repeated_image_uses),
        "empty_english_questions": empty_english,
        "empty_bangla_questions": empty_bangla,
    }

    print("Aligned pair pool")
    print(f"Pairs: {summary['pairs']}")
    print(f"Unique pair IDs: {summary['unique_pair_ids']}")
    print(f"Unique question IDs: {summary['unique_question_ids']}")
    print(f"Unique image IDs: {summary['unique_image_ids']}")
    print(f"Repeated question uses: {summary['repeated_question_uses']}")
    print(f"Repeated image uses: {summary['repeated_image_uses']}")
    print(f"Empty English questions: {empty_english}")
    print(f"Empty Bangla questions: {empty_bangla}")

    if duplicate_pair_ids:
        raise ValueError("Duplicate pair IDs were found.")

    if empty_english or empty_bangla:
        raise ValueError("Empty aligned questions were found.")

    return summary

reasoning_patterns = {
    "ocr": [
        r"\bwhat does .* say\b",
        r"\bwhat do .* say\b",
        r"\bwhat is written\b",
        r"\bwhat word\b",
        r"\bwhat letter\b",
        r"\bcan you read\b",
        r"\bsign say\b",
        r"\btext\b",
        r"\blicense plate\b",
    ],
    "multi_step": [
        r"\bbased on\b",
        r"\bjudging from\b",
        r"\bhow can you tell\b",
        r"\bwhat will happen\b",
        r"\bwhat happened before\b",
        r"\bwhat happened after\b",
        r"\bwhat might happen\b",
    ],
    "spatial": [
        r"\bwhere\b",
        r"\bleft\b",
        r"\bright\b",
        r"\bbehind\b",
        r"\bin front of\b",
        r"\bnext to\b",
        r"\bunder\b",
        r"\babove\b",
        r"\bbetween\b",
        r"\bnear\b",
        r"\bwhich side\b",
        r"\blocated\b",
    ],
    "relation": [
        r"\bholding\b",
        r"\bwearing\b",
        r"\briding\b",
        r"\bsitting on\b",
        r"\bstanding on\b",
        r"\blooking at\b",
        r"\bcarrying\b",
        r"\busing\b",
        r"\battached to\b",
        r"\bcovered by\b",
    ],
    "activity": [
        r"\bdoing\b",
        r"\bhappening\b",
        r"\bplaying\b",
        r"\bactivity\b",
        r"\bsport\b",
        r"\bworking\b",
        r"\bwalking\b",
        r"\brunning\b",
        r"\beating\b",
        r"\bdrinking\b",
        r"\bcooking\b",
    ],
    "attribute": [
        r"\bcolor\b",
        r"\bcolour\b",
        r"\bshape\b",
        r"\bmade of\b",
        r"\bold\b",
        r"\byoung\b",
        r"\bclean\b",
        r"\bdirty\b",
        r"\bweather\b",
        r"\bpattern\b",
        r"\bmaterial\b",
    ],
    "visual_commonsense": [
        r"\bwhy\b",
        r"\bpurpose\b",
        r"\blikely\b",
        r"\bprobably\b",
        r"\bsafe\b",
        r"\bcould\b",
        r"\bwould\b",
        r"\bshould\b",
        r"\bused for\b",
        r"\bneed to\b",
    ],
    "object_recognition": [
        r"^what is\b",
        r"^what are\b",
        r"^who is\b",
        r"\bwhich animal\b",
        r"\bwhat animal\b",
        r"\bwhat object\b",
        r"\bwhat type of\b",
        r"\bwhat kind of\b",
    ],
}


def matches_any(text, patterns):
    return any(re.search(pattern, text) for pattern in patterns)


def assign_reasoning_hint(question, question_type, answer_type):
    question_text = str(question).lower().strip()
    type_text = str(question_type).lower().strip()
    combined_text = f"{question_text} {type_text}"

    if str(answer_type).lower() == "number":
        return "counting"

    if matches_any(combined_text, reasoning_patterns["ocr"]):
        return "ocr"

    if matches_any(combined_text, reasoning_patterns["multi_step"]):
        return "multi_step"

    if matches_any(combined_text, reasoning_patterns["spatial"]):
        return "spatial"

    if matches_any(combined_text, reasoning_patterns["relation"]):
        return "relation"

    if matches_any(combined_text, reasoning_patterns["activity"]):
        return "activity"

    if matches_any(combined_text, reasoning_patterns["attribute"]):
        return "attribute"

    if matches_any(
        combined_text,
        reasoning_patterns["visual_commonsense"],
    ):
        return "visual_commonsense"

    if matches_any(
        combined_text,
        reasoning_patterns["object_recognition"],
    ):
        return "object_recognition"

    if str(answer_type).lower() == "yes/no":
        return "binary_perception"

    return "other"


def add_reasoning_hints(pair_table):
    categorized = pair_table.copy()

    question_type_column = (
        "question_type_a"
        if "question_type_a" in categorized.columns
        else "bangla_question_type_a"
    )

    categorized["reasoning_hint"] = [
        assign_reasoning_hint(question, question_type, answer_type)
        for question, question_type, answer_type in zip(
            categorized["question"],
            categorized[question_type_column],
            categorized["answer_type_a"],
        )
    ]

    return categorized


def show_category_summary(categorized):
    category_counts = categorized["reasoning_hint"].value_counts()

    print()
    print("Provisional reasoning categories")

    for category_name, count in category_counts.items():
        share = count / len(categorized)
        print(f"{category_name}: {count} ({share:.2%})")

    print()
    print("Answer types by category")
    category_table = pd.crosstab(
        categorized["reasoning_hint"],
        categorized["answer_type_a"],
    )
    print(category_table.to_string())


def save_categorized_data(categorized, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categorized.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    categorized.to_csv(csv_path, index=False, encoding="utf-8")

    print()
    print(f"Categorized Parquet: {output_path}")
    print(f"Categorized CSV: {csv_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/multilingual/vqa_en_bn_ready.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/interim/multilingual/"
            "vqa_en_bn_categorized.parquet"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    pair_table = load_aligned_pairs(arguments.input)
    inspect_pair_pool(pair_table)

    categorized = add_reasoning_hints(pair_table)
    show_category_summary(categorized)
    save_categorized_data(categorized, arguments.output)


if __name__ == "__main__":
    main()