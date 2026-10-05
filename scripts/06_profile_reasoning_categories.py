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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/multilingual/vqa_en_bn_ready.parquet"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    pair_table = load_aligned_pairs(arguments.input)
    inspect_pair_pool(pair_table)


if __name__ == "__main__":
    main()