import argparse
import json
import re
from pathlib import Path

import pandas as pd


dataset_name = "Tahsin-Mayeesha/vqa_bn"

needed_columns = [
    "question_id",
    "image_id",
    "question",
    "answer_bn",
    "answer_type",
    "question_type",
]


def download_bangla_data(cache_dir):
    from datasets import load_dataset

    print(f"Loading {dataset_name}")
    bangla_data = load_dataset(
        dataset_name,
        split="validation",
        cache_dir=str(cache_dir),
    )

    missing_columns = [
        column
        for column in needed_columns
        if column not in bangla_data.column_names
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing Bangla columns: {missing_text}")

    bangla_data = bangla_data.select_columns(needed_columns)
    bangla_table = bangla_data.to_pandas()
    bangla_table["question_id"] = bangla_table["question_id"].astype("int64")
    bangla_table["image_id"] = bangla_table["image_id"].astype("int64")

    return bangla_table


def inspect_bangla_data(bangla_table):
    duplicate_ids = int(bangla_table["question_id"].duplicated().sum())
    empty_questions = int(
        bangla_table["question"].fillna("").str.strip().eq("").sum()
    )
    empty_answers = int(
        bangla_table["answer_bn"].fillna("").str.strip().eq("").sum()
    )

    print()
    print("Bangla validation data")
    print(f"Rows: {len(bangla_table)}")
    print(f"Unique question IDs: {bangla_table['question_id'].nunique()}")
    print(f"Duplicate question IDs: {duplicate_ids}")
    print(f"Empty questions: {empty_questions}")
    print(f"Empty answers: {empty_answers}")

    if duplicate_ids:
        raise ValueError("Duplicate Bangla question IDs were found.")

    if len(bangla_table) != 150000:
        raise ValueError(
            f"Expected 150000 validation rows, found {len(bangla_table)}"
        )


def save_bangla_data(bangla_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bangla_table.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/huggingface_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/multilingual_alignment/"
            "vqa_bn_validation.parquet"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    bangla_table = download_bangla_data(arguments.cache_dir)
    inspect_bangla_data(bangla_table)
    save_bangla_data(bangla_table, arguments.output)


if __name__ == "__main__":
    main()