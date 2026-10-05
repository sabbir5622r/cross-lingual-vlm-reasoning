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

def read_pair_table(pair_path):
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair table not found: {pair_path}")

    pair_table = pd.read_parquet(pair_path)

    needed_pair_columns = [
        "pair_id",
        "question_id_a",
        "question_id_b",
        "image_id_a",
        "image_id_b",
    ]

    missing_columns = [
        column
        for column in needed_pair_columns
        if column not in pair_table.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing pair columns: {missing_text}")

    return pair_table


def tidy_question(question):
    return " ".join(str(question).strip().split())


def align_bangla_questions(pair_table, bangla_table):
    first_side = bangla_table.rename(
        columns={
            "question_id": "bangla_question_id_a",
            "image_id": "bangla_image_id_a",
            "question": "bangla_question_a",
            "answer_bn": "bangla_answer_a",
            "answer_type": "bangla_answer_type_a",
            "question_type": "bangla_question_type_a",
        }
    )

    second_side = bangla_table.rename(
        columns={
            "question_id": "bangla_question_id_b",
            "image_id": "bangla_image_id_b",
            "question": "bangla_question_b",
            "answer_bn": "bangla_answer_b",
            "answer_type": "bangla_answer_type_b",
            "question_type": "bangla_question_type_b",
        }
    )

    aligned = pair_table.merge(
        first_side,
        left_on="question_id_a",
        right_on="bangla_question_id_a",
        how="inner",
        validate="many_to_one",
    )

    aligned = aligned.merge(
        second_side,
        left_on="question_id_b",
        right_on="bangla_question_id_b",
        how="inner",
        validate="many_to_one",
    )

    aligned["same_bangla_question"] = [
        tidy_question(first_question) == tidy_question(second_question)
        for first_question, second_question in zip(
            aligned["bangla_question_a"],
            aligned["bangla_question_b"],
        )
    ]

    aligned["image_id_match_a"] = (
        aligned["image_id_a"] == aligned["bangla_image_id_a"]
    )
    aligned["image_id_match_b"] = (
        aligned["image_id_b"] == aligned["bangla_image_id_b"]
    )

    return aligned


def save_aligned_table(aligned_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_table.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    aligned_table.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Parquet: {output_path}")
    print(f"CSV: {csv_path}")

def has_bangla_text(text):
    return bool(re.search(r"[\u0980-\u09FF]", str(text)))


def choose_ready_pairs(clean_aligned):
    valid_questions = (
        clean_aligned["bangla_question_a"].fillna("").str.strip().ne("")
        & clean_aligned["bangla_question_b"].fillna("").str.strip().ne("")
    )

    valid_answers = (
        clean_aligned["bangla_answer_a"].fillna("").str.strip().ne("")
        & clean_aligned["bangla_answer_b"].fillna("").str.strip().ne("")
    )

    contains_bangla = (
        clean_aligned["bangla_question_a"].map(has_bangla_text)
        & clean_aligned["bangla_question_b"].map(has_bangla_text)
    )

    ready_mask = (
        clean_aligned["same_bangla_question"]
        & clean_aligned["image_id_match_a"]
        & clean_aligned["image_id_match_b"]
        & valid_questions
        & valid_answers
        & contains_bangla
    )

    ready_pairs = clean_aligned.loc[ready_mask].copy()
    ready_pairs["bangla_question"] = ready_pairs["bangla_question_a"]
    ready_pairs["translation_source"] = dataset_name
    ready_pairs["translation_verified"] = False
    ready_pairs = ready_pairs.reset_index(drop=True)

    return ready_pairs


def make_alignment_report(all_aligned, clean_aligned, ready_pairs):
    from datetime import datetime, timezone

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bangla_source": dataset_name,
        "translation_method": "automatic_google_translate",
        "full_aligned_pairs": len(all_aligned),
        "full_matching_bangla_questions": int(
            all_aligned["same_bangla_question"].sum()
        ),
        "expected_full_aligned_pairs": 46620,
        "expected_matching_bangla_questions": 46276,
        "full_count_reproduced": len(all_aligned) == 46620,
        "matching_count_reproduced": int(
            all_aligned["same_bangla_question"].sum()
        ) == 46276,
        "clean_aligned_pairs": len(clean_aligned),
        "clean_matching_bangla_questions": int(
            clean_aligned["same_bangla_question"].sum()
        ),
        "image_mismatch_a": int(
            (~clean_aligned["image_id_match_a"]).sum()
        ),
        "image_mismatch_b": int(
            (~clean_aligned["image_id_match_b"]).sum()
        ),
        "ready_pairs": len(ready_pairs),
        "translation_verified": False,
    }


def save_alignment_report(report, report_path):
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
        "--bangla-data",
        type=Path,
        default=Path(
            "data/raw/multilingual_alignment/"
            "vqa_bn_validation.parquet"
        ),
    )
    parser.add_argument(
        "--all-pairs",
        type=Path,
        default=Path(
            "data/processed/vqa_v2/"
            "vqa_v2_complementary_pairs.parquet"
        ),
    )
    parser.add_argument(
        "--clean-pairs",
        type=Path,
        default=Path(
            "data/processed/vqa_v2/"
            "vqa_v2_visual_candidates.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/multilingual"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/manifests/vqa_en_bn_alignment_report.json"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.bangla_data.exists():
        bangla_table = pd.read_parquet(arguments.bangla_data)
    else:
        bangla_table = download_bangla_data(arguments.cache_dir)
        inspect_bangla_data(bangla_table)
        save_bangla_data(bangla_table, arguments.bangla_data)

    all_pairs = read_pair_table(arguments.all_pairs)
    clean_pairs = read_pair_table(arguments.clean_pairs)

    all_aligned = align_bangla_questions(all_pairs, bangla_table)
    clean_aligned = align_bangla_questions(clean_pairs, bangla_table)
    ready_pairs = choose_ready_pairs(clean_aligned)

    save_aligned_table(
        all_aligned,
        arguments.output_dir / "vqa_en_bn_all_pairs.parquet",
    )
    save_aligned_table(
        clean_aligned,
        arguments.output_dir / "vqa_en_bn_clean_pairs.parquet",
    )
    save_aligned_table(
        ready_pairs,
        arguments.output_dir / "vqa_en_bn_ready.parquet",
    )

    report = make_alignment_report(
        all_aligned,
        clean_aligned,
        ready_pairs,
    )
    save_alignment_report(report, arguments.report)

    print()
    print("Final alignment")
    print(f"Full aligned pairs: {len(all_aligned)}")
    print(
        f"Full matching Bangla questions: "
        f"{int(all_aligned['same_bangla_question'].sum())}"
    )
    print(f"Clean aligned pairs: {len(clean_aligned)}")
    print(f"Ready English-Bangla pairs: {len(ready_pairs)}")
    print(
        f"Earlier counts reproduced: "
        f"{report['full_count_reproduced'] and report['matching_count_reproduced']}"
    )


if __name__ == "__main__":
    main()