import argparse
import json
from pathlib import Path


def read_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def load_vqa_data(input_dir):
    question_path = input_dir / "v2_OpenEnded_mscoco_val2014_questions.json"
    annotation_path = input_dir / "v2_mscoco_val2014_annotations.json"
    pair_path = input_dir / "v2_mscoco_val2014_complementary_pairs.json"

    questions = read_json(question_path).get("questions", [])
    annotations = read_json(annotation_path).get("annotations", [])
    pairs = read_json(pair_path)

    return questions, annotations, pairs


def check_input_counts(questions, annotations, pairs):
    counts = {
        "questions": len(questions),
        "annotations": len(annotations),
        "pairs": len(pairs),
    }

    print("VQA input files")
    print(f"Questions: {counts['questions']}")
    print(f"Annotations: {counts['annotations']}")
    print(f"Complementary pairs: {counts['pairs']}")

    expected_counts = {
        "questions": 214354,
        "annotations": 214354,
        "pairs": 95144,
    }

    wrong_counts = [
        name
        for name, expected in expected_counts.items()
        if counts[name] != expected
    ]

    if wrong_counts:
        wrong_text = ", ".join(wrong_counts)
        raise ValueError(f"Unexpected counts found in: {wrong_text}")

    print("Input counts passed.")
    return counts

def tidy_text(text):
    return " ".join(str(text).strip().lower().split())


def make_question_lookup(questions):
    return {
        int(question["question_id"]): question
        for question in questions
    }


def make_annotation_lookup(annotations):
    return {
        int(annotation["question_id"]): annotation
        for annotation in annotations
    }


def build_pair_rows(questions, annotations, pairs):
    question_lookup = make_question_lookup(questions)
    annotation_lookup = make_annotation_lookup(annotations)
    rows = []

    for pair_number, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Invalid pair at position {pair_number}")

        first_id = int(pair[0])
        second_id = int(pair[1])

        if first_id not in question_lookup or second_id not in question_lookup:
            raise KeyError(f"Question missing for pair {pair_number}")

        if first_id not in annotation_lookup or second_id not in annotation_lookup:
            raise KeyError(f"Annotation missing for pair {pair_number}")

        first_question = question_lookup[first_id]
        second_question = question_lookup[second_id]
        first_annotation = annotation_lookup[first_id]
        second_annotation = annotation_lookup[second_id]

        first_text = first_question["question"]
        second_text = second_question["question"]
        first_answer = first_annotation["multiple_choice_answer"]
        second_answer = second_annotation["multiple_choice_answer"]

        rows.append(
            {
                "pair_id": f"vqa_val_{pair_number:06d}",
                "question_id_a": first_id,
                "question_id_b": second_id,
                "image_id_a": int(first_question["image_id"]),
                "image_id_b": int(second_question["image_id"]),
                "question_a": first_text,
                "question_b": second_text,
                "answer_a": first_answer,
                "answer_b": second_answer,
                "question_type_a": first_annotation.get("question_type"),
                "question_type_b": second_annotation.get("question_type"),
                "answer_type_a": first_annotation.get("answer_type"),
                "answer_type_b": second_annotation.get("answer_type"),
                "same_question": tidy_text(first_text) == tidy_text(second_text),
                "answer_changed": tidy_text(first_answer) != tidy_text(second_answer),
                "source": "vqa_v2_val2014",
            }
        )

    return rows


def save_pair_table(rows, output_dir):
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    pair_table = pd.DataFrame(rows)
    parquet_path = output_dir / "vqa_v2_complementary_pairs.parquet"
    csv_path = output_dir / "vqa_v2_complementary_pairs.csv"

    pair_table.to_parquet(parquet_path, index=False)
    pair_table.to_csv(csv_path, index=False, encoding="utf-8")

    print()
    print(f"Rows saved: {len(pair_table)}")
    print(f"Parquet: {parquet_path}")
    print(f"CSV: {csv_path}")

    return pair_table

def summarize_pair_table(pair_table):
    from collections import Counter

    same_question_count = int(pair_table["same_question"].sum())
    changed_answer_count = int(pair_table["answer_changed"].sum())
    total_pairs = len(pair_table)

    question_types = Counter(
        pair_table["question_type_a"].dropna().astype(str)
    )
    answer_types = Counter(
        pair_table["answer_type_a"].dropna().astype(str)
    )

    summary = {
        "pair_count": total_pairs,
        "unique_question_ids": int(
            len(
                set(pair_table["question_id_a"])
                | set(pair_table["question_id_b"])
            )
        ),
        "unique_image_ids": int(
            len(
                set(pair_table["image_id_a"])
                | set(pair_table["image_id_b"])
            )
        ),
        "same_question_count": same_question_count,
        "same_question_rate": same_question_count / total_pairs,
        "changed_answer_count": changed_answer_count,
        "changed_answer_rate": changed_answer_count / total_pairs,
        "question_type_counts": dict(question_types.most_common()),
        "answer_type_counts": dict(answer_types.most_common()),
    }

    return summary


def check_pair_summary(summary):
    problems = []

    if summary["pair_count"] != 95144:
        problems.append(
            f"Expected 95144 pairs, found {summary['pair_count']}"
        )

    if summary["same_question_count"] != summary["pair_count"]:
        difference = (
            summary["pair_count"] - summary["same_question_count"]
        )
        problems.append(
            f"{difference} pairs do not have matching question text"
        )

    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Pair validation failed:\n{problem_text}")


def save_summary(summary, report_path):
    from datetime import datetime, timezone

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VQA v2 validation complementary pairs",
        **summary,
    }

    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print()
    print("Pair summary")
    print(f"Pairs: {summary['pair_count']}")
    print(f"Unique questions: {summary['unique_question_ids']}")
    print(f"Unique images: {summary['unique_image_ids']}")
    print(
        f"Same questions: "
        f"{summary['same_question_count']} "
        f"({summary['same_question_rate']:.2%})"
    )
    print(
        f"Changed answers: "
        f"{summary['changed_answer_count']} "
        f"({summary['changed_answer_rate']:.2%})"
    )
    print(f"Report: {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/vqa_v2/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/vqa_v2"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/manifests/vqa_v2_pair_report.json"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    questions, annotations, pairs = load_vqa_data(arguments.input_dir)
    check_input_counts(questions, annotations, pairs)

    pair_rows = build_pair_rows(questions, annotations, pairs)
    pair_table = save_pair_table(pair_rows, arguments.output_dir)
    summary = summarize_pair_table(pair_table)
    check_pair_summary(summary)
    save_summary(summary, arguments.report)

    print()
    print("Complementary pair preparation passed.")


if __name__ == "__main__":
    main()