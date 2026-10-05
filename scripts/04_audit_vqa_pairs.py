import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


number_words = {
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

articles = {"a", "an", "the"}


def read_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def normalize_answer(answer):
    text = str(answer).lower().strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[,!?;:\"()\[\]{}]", " ", text)

    words = []

    for word in text.split():
        word = number_words.get(word, word)

        if word not in articles:
            words.append(word)

    return " ".join(words)


def load_annotations(annotation_path):
    annotation_data = read_json(annotation_path)
    annotations = annotation_data.get("annotations", [])

    if not annotations:
        raise ValueError("No VQA annotations were found.")

    return annotations


def make_answer_profiles(annotations):
    profiles = {}

    for annotation in annotations:
        question_id = int(annotation["question_id"])
        label = normalize_answer(annotation["multiple_choice_answer"])
        answers = [
            normalize_answer(answer["answer"])
            for answer in annotation.get("answers", [])
        ]

        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)

        if most_common:
            top_answer, top_votes = most_common[0]
        else:
            top_answer, top_votes = "", 0

        profiles[question_id] = {
            "label": label,
            "label_votes": int(answer_counts.get(label, 0)),
            "top_answer": top_answer,
            "top_votes": int(top_votes),
            "distinct_answers": len(answer_counts),
            "answer_count": len(answers),
        }

    return profiles


def show_profile_summary(profiles):
    vote_counts = Counter(
        profile["label_votes"] for profile in profiles.values()
    )

    print("Human-answer profiles")
    print(f"Questions: {len(profiles)}")
    print(f"Questions with 10 answers: {sum(p['answer_count'] == 10 for p in profiles.values())}")
    print(f"Questions with at least 6 label votes: {sum(p['label_votes'] >= 6 for p in profiles.values())}")
    print(f"Questions with at least 8 label votes: {sum(p['label_votes'] >= 8 for p in profiles.values())}")
    print()
    print("Label vote distribution")

    for vote_count in sorted(vote_counts):
        print(f"{vote_count}: {vote_counts[vote_count]}")

def load_pair_table(pair_path):
    if not pair_path.exists():
        raise FileNotFoundError(f"Pair table not found: {pair_path}")

    pair_table = pd.read_parquet(pair_path)

    if len(pair_table) != 95144:
        raise ValueError(
            f"Expected 95144 pairs, found {len(pair_table)}"
        )

    return pair_table


def build_pair_audit(pair_table, profiles):
    audit_rows = []

    for row in pair_table.itertuples(index=False):
        first_id = int(row.question_id_a)
        second_id = int(row.question_id_b)

        if first_id not in profiles or second_id not in profiles:
            raise KeyError(f"Answer profile missing for {row.pair_id}")

        first_profile = profiles[first_id]
        second_profile = profiles[second_id]
        minimum_votes = min(
            first_profile["label_votes"],
            second_profile["label_votes"],
        )

        same_answer_type = (
            str(row.answer_type_a) == str(row.answer_type_b)
        )

        audit_rows.append(
            {
                "pair_id": row.pair_id,
                "question_id_a": first_id,
                "question_id_b": second_id,
                "image_id_a": int(row.image_id_a),
                "image_id_b": int(row.image_id_b),
                "question": row.question_a,
                "answer_a": row.answer_a,
                "answer_b": row.answer_b,
                "answer_type_a": row.answer_type_a,
                "answer_type_b": row.answer_type_b,
                "same_question": bool(row.same_question),
                "answer_changed": bool(row.answer_changed),
                "same_answer_type": same_answer_type,
                "label_votes_a": first_profile["label_votes"],
                "label_votes_b": second_profile["label_votes"],
                "minimum_label_votes": minimum_votes,
                "distinct_answers_a": first_profile["distinct_answers"],
                "distinct_answers_b": second_profile["distinct_answers"],
                "top_answer_a": first_profile["top_answer"],
                "top_answer_b": second_profile["top_answer"],
            }
        )

    return pd.DataFrame(audit_rows)


def show_threshold_results(audit_table):
    changed_pairs = audit_table["answer_changed"]
    changed_count = int(changed_pairs.sum())

    print()
    print("Pair agreement thresholds")
    print(f"All pairs: {len(audit_table)}")
    print(f"Different-answer pairs: {changed_count}")
    print(
        f"Answer-type mismatches: "
        f"{int((~audit_table['same_answer_type']).sum())}"
    )

    for minimum_votes in [3, 5, 6, 7, 8, 9]:
        selected = (
            audit_table["same_question"]
            & audit_table["answer_changed"]
            & audit_table["same_answer_type"]
            & (audit_table["minimum_label_votes"] >= minimum_votes)
        )

        selected_count = int(selected.sum())
        share = selected_count / changed_count if changed_count else 0

        print(
            f"At least {minimum_votes} votes: "
            f"{selected_count} ({share:.2%} of different-answer pairs)"
        )


def save_audit_table(audit_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_table.to_parquet(output_path, index=False)
    print(f"Audit table: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/raw/vqa_v2/extracted/"
            "v2_mscoco_val2014_annotations.json"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path(
            "data/processed/vqa_v2/"
            "vqa_v2_complementary_pairs.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/interim/vqa_v2/"
            "vqa_v2_pair_audit.parquet"
        ),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    annotations = load_annotations(arguments.annotations)
    profiles = make_answer_profiles(annotations)
    pair_table = load_pair_table(arguments.pairs)
    audit_table = build_pair_audit(pair_table, profiles)

    show_threshold_results(audit_table)
    save_audit_table(audit_table, arguments.output)


if __name__ == "__main__":
    main()