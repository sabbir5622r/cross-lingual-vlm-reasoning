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
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    annotations = load_annotations(arguments.annotations)
    profiles = make_answer_profiles(annotations)
    show_profile_summary(profiles)


if __name__ == "__main__":
    main()