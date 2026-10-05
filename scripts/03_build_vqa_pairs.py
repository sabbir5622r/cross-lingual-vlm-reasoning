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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/vqa_v2/extracted"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    questions, annotations, pairs = load_vqa_data(arguments.input_dir)
    check_input_counts(questions, annotations, pairs)


if __name__ == "__main__":
    main()