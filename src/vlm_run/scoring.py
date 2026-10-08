import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from .storage import read_records, run_folder, write_json


NUMBERS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), map(str, range(21))))
CONTRACTIONS = {"cant": "can't", "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't", "doesnt": "doesn't", "didnt": "didn't", "couldnt": "couldn't", "wouldnt": "wouldn't", "shouldnt": "shouldn't"}


def normalize(value):
    text = str(value or "").lower().strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s'.]", " ", text, flags=re.UNICODE)
    return " ".join(CONTRACTIONS.get(word, NUMBERS.get(word, word)) for word in text.split() if word not in {"a", "an", "the"})


def references(row):
    value = row.get("accepted_answers_json")
    accepted = []
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
            raise ValueError("accepted_answers_json must contain a list of answer strings")
        accepted = parsed
    accepted.append(str(row["answer_en"]))
    return sorted(set(normalize(item) for item in accepted if str(item).strip()))


def token_f1(prediction, answer):
    predicted = Counter(prediction.split())
    expected = Counter(answer.split())
    common = sum((predicted & expected).values())
    denominator = sum(predicted.values()) + sum(expected.values())
    return 2 * common / denominator if denominator else 0.0


def vote_score(prediction, answers):
    labels = [normalize(answer) for answer in answers]
    if len(labels) != 10:
        raise ValueError("VQA consensus scoring requires the original ten annotations")
    return sum(min(sum(label == prediction for j, label in enumerate(labels) if j != i) / 3, 1) for i in range(10)) / 10


def annotation_lookup(path):
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["question_id"]): [answer["answer"] for answer in item["answers"]] for item in data["annotations"]}


def score_runs(args):
    folder = run_folder(args)
    annotations = annotation_lookup(args.vqa_annotations)
    records = []
    manifests = sorted(folder.glob("*/*/manifest.json"))
    if not manifests:
        raise FileNotFoundError("No experiment prediction manifests found")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        rows = read_records(manifest_path.parent / "predictions.jsonl")
        ids = [row["variant_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(manifest["variant_ids"]):
            raise ValueError(f"Incomplete or duplicate predictions: {manifest_path.parent}. Resume inference first.")
        for row in rows:
            prediction = normalize(row["prediction"])
            gold = references(row)
            question_id = str(row.get("source_question_id", ""))
            if question_id.endswith(".0"):
                question_id = question_id[:-2]
            votes = annotations.get(question_id) if row["source"] == "vqa_v2" else None
            row.update({
                "normalized_prediction": prediction, "normalized_answer": normalize(row["answer_en"]),
                "normalized_exact_match": float(prediction in gold),
                "token_f1_diagnostic": max((token_f1(prediction, answer) for answer in gold), default=0.0),
                "empty_prediction": not prediction,
                "non_latin_output": bool(re.search(r"[\u0980-\u09ff\u0900-\u097f\u0600-\u06ff]", row["prediction"])),
                "consensus_vote_score": vote_score(prediction, votes) if votes else np.nan,
                "original_answer_match": float(prediction == normalize(row.get("original_answer_en", row["answer_en"]))),
            })
            records.append(row)
    table = pd.DataFrame(records)
    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    table.to_parquet(output / "scored_predictions.parquet", index=False)
    keys = ["model", "condition", "benchmark", "source", "language"]
    summary = table.groupby(keys).agg(
        predictions=("variant_id", "size"), normalized_exact_match=("normalized_exact_match", "mean"),
        token_f1_diagnostic=("token_f1_diagnostic", "mean"), consensus_vote_score=("consensus_vote_score", "mean"),
        consensus_scored_predictions=("consensus_vote_score", "count"),
        empty_predictions=("empty_prediction", "sum"), non_latin_outputs=("non_latin_output", "sum"),
        generation_limit_hits=("generation_limit_reached", "sum"),
        mean_generation_seconds=("generation_seconds", "mean"), mean_example_seconds=("example_seconds", "mean"),
    ).reset_index()
    summary.to_csv(output / "accuracy_by_source_language.csv", index=False)
    write_json(output / "metric_notes.json", {
        "primary": "Normalized exact match to stored references, not the official VQA metric.",
        "consensus_vote_score": "Optional original ten-vote leave-one-out consensus using this project's normalizer; missing without --vqa-annotations. Not a claim of official evaluator equivalence.",
        "token_f1_diagnostic": "Token overlap diagnostic only; does not replace exact match or validate OCR completeness.",
        "language": "Questions vary by language; all runs request English answers.",
        "aggregation": "Source-specific metrics are primary. Source sizes differ; no pooled result implies equal source weighting.",
        "limits": "Truncated, empty, and non-English answers remain in denominators; no substring rescue or automatic translation.",
    })
    print(summary.to_string(index=False))


def scored(args):
    path = run_folder(args) / "analysis/scored_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError("Run script 18 to score the predictions first")
    return pd.read_parquet(path)
