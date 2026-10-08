import json
from pathlib import Path

import pandas as pd

from .storage import ROOT, digest, fingerprint


TABLES = {
    "general": "processed/benchmark/general_benchmark_multilingual.parquet",
    "cultural": "processed/cultural/cultural_benchmark_multilingual.parquet",
}


def data_folder(value=None):
    folder = Path(value) if value else ROOT / "data"
    if (folder / "data/processed").exists():
        folder = folder / "data"
    if not (folder / "processed").exists():
        raise FileNotFoundError(f"No processed benchmark folder under {folder}")
    return folder.resolve()


def image_file(value, folder):
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() in {"nan", "none"}:
        raise ValueError("Empty image path")
    supplied = Path(text)
    if supplied.is_absolute() and supplied.is_file():
        return supplied
    relative = text
    if "/data/" in relative:
        relative = relative.split("/data/", 1)[1]
    elif relative.startswith("data/"):
        relative = relative[5:]
    for candidate in [folder / relative, ROOT / text]:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image not found: {text}, data root={folder}")


def load_tables(folder):
    report_path = folder / "manifests/final_benchmark_audit.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Final audit report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("status", "")).lower() != "passed":
        raise ValueError("The final dataset audit has not passed")
    frames = []
    for name, relative in TABLES.items():
        path = folder / relative
        expected_hash = report.get(f"{name}_sha256")
        if expected_hash and digest(path) != expected_hash:
            raise ValueError(f"{path.name} changed after the final audit; rerun script 14")
        table = pd.read_parquet(path)
        needed = {"variant_id", "stimulus_id", "group_id", "language", "question", "answer_en", "image_path"}
        missing = needed.difference(table.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        table["benchmark"] = name
        if "source" not in table:
            table["source"] = "banglaverse" if name == "cultural" else "unknown"
        if name == "cultural":
            table["source"] = "banglaverse"
        for column in needed:
            if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty {column} in {path.name}")
        if table["variant_id"].duplicated().any():
            raise ValueError(f"Duplicate variant IDs in {path.name}")
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    if combined["variant_id"].duplicated().any():
        raise ValueError("Variant IDs overlap between the general and cultural benchmarks")
    return combined


def select_rows(table, languages, sources=None, limit_groups=None, seed=42):
    if sources:
        unknown = set(sources).difference(table["source"].unique())
        if unknown:
            raise ValueError(f"Unknown sources: {sorted(unknown)}")
        table = table[table["source"].isin(sources)]
    if not set(languages).issubset(set(table["language"])):
        raise ValueError("One or more requested languages are missing")
    table = table[table["language"].isin(languages)].copy()
    if limit_groups is not None:
        if limit_groups < 1:
            raise ValueError("--limit-groups must be positive")
        chosen = []
        for _, source in table.groupby("source", sort=True):
            ids = source["group_id"].astype(str).drop_duplicates().sort_values()
            chosen.extend(ids.sample(min(limit_groups, len(ids)), random_state=seed).tolist())
        table = table[table["group_id"].astype(str).isin(chosen)]
    for stimulus, group in table.groupby("stimulus_id"):
        if sorted(group["language"].tolist()) != sorted(languages):
            raise ValueError(f"Incomplete or duplicate languages for {stimulus}")
    return table.sort_values(["source", "group_id", "stimulus_id", "language"]).reset_index(drop=True)


def row_fingerprint(table):
    columns = [c for c in ["variant_id", "group_id", "source", "question", "answer_en", "image_path", "accepted_answers_json", "options_json", "answer_mode"] if c in table]
    return fingerprint(json.loads(table[columns].to_json(orient="records", force_ascii=False)))


def pair_mates(table):
    rows = table[table["source"].eq("vqa_v2")].drop_duplicates("stimulus_id")
    mates = {}
    for group_id, group in rows.groupby("group_id"):
        if len(group) != 2:
            raise ValueError(f"VQA group {group_id} does not have two images")
        a, b = group.iloc[0].to_dict(), group.iloc[1].to_dict()
        mates[a["stimulus_id"]] = b
        mates[b["stimulus_id"]] = a
    return mates


def make_jobs(table, condition):
    if condition == "paired_swap":
        table = table[table["source"].eq("vqa_v2")].copy()
    mates = pair_mates(table) if condition == "paired_swap" else {}
    jobs = []
    for _, row in table.iterrows():
        job = row.to_dict()
        observed = mates[row["stimulus_id"]] if condition == "paired_swap" else job
        job["observed_stimulus_id"] = observed["stimulus_id"]
        job["observed_image_path"] = observed["image_path"]
        job["original_answer_en"] = job["answer_en"]
        job["answer_en"] = observed["answer_en"]
        for column in ["source_question_id", "accepted_answers_json"]:
            job[column] = observed.get(column)
        job["condition"] = condition
        jobs.append(job)
    if not jobs:
        raise ValueError(f"No examples available for {condition}")
    return jobs
