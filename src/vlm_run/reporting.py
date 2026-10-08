import json
import zipfile
from pathlib import Path

import pandas as pd

from .data import data_folder, load_tables, select_rows
from .scoring import scored
from .storage import digest, load_config, read_records, run_folder, write_json


def figures(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = run_folder(args) / "analysis"
    table = pd.read_csv(root / "accuracy_cluster_intervals.csv")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    order = ["en", "bn", "hi", "ur", "en_bn_cs"]
    for model, group in table.groupby("model"):
        sources = sorted(group.source.unique())
        fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 4), squeeze=False, sharey=True)
        for axis, source in zip(axes[0], sources):
            part = group[group.source.eq(source)].set_index("language").reindex(order).dropna(subset=["accuracy"])
            errors = [part.accuracy - part.ci_low, part.ci_high - part.accuracy]
            axis.bar(part.index, part.accuracy, yerr=errors, capsize=3, color="#3879a8")
            axis.set_title(source)
            axis.set_ylim(0, 1)
            axis.tick_params(axis="x", rotation=35)
            axis.set_ylabel("Normalized exact match")
        fig.suptitle(f"{model}: baseline, 95% group bootstrap intervals")
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(output / f"{model}_accuracy.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    pair_path = root / "complementary_pair_summary.csv"
    if pair_path.exists():
        pairs = pd.read_csv(pair_path)
        for model, group in pairs.groupby("model"):
            part = group.pivot(index="language", columns="condition", values="both_correct").reindex(order).dropna(how="all")
            axis = part.plot.bar(figsize=(7, 4), ylim=(0, 1))
            axis.set_ylabel("Both complementary images correct")
            axis.set_title(model)
            axis.figure.tight_layout()
            for suffix in ["png", "pdf"]:
                axis.figure.savefig(output / f"{model}_pair_accuracy.{suffix}", dpi=200, bbox_inches="tight")
            plt.close(axis.figure)


def verify_run(args):
    folder = run_folder(args)
    config = load_config(args.config)
    selected = select_rows(load_tables(data_folder(args.data_root)), args.languages or config["languages"], args.sources, args.limit_groups, config["seed"])
    expected = set(selected.variant_id)
    problems = []
    checked = []
    for model in args.models:
        manifest_path = folder / model / "baseline/manifest.json"
        if not manifest_path.exists():
            problems.append(f"Baseline missing for {model}")
            continue
        manifest = json.loads(manifest_path.read_text())
        rows = read_records(manifest_path.parent / "predictions.jsonl")
        ids = [row["variant_id"] for row in rows]
        if set(ids) != expected or len(ids) != len(expected):
            problems.append(f"Baseline coverage does not match requested selection for {model}")
        if set(manifest["variant_ids"]) != expected:
            problems.append(f"Manifest selection mismatch for {model}")
        if any(row["model_revision"] != manifest["model_revision"] for row in rows):
            problems.append(f"Mixed model revisions for {model}")
        checked.append({"model": model, "predictions": len(rows), "expected": len(expected)})
    report = {"status": "failed" if problems else "passed", "checked": checked, "problems": problems,
        "scope": "Computational coverage check only; does not certify translation quality or research validity."}
    write_json(folder / "run_verification.json", report)
    print(json.dumps(report, indent=2))
    if problems:
        raise ValueError("Experiment verification failed")


def review_export(args):
    table = scored(args)
    baseline = table[table.condition.eq("baseline")]
    samples = []
    for _, group in baseline.groupby(["model", "source", "language"]):
        samples.append(group.sample(min(args.review_size, len(group)), random_state=args.seed))
    flagged = table[table.generation_limit_reached | table.empty_prediction | table.non_latin_output]
    result = pd.concat(samples + [flagged], ignore_index=True).drop_duplicates(["model", "condition", "variant_id"])
    columns = ["model", "condition", "source", "language", "variant_id", "stimulus_id", "group_id", "question", "image_path", "observed_image_path", "answer_en", "prediction", "normalized_exact_match", "generation_limit_reached", "non_latin_output"]
    result = result[columns].copy()
    for name in ["translation_faithful", "reference_adequate", "prediction_semantically_correct", "reviewer", "review_notes"]:
        result[name] = ""
    result.to_csv(run_folder(args) / "analysis/human_review.csv", index=False)


def export_run(args):
    folder = run_folder(args)
    if not folder.exists():
        raise FileNotFoundError(folder)
    analysis = folder / "analysis"
    if analysis.exists():
        for path in sorted(analysis.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if len(frame) <= 100:
                path.with_suffix(".tex").write_text(frame.to_latex(index=False, float_format="%.4f", escape=True), encoding="utf-8")
    output = folder.parent / "downloads" / args.run
    output.mkdir(parents=True, exist_ok=True)
    limit = 95 * 1024 * 1024
    staging = output / "staging"
    staging.mkdir(exist_ok=True)
    entries = []
    restoration = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = (Path("results/experiments") / args.run / path.relative_to(folder)).as_posix()
        if path.stat().st_size <= limit:
            entries.append((path, relative))
            continue
        parts = []
        with path.open("rb") as handle:
            number = 0
            while block := handle.read(limit):
                number += 1
                name = relative + f".part{number:03d}"
                piece = staging / name
                piece.parent.mkdir(parents=True, exist_ok=True)
                piece.write_bytes(block)
                entries.append((piece, name))
                parts.append(name)
        restoration.append({"file": relative, "parts": parts, "sha256": digest(path)})
    restore_path = staging / "restore_manifest.json"
    write_json(restore_path, {"files": restoration})
    entries.append((restore_path, f"results/experiments/{args.run}/restore_manifest.json"))
    chunks, current, size = [], [], 0
    for path, name in entries:
        if current and size + path.stat().st_size > limit:
            chunks.append(current)
            current, size = [], 0
        current.append((path, name))
        size += path.stat().st_size
    if current:
        chunks.append(current)
    archive_paths = []
    for number, paths in enumerate(chunks, 1):
        archive = output / f"{args.run}_part_{number:03d}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path, name in paths:
                handle.write(path, name)
        if archive.stat().st_size >= 100_000_000:
            raise ValueError(f"Archive exceeds 100 MB: {archive}")
        archive_paths.append(str(archive))
    import shutil

    shutil.rmtree(staging)
    write_json(output / "download_manifest.json", {"archives": archive_paths, "extract": "Extract every part into the same repository root. Parts are independent ZIP archives. Then run scripts/27_restore_result_parts.py --run RUN."})
    print("\n".join(archive_paths))
