import json
import math
import time
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from .data import data_folder, image_file, load_tables, make_jobs, row_fingerprint, select_rows
from .models import Runner
from .storage import append_record, digest, environment, fingerprint, load_config, read_records, run_folder, write_json


def clean(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "ndim") and value.ndim > 0:
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def run_inference(args, runner_class=Runner, revision_resolver=None):
    config = load_config(args.config)
    details = config["models"][args.model]
    folder = data_folder(args.data_root)
    languages = args.languages or config["languages"]
    selected = select_rows(load_tables(folder), languages, args.sources, args.limit_groups, config["seed"])
    condition = args.condition
    jobs = make_jobs(selected, condition)
    if condition == "paired_swap":
        questions = selected.set_index("variant_id")["question"].to_dict()
        lookup = selected.set_index(["stimulus_id", "language"])["question"].to_dict()
        for job in jobs:
            if questions[job["variant_id"]].strip() != lookup[(job["observed_stimulus_id"], job["language"])].strip():
                raise ValueError("Paired swaps require identical question text for both images")
    paths = {}
    hashes = {}
    for value in tqdm(sorted({str(job["observed_image_path"]) for job in jobs}), desc="Checking images"):
        path = image_file(value, folder)
        with Image.open(path) as image:
            image.verify()
        paths[value] = path
        hashes[value] = digest(path)
    output = run_folder(args) / args.model / condition
    output.mkdir(parents=True, exist_ok=True)
    maximum = args.max_new_tokens or config["max_new_tokens"]
    if maximum < 1:
        raise ValueError("max_new_tokens must be positive")
    settings = {
        "model": args.model, "details": details, "condition": condition,
        "max_new_tokens": maximum, "instruction": config["instruction"],
        "languages": languages, "seed": config["seed"], "selection_hash": row_fingerprint(selected),
        "image_hash": fingerprint(hashes), "expected_predictions": len(jobs),
        "variant_ids": [job["variant_id"] for job in jobs],
        "limit_groups_per_source": args.limit_groups,
        "code_hash": fingerprint({path.name: digest(path) for path in Path(__file__).parent.glob("*.py")}),
    }
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    signature = fingerprint(settings)
    current_environment = environment()
    if old:
        if old["signature"] != signature:
            raise ValueError("Settings or data changed. Choose a new --run name rather than mixing checkpoints.")
        for name in ["torch", "transformers"]:
            if old["environment"]["packages"].get(name) != current_environment["packages"].get(name):
                raise ValueError(f"{name} changed since this run. Choose a new --run name.")
        revision = old["model_revision"]
    else:
        if revision_resolver is None:
            from huggingface_hub import HfApi

            revision = HfApi().model_info(details["model_id"], revision=details.get("revision", "main")).sha
        else:
            revision = revision_resolver(details)
        write_json(manifest_path, {**settings, "signature": signature, "model_revision": revision, "environment": current_environment})
    checkpoint = output / "predictions.jsonl"
    records = read_records(checkpoint)
    completed = [row["variant_id"] for row in records]
    expected = set(settings["variant_ids"])
    if len(completed) != len(set(completed)) or not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains duplicate or unexpected variant IDs")
    completed_ids = set(completed)
    remaining = [job for job in jobs if job["variant_id"] not in completed_ids]
    print(f"{args.model} / {condition}: {len(records)} completed, {len(remaining)} remaining")
    if not remaining:
        return output
    runner = runner_class(details, revision)
    session_started = time.perf_counter()
    try:
        runner.load()
        for job in tqdm(remaining, desc=f"{args.model} {condition}"):
            started = time.perf_counter()
            image = None
            if condition != "text_only":
                with Image.open(paths[str(job["observed_image_path"])]) as opened:
                    image = opened.convert("RGB")
                if condition == "blank":
                    image = Image.new("RGB", image.size, (127, 127, 127))
            mode = clean(job.get("answer_mode"))
            if mode and mode != "short_english":
                raise ValueError(f"Unsupported answer mode {mode!r}; this experiment uses English short-answer scoring")
            options = clean(job.get("options_json"))
            inputs, prompt = runner.prepare(image, job["question"], config["instruction"], options)
            result = runner.generate(inputs, maximum)
            record = {key: clean(value) for key, value in job.items()}
            record.update(result)
            record.update({
                "model": args.model, "model_id": details["model_id"], "model_revision": revision,
                "precision": details["precision"], "run": args.run, "prompt": prompt,
                "image_sha256": hashes[str(job["observed_image_path"])],
                "example_seconds": time.perf_counter() - started,
            })
            append_record(checkpoint, record)
        memory = runner.memory()
        total = len(read_records(checkpoint))
        append_record(output / "sessions.jsonl", {
            "environment": current_environment, "predictions_this_session": len(remaining),
            "session_seconds_including_load": time.perf_counter() - session_started, "memory": memory,
        })
        write_json(output / "completion.json", {"status": "complete", "predictions": total, "expected": len(jobs)})
    except Exception as error:
        append_record(output / "errors.jsonl", {"error": repr(error), "completed": len(read_records(checkpoint))})
        raise
    finally:
        runner.close()
    return output


def measure_efficiency(args):
    import pandas as pd

    config = load_config(args.config)
    folder = data_folder(args.data_root)
    table = select_rows(load_tables(folder), args.languages or config["languages"], args.sources or ["vqa_v2"], args.limit_groups or 3, config["seed"])
    baseline = run_folder(args) / args.model / "baseline/manifest.json"
    details = config["models"][args.model]
    if baseline.exists():
        previous = json.loads(baseline.read_text())
        if previous["details"] != details:
            raise ValueError("Model settings differ from baseline")
        revision = previous["model_revision"]
    else:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(details["model_id"], revision=details.get("revision", "main")).sha
    runner = Runner(details, revision)
    records = []
    try:
        runner.load()
        rows = table.to_dict("records")
        for index, row in enumerate(tqdm(rows[:3] + rows, desc="Efficiency warmup and measurement")):
            started = time.perf_counter()
            with Image.open(image_file(row["image_path"], folder)) as opened:
                image = opened.convert("RGB")
            inputs, _ = runner.prepare(image, row["question"], config["instruction"], clean(row.get("options_json")))
            result = runner.generate(inputs, args.max_new_tokens or config["max_new_tokens"])
            if index >= 3:
                records.append({"variant_id": row["variant_id"], "language": row["language"], **result, "example_seconds": time.perf_counter() - started})
        output = run_folder(args) / args.model / "efficiency"
        output.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(records)
        table.to_csv(output / "measurements.csv", index=False)
        summary = table.groupby("language").agg(examples=("variant_id", "size"), mean_generation_seconds=("generation_seconds", "mean"), mean_example_seconds=("example_seconds", "mean"), output_tokens=("output_tokens", "sum"), generation_seconds=("generation_seconds", "sum"))
        summary["generated_tokens_per_second"] = summary.output_tokens / summary.generation_seconds
        summary.to_csv(output / "summary.csv")
        write_json(output / "settings.json", {"details": details, "model_revision": revision, "environment": environment(), "memory": runner.memory(), "warmup_examples": 3, "selection_hash": row_fingerprint(pd.DataFrame(rows)), "max_new_tokens": args.max_new_tokens or config["max_new_tokens"], "note": "Batch size 1; generation timer uses CUDA synchronization; example timer includes preprocessing. Memory includes weights."})
    finally:
        runner.close()
