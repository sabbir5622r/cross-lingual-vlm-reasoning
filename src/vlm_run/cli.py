import argparse
from pathlib import Path

from .analysis import analyze_images, analyze_languages, statistics
from .engine import measure_efficiency, run_inference
from .reporting import export_run, figures, review_export, verify_run
from .scoring import score_runs


def main(stage):
    parser = argparse.ArgumentParser(description=f"Cross-lingual VLM experiments: {stage}")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", default="full_v1")
    parser.add_argument("--model", default="qwen3b")
    parser.add_argument("--models", nargs="+", default=["qwen3b"])
    parser.add_argument("--languages", nargs="+", choices=["en", "bn", "hi", "ur", "en_bn_cs"])
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--condition", choices=["baseline", "blank", "text_only", "paired_swap"], default="baseline")
    parser.add_argument("--vqa-annotations", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-size", type=int, default=30)
    args = parser.parse_args()
    actions = {"infer": run_inference, "controls": run_inference, "efficiency": measure_efficiency,
        "score": score_runs, "languages": analyze_languages, "images": analyze_images,
        "statistics": statistics, "figures": figures, "export": export_run, "verify": verify_run,
        "review": review_export}
    if stage == "smoke":
        if args.run == "full_v1":
            args.run = "smoke_v2"
        args.sources = args.sources or ["vqa_v2"]
        args.limit_groups = args.limit_groups or 1
        args.condition = "baseline"
        run_inference(args)
    else:
        if stage == "controls" and args.condition == "baseline":
            parser.error("Choose --condition blank, text_only, or paired_swap")
        actions[stage](args)
