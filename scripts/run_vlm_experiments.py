import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", default="full_v1")
    parser.add_argument("--models", nargs="+", default=["qwen3b"])
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--controls", nargs="*", choices=["blank", "text_only", "paired_swap"], default=[])
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--vqa-annotations", type=Path)
    args = parser.parse_args()
    common = ["--run", args.run]
    for name in ["data_root", "config", "limit_groups"]:
        value = getattr(args, name)
        if value is not None:
            common.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ["languages", "sources"]:
        value = getattr(args, name)
        if value:
            common.extend(["--" + name, *value])

    def execute(script, extra=None):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *common, *(extra or [])], cwd=ROOT, check=True)

    for model in args.models:
        execute("16_run_baselines.py", ["--model", model])
        for condition in args.controls:
            execute("17_run_image_controls.py", ["--model", model, "--condition", condition])
        if args.efficiency:
            execute("21_measure_efficiency.py", ["--model", model])
    score_options = ["--vqa-annotations", str(args.vqa_annotations)] if args.vqa_annotations else []
    execute("18_score_predictions.py", score_options)
    execute("19_check_language_consistency.py")
    execute("20_check_image_interventions.py")
    execute("22_analyze_statistics.py", ["--bootstrap", str(args.bootstrap)])
    execute("23_make_figures.py")
    execute("24_verify_experiments.py", ["--models", *args.models])
    execute("25_prepare_human_review.py")
    execute("26_export_results.py")


if __name__ == "__main__":
    main()
