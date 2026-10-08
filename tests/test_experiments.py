import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_run.analysis import analyze_images, analyze_languages, holm_adjust, statistics
from vlm_run.data import image_file, load_tables, make_jobs, select_rows
from vlm_run.engine import run_inference
from vlm_run.reporting import export_run, figures, review_export, verify_run
from vlm_run.scoring import normalize, score_runs, token_f1, vote_score
from vlm_run.storage import read_records


class FakeRunner:
    calls = 0

    def __init__(self, details, revision):
        pass

    def load(self):
        return self

    def prepare(self, image, question, instruction, options=None):
        answer = "yes" if image is not None and image.getpixel((0, 0))[0] > 150 else "no"
        return answer, question

    def generate(self, answer, maximum):
        type(self).calls += 1
        return {"prediction": answer, "generation_seconds": 0.01, "input_tokens": 10, "output_tokens": 2, "ended_with_eos": True, "generation_limit_reached": False}

    def memory(self):
        return []

    def close(self):
        pass


class ExperimentTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(normalize("The TWO cats!"), "2 cats")
        self.assertEqual(normalize("1,000.5"), "1000.5")
        self.assertEqual(vote_score("yes", ["yes"] * 3 + ["no"] * 7), 0.9)
        self.assertEqual(token_f1("attention dog owners", "attention dog owners pick up"), 0.75)
        self.assertEqual(list(holm_adjust([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])

    def test_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            for name in ["manifests", "raw/vqa_v2/images", "raw/banglaverse/images", "processed/benchmark", "processed/cultural"]:
                (data / name).mkdir(parents=True)
            (data / "manifests/final_benchmark_audit.json").write_text(json.dumps({"status": "passed"}))
            rows = []
            for stimulus, answer, color in [("a", "yes", "white"), ("b", "no", "black")]:
                Image.new("RGB", (8, 8), color).save(data / f"raw/vqa_v2/images/{stimulus}.png")
                for language in ["en", "bn"]:
                    rows.append({"variant_id": f"{stimulus}__{language}", "stimulus_id": stimulus, "group_id": "pair", "source": "vqa_v2", "language": language, "question": "Are these cattle?", "answer_en": answer, "image_path": f"data/raw/vqa_v2/images/{stimulus}.png", "accepted_answers_json": json.dumps([answer]), "answer_mode": "short_english"})
            pd.DataFrame(rows).to_parquet(data / "processed/benchmark/general_benchmark_multilingual.parquet", index=False)
            Image.new("RGB", (8, 8), "white").save(data / "raw/banglaverse/images/c.png")
            cultural = [{**rows[0], "variant_id": f"c__{language}", "stimulus_id": "c", "group_id": "culture", "source": "banglaverse", "language": language, "image_path": "data/raw/banglaverse/images/c.png"} for language in ["en", "bn"]]
            pd.DataFrame(cultural).to_parquet(data / "processed/cultural/cultural_benchmark_multilingual.parquet", index=False)
            args = SimpleNamespace(data_root=data, config=None, model="qwen3b", models=["qwen3b"], run="fixture", languages=["en", "bn"], sources=None, limit_groups=None, condition="baseline", max_new_tokens=None, vqa_annotations=None, seed=42, bootstrap=100, review_size=2)
            table = load_tables(data)
            self.assertEqual(len(table), 6)
            self.assertEqual(len(select_rows(table, ["en", "bn"], ["vqa_v2"], 1)), 4)
            swaps = make_jobs(table, "paired_swap")
            self.assertEqual(swaps[0]["answer_en"], "no")
            self.assertEqual(image_file("data\\raw\\vqa_v2\\images\\a.png", data).name, "a.png")
            def output_folder(a):
                return root / "results" / a.run
            with patch("vlm_run.engine.run_folder", output_folder), patch("vlm_run.scoring.run_folder", output_folder), patch("vlm_run.analysis.run_folder", output_folder), patch("vlm_run.reporting.run_folder", output_folder):
                FakeRunner.calls = 0
                first = run_inference(args, FakeRunner, lambda details: "fixed-revision")
                self.assertEqual(FakeRunner.calls, 6)
                run_inference(args, FakeRunner, lambda details: "fixed-revision")
                self.assertEqual(FakeRunner.calls, 6)
                for condition in ["paired_swap", "blank", "text_only"]:
                    args.condition = condition
                    run_inference(args, FakeRunner, lambda details: "fixed-revision")
                args.condition = "baseline"
                score_runs(args)
                analyze_languages(args)
                analyze_images(args)
                statistics(args)
                figures(args)
                review_export(args)
                verify_run(args)
                export_run(args)
                import zipfile

                archives = list((root / "results/downloads/fixture").glob("*.zip"))
                self.assertTrue(archives)
                with zipfile.ZipFile(archives[0]) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertTrue(all("\\" not in name for name in archive.namelist()))
                scores = pd.read_parquet(root / "results/fixture/analysis/scored_predictions.parquet")
                self.assertTrue(scores[scores.condition.eq("paired_swap")].normalized_exact_match.eq(1).all())
                self.assertTrue(scores[scores.condition.eq("baseline")].normalized_exact_match.eq(1).all())
                self.assertEqual(len(read_records(first / "predictions.jsonl")), 6)
                args.max_new_tokens = 65
                with self.assertRaisesRegex(ValueError, "Settings or data changed"):
                    run_inference(args, FakeRunner, lambda details: "fixed-revision")


if __name__ == "__main__":
    unittest.main()
