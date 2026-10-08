import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .scoring import scored
from .storage import run_folder, write_json


def analyze_languages(args):
    table = scored(args)
    table = table[table.condition.eq("baseline")]
    output = run_folder(args) / "analysis"
    agreement = []
    transfer = []
    for (model, source), group in table.groupby(["model", "source"]):
        predictions = group.pivot(index="stimulus_id", columns="language", values="normalized_prediction")
        correctness = group.pivot(index="stimulus_id", columns="language", values="normalized_exact_match")
        for a, b in itertools.combinations(sorted(predictions.columns), 2):
            usable = predictions[[a, b]].dropna()
            hits = correctness.loc[usable.index]
            agreement.append({"model": model, "source": source, "language_a": a, "language_b": b,
                "stimuli": len(usable), "prediction_agreement": float(usable[a].eq(usable[b]).mean()),
                "both_correct": float((hits[a].eq(1) & hits[b].eq(1)).mean())})
        if "en" in correctness:
            for language in correctness.columns:
                if language == "en":
                    continue
                valid = correctness[["en", language]].dropna()
                english_correct = valid[valid.en.eq(1)]
                transfer.append({"model": model, "source": source, "language": language, "stimuli": len(valid),
                    "english_correct_stimuli": len(english_correct),
                    "target_correct_given_english_correct": float(english_correct[language].mean()) if len(english_correct) else np.nan,
                    "english_minus_target_accuracy": float((valid.en - valid[language]).mean())})
    pd.DataFrame(agreement).to_csv(output / "cross_language_agreement.csv", index=False)
    pd.DataFrame(transfer).to_csv(output / "cross_language_transfer.csv", index=False)


def analyze_images(args):
    table = scored(args)
    output = run_folder(args) / "analysis"
    pairs = []
    vqa = table[table.source.eq("vqa_v2")]
    for (model, condition, language, group_id), group in vqa.groupby(["model", "condition", "language", "group_id"]):
        if len(group) != 2:
            raise ValueError(f"Incomplete VQA pair: {group_id}")
        group = group.sort_values("stimulus_id")
        pairs.append({"model": model, "condition": condition, "language": language, "group_id": group_id,
            "both_correct": float(group.normalized_exact_match.eq(1).all()),
            "mean_member_accuracy": float(group.normalized_exact_match.mean()),
            "prediction_changed": float(group.normalized_prediction.nunique() == 2),
            "reference_changed": float(group.normalized_answer.nunique() == 2),
            "original_answer_match": float(group.original_answer_match.mean())})
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(output / "complementary_pair_details.csv", index=False)
    if len(pair_table):
        summary = pair_table.groupby(["model", "condition", "language"]).agg(pairs=("group_id", "size"), both_correct=("both_correct", "mean"), member_accuracy=("mean_member_accuracy", "mean"), prediction_changed=("prediction_changed", "mean"), reference_changed=("reference_changed", "mean"), original_answer_match=("original_answer_match", "mean")).reset_index()
        summary.to_csv(output / "complementary_pair_summary.csv", index=False)
    effects = []
    for (model, source, language), group in table.groupby(["model", "source", "language"]):
        baseline = group[group.condition.eq("baseline")].set_index("variant_id")
        for condition in ["blank", "text_only", "paired_swap"]:
            control = group[group.condition.eq(condition)].set_index("variant_id")
            ids = baseline.index.intersection(control.index)
            if not len(ids):
                continue
            base = baseline.loc[ids]
            alternate = control.loc[ids]
            effects.append({"model": model, "source": source, "language": language, "condition": condition,
                "matched_variants": len(ids), "baseline_accuracy": float(base.normalized_exact_match.mean()),
                "control_accuracy": float(alternate.normalized_exact_match.mean()),
                "baseline_minus_control_accuracy": float((base.normalized_exact_match - alternate.normalized_exact_match).mean()),
                "prediction_changed": float(base.normalized_prediction.ne(alternate.normalized_prediction).mean()),
                "control_original_answer_match": float(alternate.original_answer_match.mean())})
    pd.DataFrame(effects).to_csv(output / "image_control_effects.csv", index=False)
    write_json(output / "intervention_notes.json", {
        "baseline": "Original question and image; score original reference.",
        "blank": "Same image dimensions, constant gray pixels; original reference is only a diagnostic comparison.",
        "text_only": "Image omitted; original reference is only a diagnostic comparison.",
        "paired_swap": "VQA complementary image replaces original image; question must match; score the observed image's reference.",
        "interpretation": "Prediction changes alone are not evidence of correct visual grounding. Report joint pair correctness alongside changes. Blank/text-only accuracy gaps are diagnostic, not a causal proof.",
    })


def bootstrap_interval(values, repeats, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty sample")
    estimates = []
    for _ in range(repeats):
        estimates.append(values[rng.integers(0, len(values), len(values))].mean())
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def statistics(args):
    table = scored(args)
    table = table[table.condition.eq("baseline")]
    output = run_folder(args) / "analysis"
    rng = np.random.default_rng(args.seed)
    intervals = []
    comparisons = []
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    for (model, source, language), group in table.groupby(["model", "source", "language"]):
        units = group.groupby("group_id").normalized_exact_match.mean()
        low, high = bootstrap_interval(units.values, args.bootstrap, rng)
        intervals.append({"model": model, "source": source, "language": language, "groups": len(units),
            "accuracy": float(units.mean()), "ci_low": low, "ci_high": high})
    for (model, source), group in table.groupby(["model", "source"]):
        units = group.groupby(["group_id", "language"]).normalized_exact_match.mean().unstack()
        if "en" not in units:
            continue
        for language in units.columns:
            if language == "en":
                continue
            valid = units[["en", language]].dropna()
            differences = (valid.en - valid[language]).values
            low, high = bootstrap_interval(differences, args.bootstrap, rng)
            baseline_joint = valid.en.eq(1)
            target_joint = valid[language].eq(1)
            wins = int((baseline_joint & ~target_joint).sum())
            losses = int((~baseline_joint & target_joint).sum())
            pvalue = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
            comparisons.append({"model": model, "source": source, "language": language, "groups": len(valid),
                "english_minus_target_accuracy": float(differences.mean()), "ci_low": low, "ci_high": high,
                "english_only_joint_correct_groups": wins, "target_only_joint_correct_groups": losses,
                "joint_correct_mcnemar_p": pvalue})
    pd.DataFrame(intervals).to_csv(output / "accuracy_cluster_intervals.csv", index=False)
    comparisons = pd.DataFrame(comparisons)
    if len(comparisons):
        comparisons["joint_correct_holm_p"] = holm_adjust(comparisons.joint_correct_mcnemar_p)
    comparisons.to_csv(output / "language_gap_statistics.csv", index=False)
    model_comparisons = []
    for (source, language), group in table.groupby(["source", "language"]):
        units = group.groupby(["group_id", "model"]).normalized_exact_match.mean().unstack()
        for a, b in itertools.combinations(sorted(units.columns), 2):
            valid = units[[a, b]].dropna()
            if len(valid) != len(units):
                raise ValueError("Model comparisons require identical group coverage")
            differences = (valid[a] - valid[b]).values
            low, high = bootstrap_interval(differences, args.bootstrap, rng)
            wins = int((valid[a].eq(1) & valid[b].ne(1)).sum())
            losses = int((valid[b].eq(1) & valid[a].ne(1)).sum())
            model_comparisons.append({"source": source, "language": language, "model_a": a, "model_b": b,
                "groups": len(valid), "model_a_minus_b_accuracy": float(differences.mean()), "ci_low": low, "ci_high": high,
                "joint_correct_mcnemar_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0})
    model_comparisons = pd.DataFrame(model_comparisons)
    if len(model_comparisons):
        model_comparisons["joint_correct_holm_p"] = holm_adjust(model_comparisons.joint_correct_mcnemar_p)
    model_comparisons.to_csv(output / "model_comparison_statistics.csv", index=False)
    write_json(output / "statistics_notes.json", {
        "bootstrap": args.bootstrap, "seed": args.seed,
        "unit": "Within each source, resample group_id with replacement. VQA complementary images stay together; other sources have singleton groups.",
        "intervals": "95% percentile intervals over group mean accuracy; language gaps use matched groups.",
        "test": "Exact McNemar on group-level all-members-correct status, not on dependent image rows. Holm correction across all model/source/non-English comparisons in this run.",
        "model_comparisons": "Matched groups for each source/language; pairwise model tests use a separate Holm family. Model selection, precision, and image processing differ by configuration and must be reported.",
        "limits": "These intervals describe this benchmark and do not correct translation errors, unreviewed synthetic text, or training-set contamination.",
    })
