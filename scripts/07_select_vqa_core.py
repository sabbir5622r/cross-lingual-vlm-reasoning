import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


category_quotas = {
    "binary_perception": 700,
    "counting": 550,
    "attribute": 550,
    "object_recognition": 500,
    "activity": 450,
    "spatial": 450,
    "relation": 350,
    "other": 250,
    "visual_commonsense": 150,
    "ocr": 50,
}


def load_categorized_pairs(input_path):
    if not input_path.exists():
        raise FileNotFoundError(f"Categorized data not found: {input_path}")

    pair_table = pd.read_parquet(input_path)

    required_columns = [
        "pair_id",
        "image_id_a",
        "image_id_b",
        "question_id_a",
        "question_id_b",
        "reasoning_hint",
        "question",
        "bangla_question",
        "answer_a",
        "answer_b",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in pair_table.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing columns: {missing_text}")

    if pair_table["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs were found.")

    return pair_table


def check_category_supply(pair_table):
    available_counts = (
        pair_table["reasoning_hint"]
        .value_counts()
        .to_dict()
    )

    print("Category supply")

    for category_name, target_count in category_quotas.items():
        available_count = int(available_counts.get(category_name, 0))
        print(
            f"{category_name}: "
            f"{available_count} available, {target_count} needed"
        )

        if available_count < target_count:
            raise ValueError(
                f"Not enough {category_name} pairs: "
                f"{available_count} available, {target_count} needed"
            )


def category_selection_order(pair_table):
    available_counts = (
        pair_table["reasoning_hint"]
        .value_counts()
        .to_dict()
    )

    return sorted(
        category_quotas,
        key=lambda category_name: (
            available_counts.get(category_name, 0)
            / category_quotas[category_name]
        ),
    )


def select_core_pairs(pair_table, seed):
    used_images = set()
    selected_indices = []
    selection_order = category_selection_order(pair_table)

    print()
    print("Selection order")
    print(", ".join(selection_order))

    for category_number, category_name in enumerate(selection_order):
        target_count = category_quotas[category_name]

        category_rows = pair_table[
            pair_table["reasoning_hint"] == category_name
        ].sample(
            frac=1,
            random_state=seed + category_number,
        )

        selected_for_category = []

        for row_index, row in category_rows.iterrows():
            first_image = int(row["image_id_a"])
            second_image = int(row["image_id_b"])

            if first_image in used_images or second_image in used_images:
                continue

            selected_for_category.append(row_index)
            used_images.add(first_image)
            used_images.add(second_image)

            if len(selected_for_category) == target_count:
                break

        if len(selected_for_category) != target_count:
            raise ValueError(
                f"Could only select {len(selected_for_category)} "
                f"of {target_count} pairs for {category_name}"
            )

        selected_indices.extend(selected_for_category)

        print(
            f"{category_name}: "
            f"{len(selected_for_category)} selected"
        )

    selected = pair_table.loc[selected_indices].copy()

    category_rank = {
        category_name: position
        for position, category_name in enumerate(category_quotas)
    }

    selected["category_rank"] = (
        selected["reasoning_hint"].map(category_rank)
    )

    selected = selected.sort_values(
        by=["category_rank", "pair_id"],
        kind="stable",
    ).reset_index(drop=True)

    selected = selected.drop(columns=["category_rank"])
    selected["benchmark_group_id"] = [
        f"vqa_core_{row_number:05d}"
        for row_number in range(len(selected))
    ]
    selected["benchmark_source"] = "vqa_v2"
    selected["selection_seed"] = seed
    selected["selection_status"] = "selected_core"

    return selected, used_images, set(selected_indices)


def select_reserve_pairs(
    pair_table,
    selected_indices,
    core_images,
    seed,
):
    remaining = pair_table.drop(index=list(selected_indices)).copy()

    remaining = remaining[
        ~remaining["image_id_a"].isin(core_images)
        & ~remaining["image_id_b"].isin(core_images)
    ].copy()

    category_sizes = (
        remaining["reasoning_hint"]
        .value_counts()
        .to_dict()
    )

    remaining["category_size"] = (
        remaining["reasoning_hint"].map(category_sizes)
    )

    remaining = remaining.sample(
        frac=1,
        random_state=seed + 100,
    )

    remaining = remaining.sort_values(
        by=["category_size", "reasoning_hint"],
        ascending=[True, True],
        kind="stable",
    )

    reserve_indices = []
    reserve_images = set()

    for row_index, row in remaining.iterrows():
        first_image = int(row["image_id_a"])
        second_image = int(row["image_id_b"])

        if first_image in reserve_images or second_image in reserve_images:
            continue

        reserve_indices.append(row_index)
        reserve_images.add(first_image)
        reserve_images.add(second_image)

    reserve = pair_table.loc[reserve_indices].copy()
    reserve = reserve.sort_values(
        by=["reasoning_hint", "pair_id"],
        kind="stable",
    ).reset_index(drop=True)

    reserve["reserve_group_id"] = [
        f"vqa_reserve_{row_number:05d}"
        for row_number in range(len(reserve))
    ]
    reserve["benchmark_source"] = "vqa_v2"
    reserve["selection_seed"] = seed
    reserve["selection_status"] = "reserve"

    return reserve


def validate_selection(core_pairs, reserve_pairs):
    if len(core_pairs) != 4000:
        raise ValueError(
            f"Expected 4000 core pairs, found {len(core_pairs)}"
        )

    actual_counts = (
        core_pairs["reasoning_hint"]
        .value_counts()
        .to_dict()
    )

    for category_name, expected_count in category_quotas.items():
        actual_count = int(actual_counts.get(category_name, 0))

        if actual_count != expected_count:
            raise ValueError(
                f"{category_name}: expected {expected_count}, "
                f"found {actual_count}"
            )

    core_images = pd.concat(
        [
            core_pairs["image_id_a"],
            core_pairs["image_id_b"],
        ],
        ignore_index=True,
    )

    reserve_images = pd.concat(
        [
            reserve_pairs["image_id_a"],
            reserve_pairs["image_id_b"],
        ],
        ignore_index=True,
    )

    if core_images.duplicated().any():
        raise ValueError("The core selection contains repeated images.")

    if reserve_images.duplicated().any():
        raise ValueError("The reserve selection contains repeated images.")

    if set(core_images) & set(reserve_images):
        raise ValueError("Core and reserve selections share images.")

    if set(core_pairs["pair_id"]) & set(reserve_pairs["pair_id"]):
        raise ValueError("Core and reserve selections share pairs.")

    if core_pairs["benchmark_group_id"].duplicated().any():
        raise ValueError("Duplicate benchmark group IDs were found.")


def pair_id_checksum(pair_table):
    pair_ids = sorted(pair_table["pair_id"].astype(str))
    joined_ids = "\n".join(pair_ids)
    return hashlib.sha256(joined_ids.encode("utf-8")).hexdigest()


def save_table(pair_table, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_table.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    pair_table.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Parquet: {output_path}")
    print(f"CSV: {csv_path}")


def make_report(source_pairs, core_pairs, reserve_pairs, seed):
    from datetime import datetime, timezone

    core_images = set(core_pairs["image_id_a"]) | set(
        core_pairs["image_id_b"]
    )
    reserve_images = set(reserve_pairs["image_id_a"]) | set(
        reserve_pairs["image_id_b"]
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "source_pairs": len(source_pairs),
        "core_pairs": len(core_pairs),
        "core_unique_images": len(core_images),
        "reserve_pairs": len(reserve_pairs),
        "reserve_unique_images": len(reserve_images),
        "category_quotas": category_quotas,
        "selected_category_counts": {
            str(name): int(count)
            for name, count in core_pairs["reasoning_hint"]
            .value_counts()
            .items()
        },
        "reserve_category_counts": {
            str(name): int(count)
            for name, count in reserve_pairs["reasoning_hint"]
            .value_counts()
            .items()
        },
        "core_pair_id_sha256": pair_id_checksum(core_pairs),
        "reserve_pair_id_sha256": pair_id_checksum(reserve_pairs),
        "image_overlap_between_core_and_reserve": 0,
        "category_labels": "provisional_heuristic_labels",
    }


def save_report(report, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(f"Report: {report_path}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/interim/multilingual/"
            "vqa_en_bn_categorized.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/benchmark"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/manifests/vqa_core_selection_report.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    pair_table = load_categorized_pairs(arguments.input)
    check_category_supply(pair_table)

    core_pairs, core_images, selected_indices = select_core_pairs(
        pair_table,
        arguments.seed,
    )

    reserve_pairs = select_reserve_pairs(
        pair_table,
        selected_indices,
        core_images,
        arguments.seed,
    )

    validate_selection(core_pairs, reserve_pairs)

    print()
    print("Core selection")
    print(f"Pairs: {len(core_pairs)}")
    print(
        f"Unique images: "
        f"{len(set(core_pairs['image_id_a']) | set(core_pairs['image_id_b']))}"
    )

    for category_name in category_quotas:
        category_count = int(
            (core_pairs["reasoning_hint"] == category_name).sum()
        )
        print(f"{category_name}: {category_count}")

    print()
    print("Reserve selection")
    print(f"Pairs: {len(reserve_pairs)}")
    print(
        f"Unique images: "
        f"{len(set(reserve_pairs['image_id_a']) | set(reserve_pairs['image_id_b']))}"
    )

    save_table(
        core_pairs,
        arguments.output_dir / "vqa_core_4000.parquet",
    )
    save_table(
        reserve_pairs,
        arguments.output_dir / "vqa_reserve.parquet",
    )

    report = make_report(
        pair_table,
        core_pairs,
        reserve_pairs,
        arguments.seed,
    )
    save_report(report, arguments.report)

    print()
    print("VQA core selection completed.")


if __name__ == "__main__":
    main()