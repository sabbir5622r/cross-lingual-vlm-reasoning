import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "processed" / "benchmark"
CULTURAL_DIR = ROOT / "data" / "processed" / "cultural"
MANIFEST_DIR = ROOT / "data" / "manifests"

VQA_PATH = BENCHMARK_DIR / "vqa_core_4000.parquet"
XGQA_PATH = BENCHMARK_DIR / "xgqa_300.parquet"
TEXTVQA_PATH = BENCHMARK_DIR / "textvqa_300.parquet"
AOKVQA_PATH = BENCHMARK_DIR / "aokvqa_400.parquet"
CULTURAL_PATH = CULTURAL_DIR / "banglaverse_500.parquet"

OUTPUT_PATH = BENCHMARK_DIR / "general_benchmark_canonical.parquet"
OUTPUT_CSV = BENCHMARK_DIR / "general_benchmark_canonical.csv"
REPORT_PATH = MANIFEST_DIR / "benchmark_assembly_report.json"


def first_value(row, names, default=""):
    lookup = {column.lower(): column for column in row.index}

    for name in names:
        column = lookup.get(name.lower())
        if column is not None:
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return value

    return default


def member_names(base, member):
    letter = "a" if member == 1 else "b"

    return [
        f"{base}_{member}",
        f"{base}{member}",
        f"{base}_{letter}",
        f"{letter}_{base}",
        f"{base}_image_{member}",
        f"image_{member}_{base}",
    ]


def vqa_value(row, bases, member, default=""):
    names = []

    for base in bases:
        names.extend(member_names(base, member))

    return first_value(row, names, default)

def prepare_vqa():
    table = pd.read_parquet(VQA_PATH)
    rows = []

    for row_number, row in table.iterrows():
        group_id = str(
            first_value(
                row,
                ["benchmark_id", "pair_id"],
                f"vqa_{row_number + 1:04d}",
            )
        )

        category = str(
            first_value(
                row,
                ["reasoning_category", "reasoning_hint", "category"],
                "other",
            )
        )

        shared_question_en = first_value(
            row,
            ["question_en", "english_question", "question"],
            "",
        )

        shared_question_bn = first_value(
            row,
            ["question_bn", "bangla_question", "bn_question"],
            "",
        )

        for member in [1, 2]:
            question_en = vqa_value(
                row,
                ["question_en", "english_question"],
                member,
            )

            if not str(question_en).strip():
                question_en = shared_question_en

            question_bn = vqa_value(
                row,
                ["question_bn", "bangla_question", "bn_question"],
                member,
            )

            if not str(question_bn).strip():
                question_bn = shared_question_bn

            answer = vqa_value(
                row,
                ["answer_en", "majority_answer", "answer"],
                member,
            )

            image_id = vqa_value(
                row,
                ["image_id", "coco_image_id"],
                member,
            )

            image_path = vqa_value(
                row,
                ["image_path", "path"],
                member,
            )

            question_id = vqa_value(
                row,
                ["question_id"],
                member,
            )

            if not str(question_en).strip():
                raise ValueError(
                    f"English VQA question missing for row {row_number}, "
                    f"member {member}. Columns: {list(table.columns)}"
                )

            if not str(question_bn).strip():
                raise ValueError(
                    f"Bangla VQA question missing for row {row_number}, "
                    f"member {member}"
                )

            if not str(answer).strip():
                raise ValueError(
                    f"VQA answer missing for row {row_number}, "
                    f"member {member}"
                )

            rows.append(
                {
                    "group_id": group_id,
                    "stimulus_id": f"{group_id}_{member}",
                    "source": "vqa_v2",
                    "pair_member": member,
                    "source_question_id": str(question_id),
                    "source_image_id": str(image_id),
                    "image_path": str(image_path),
                    "question_en": str(question_en).strip(),
                    "question_bn": str(question_bn).strip(),
                    "answer_en": str(answer).strip(),
                    "accepted_answers_json": json.dumps(
                        [str(answer).strip()],
                        ensure_ascii=False,
                    ),
                    "reasoning_category": category,
                    "answer_mode": "short_english",
                }
            )

    return pd.DataFrame(rows)

def prepare_single_source(path, source, expected_size):
    table = pd.read_parquet(path)
    rows = []

    for row_number, row in table.iterrows():
        group_id = str(
            first_value(
                row,
                ["benchmark_id"],
                f"{source}_{row_number + 1:04d}",
            )
        )

        answer = str(
            first_value(
                row,
                ["answer_en", "answer"],
            )
        ).strip()

        accepted = first_value(
            row,
            [
                "accepted_answers_json",
                "answers_json",
                "direct_answers_json",
            ],
            json.dumps([answer], ensure_ascii=False),
        )

        rows.append(
            {
                "group_id": group_id,
                "stimulus_id": group_id,
                "source": source,
                "pair_member": 1,
                "source_question_id": str(
                    first_value(row, ["question_id"])
                ),
                "source_image_id": str(
                    first_value(row, ["image_id"])
                ),
                "image_path": str(
                    first_value(row, ["image_path"])
                ),
                "question_en": str(
                    first_value(row, ["question_en", "question"])
                ).strip(),
                "question_bn": str(
                    first_value(
                        row,
                        ["question_bn", "bangla_question"],
                    )
                ).strip(),
                "answer_en": answer,
                "accepted_answers_json": str(accepted),
                "reasoning_category": str(
                    first_value(
                        row,
                        ["reasoning_category", "reasoning_hint"],
                        "other",
                    )
                ),
                "answer_mode": "short_english",
            }
        )

    result = pd.DataFrame(rows)

    if len(result) != expected_size:
        raise ValueError(
            f"{source} expected {expected_size} rows, found {len(result)}"
        )

    return result


def verify_table(table):
    expected_stimuli = {
        "vqa_v2": 8000,
        "xgqa": 300,
        "textvqa": 300,
        "aokvqa": 400,
    }

    expected_groups = {
        "vqa_v2": 4000,
        "xgqa": 300,
        "textvqa": 300,
        "aokvqa": 400,
    }

    stimulus_counts = table.groupby("source").size().to_dict()
    group_counts = (
        table.groupby("source")["group_id"].nunique().to_dict()
    )

    if stimulus_counts != expected_stimuli:
        raise ValueError(
            f"Unexpected stimulus counts: {stimulus_counts}"
        )

    if group_counts != expected_groups:
        raise ValueError(f"Unexpected group counts: {group_counts}")

    if table["stimulus_id"].duplicated().any():
        raise ValueError("Duplicate stimulus IDs were found")

    for column in ["question_en", "answer_en"]:
        if table[column].fillna("").str.strip().eq("").any():
            raise ValueError(f"Empty values found in {column}")


def main():
    required_paths = [
        VQA_PATH,
        XGQA_PATH,
        TEXTVQA_PATH,
        AOKVQA_PATH,
        CULTURAL_PATH,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Run the earlier preparation scripts first:\n"
            + "\n".join(missing)
        )

    tables = [
        prepare_vqa(),
        prepare_single_source(XGQA_PATH, "xgqa", 300),
        prepare_single_source(TEXTVQA_PATH, "textvqa", 300),
        prepare_single_source(AOKVQA_PATH, "aokvqa", 400),
    ]

    combined = pd.concat(tables, ignore_index=True)
    verify_table(combined)

    combined.to_parquet(OUTPUT_PATH, index=False)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    report = {
        "general_groups": int(combined["group_id"].nunique()),
        "general_stimuli": int(len(combined)),
        "source_stimulus_counts": (
            combined.groupby("source").size().to_dict()
        ),
        "source_group_counts": (
            combined.groupby("source")["group_id"].nunique().to_dict()
        ),
        "cultural_groups": int(len(pd.read_parquet(CULTURAL_PATH))),
        "answer_language": "English",
        "parquet_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Benchmark assembly")
    print(f"General groups: {report['general_groups']}")
    print(f"General stimuli: {report['general_stimuli']}")
    print(f"Cultural groups: {report['cultural_groups']}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()