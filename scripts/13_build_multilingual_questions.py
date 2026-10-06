import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"
BATCH_SIZE_GPU = 24
BATCH_SIZE_CPU = 4

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "processed" / "benchmark"
CULTURAL_DIR = ROOT / "data" / "processed" / "cultural"
INTERIM_DIR = ROOT / "data" / "interim" / "multilingual"
MANIFEST_DIR = ROOT / "data" / "manifests"

GENERAL_PATH = BENCHMARK_DIR / "general_benchmark_canonical.parquet"
CULTURAL_PATH = CULTURAL_DIR / "banglaverse_500.parquet"
CHECKPOINT_PATH = INTERIM_DIR / "general_translation_checkpoint.parquet"

GENERAL_OUTPUT = BENCHMARK_DIR / "general_benchmark_multilingual.parquet"
GENERAL_CSV = BENCHMARK_DIR / "general_benchmark_multilingual.csv"
CULTURAL_OUTPUT = CULTURAL_DIR / "cultural_benchmark_multilingual.parquet"
CULTURAL_CSV = CULTURAL_DIR / "cultural_benchmark_multilingual.csv"
REPORT_PATH = MANIFEST_DIR / "multilingual_construction_report.json"

TARGETS = {
    "question_bn": "ben_Beng",
    "question_hi": "hin_Deva",
    "question_ur": "urd_Arab",
}

CODE_SWITCH_TERMS = [
    ("ছবিতে", "image-এ"),
    ("ছবির", "image-এর"),
    ("কোন রঙ", "কোন color"),
    ("রঙ", "color"),
    ("মানুষ", "person"),
    ("ব্যক্তি", "person"),
    ("গাড়ি", "car"),
    ("বস্তু", "object"),
    ("প্রাণী", "animal"),
    ("খেলা", "game"),
    ("খাবার", "food"),
    ("সংখ্যা", "number"),
    ("কতটি", "how many"),
    ("কত", "how many"),
    ("কোথায়", "where"),
    ("কে", "who"),
    ("কোনটি", "which one"),
]


def make_directories():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def present(value):
    return pd.notna(value) and bool(str(value).strip())


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {MODEL_NAME} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def translate_batch(texts, target_code, tokenizer, model, device):
    tokenizer.src_lang = "eng_Latn"

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=192,
    ).to(device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                target_code
            ),
            max_new_tokens=128,
            num_beams=4,
        )

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )


def fill_translation_column(
    table,
    column,
    target_code,
    tokenizer,
    model,
    device,
):
    if column not in table.columns:
        table[column] = ""

    missing_indices = [
        index
        for index, value in table[column].items()
        if not present(value)
    ]

    cache = {}
    batch_size = (
        BATCH_SIZE_GPU if device == "cuda" else BATCH_SIZE_CPU
    )

    print(f"{column}: {len(missing_indices)} translations needed")

    for start in tqdm(
        range(0, len(missing_indices), batch_size),
        desc=column,
    ):
        batch_indices = missing_indices[start:start + batch_size]
        source_texts = [
            str(table.at[index, "question_en"]).strip()
            for index in batch_indices
        ]

        new_texts = []
        new_positions = []

        for position, text in enumerate(source_texts):
            if text not in cache:
                new_texts.append(text)
                new_positions.append(position)

        if new_texts:
            translated = translate_batch(
                new_texts,
                target_code,
                tokenizer,
                model,
                device,
            )

            for source, result in zip(new_texts, translated):
                cache[source] = result.strip()

        for index, source in zip(batch_indices, source_texts):
            table.at[index, column] = cache[source]

        if start % (batch_size * 20) == 0:
            table.to_parquet(CHECKPOINT_PATH, index=False)

    table.to_parquet(CHECKPOINT_PATH, index=False)
    return table


def make_code_switched(question, stimulus_id):
    text = str(question).strip()
    digest = hashlib.sha256(str(stimulus_id).encode()).hexdigest()
    start = int(digest[:8], 16) % len(CODE_SWITCH_TERMS)

    replacements = 0

    for offset in range(len(CODE_SWITCH_TERMS)):
        source, replacement = CODE_SWITCH_TERMS[
            (start + offset) % len(CODE_SWITCH_TERMS)
        ]

        if source in text:
            text = text.replace(source, replacement, 1)
            replacements += 1

        if replacements == 2:
            break

    if replacements == 0:
        text = f"Image দেখে, {text}"

    return text


def expand_general(table):
    rows = []

    for _, row in table.iterrows():
        questions = {
            "en": row["question_en"],
            "bn": row["question_bn"],
            "hi": row["question_hi"],
            "ur": row["question_ur"],
            "en_bn_cs": make_code_switched(
                row["question_bn"],
                row["stimulus_id"],
            ),
        }

        for language, question in questions.items():
            item = row.to_dict()
            item["language"] = language
            item["question"] = str(question).strip()
            item["variant_id"] = (
                f"{row['stimulus_id']}__{language}"
            )
            rows.append(item)

    result = pd.DataFrame(rows)
    result = result.drop(
        columns=[
            "question_en",
            "question_bn",
            "question_hi",
            "question_ur",
        ],
        errors="ignore",
    )

    return result


def expand_cultural(table):
    rows = []

    for _, row in table.iterrows():
        questions = {
            "en": row["question_en"],
            "bn": row["question_bn"],
            "hi": row["question_hi"],
            "ur": row["question_ur"],
            "en_bn_cs": make_code_switched(
                row["question_bn"],
                row["benchmark_id"],
            ),
        }

        for language, question in questions.items():
            item = row.to_dict()
            item["group_id"] = row["benchmark_id"]
            item["stimulus_id"] = row["benchmark_id"]
            item["language"] = language
            item["question"] = str(question).strip()
            item["variant_id"] = (
                f"{row['benchmark_id']}__{language}"
            )
            item["answer_en"] = row["answer_en"]
            item["answer_mode"] = "multiple_choice"
            rows.append(item)

    result = pd.DataFrame(rows)
    result = result.drop(
        columns=[
            "question_en",
            "question_bn",
            "question_hi",
            "question_ur",
        ],
        errors="ignore",
    )

    return result


def verify_output(general, cultural):
    if len(general) != 45000:
        raise ValueError(
            f"Expected 45,000 general variants, found {len(general)}"
        )

    if len(cultural) != 2500:
        raise ValueError(
            f"Expected 2,500 cultural variants, found {len(cultural)}"
        )

    if general["variant_id"].duplicated().any():
        raise ValueError("Duplicate general variant IDs found")

    if cultural["variant_id"].duplicated().any():
        raise ValueError("Duplicate cultural variant IDs found")

    if general["question"].fillna("").str.strip().eq("").any():
        raise ValueError("Empty general multilingual questions found")

    if cultural["question"].fillna("").str.strip().eq("").any():
        raise ValueError("Empty cultural multilingual questions found")


def main():
    make_directories()

    if CHECKPOINT_PATH.exists():
        general = pd.read_parquet(CHECKPOINT_PATH)
        print("Resuming from the translation checkpoint")
    else:
        general = pd.read_parquet(GENERAL_PATH)

    cultural = pd.read_parquet(CULTURAL_PATH)

    needed = False

    for column in TARGETS:
        if column not in general.columns:
            needed = True
            break
        if general[column].fillna("").str.strip().eq("").any():
            needed = True
            break

    if needed:
        tokenizer, model, device = load_model()

        for column, target_code in TARGETS.items():
            general = fill_translation_column(
                general,
                column,
                target_code,
                tokenizer,
                model,
                device,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    general_long = expand_general(general)
    cultural_long = expand_cultural(cultural)
    verify_output(general_long, cultural_long)

    general_long.to_parquet(GENERAL_OUTPUT, index=False)
    general_long.to_csv(GENERAL_CSV, index=False, encoding="utf-8")

    cultural_long.to_parquet(CULTURAL_OUTPUT, index=False)
    cultural_long.to_csv(CULTURAL_CSV, index=False, encoding="utf-8")

    report = {
        "translation_model": MODEL_NAME,
        "general_groups": int(general["group_id"].nunique()),
        "general_stimuli": int(general["stimulus_id"].nunique()),
        "general_language_variants": len(general_long),
        "cultural_groups": int(
            cultural_long["group_id"].nunique()
        ),
        "cultural_language_variants": len(cultural_long),
        "languages": ["en", "bn", "hi", "ur", "en_bn_cs"],
        "answer_language": "English",
        "code_switch_method": (
            "Deterministic English lexical substitutions in Bangla"
        ),
        "human_review_required": True,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Multilingual construction")
    print(f"General variants: {len(general_long)}")
    print(f"Cultural variants: {len(cultural_long)}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()