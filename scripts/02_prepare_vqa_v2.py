import argparse
from pathlib import Path


vqa_files = {
    "annotations": {
        "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "archive": "v2_Annotations_Val_mscoco.zip",
        "large": False,
    },
    "questions": {
        "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "archive": "v2_Questions_Val_mscoco.zip",
        "large": False,
    },
    "complementary_pairs": {
        "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
        "archive": "v2_Complementary_Pairs_Val_mscoco.zip",
        "large": False,
    },
    "images": {
        "url": "https://images.cocodataset.org/zips/val2014.zip",
        "archive": "val2014.zip",
        "large": True,
    },
}


def show_registered_files():
    print("Registered VQA v2 files")

    for file_name, details in vqa_files.items():
        file_type = "large" if details["large"] else "metadata"
        print(f"{file_name}: {details['archive']} ({file_type})")


def download_file(url, destination, chunk_size=1024 * 1024):
    import requests
    from tqdm import tqdm

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = destination.with_suffix(destination.suffix + ".part")
    downloaded_bytes = temporary_file.stat().st_size if temporary_file.exists() else 0

    headers = {}

    if downloaded_bytes:
        headers["Range"] = f"bytes={downloaded_bytes}-"

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    if downloaded_bytes and response.status_code != 206:
        downloaded_bytes = 0
        temporary_file.unlink(missing_ok=True)

    total_bytes = int(response.headers.get("content-length", 0)) + downloaded_bytes
    write_mode = "ab" if downloaded_bytes else "wb"

    with temporary_file.open(write_mode) as output_file:
        with tqdm(
            total=total_bytes,
            initial=downloaded_bytes,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for data_chunk in response.iter_content(chunk_size=chunk_size):
                if data_chunk:
                    output_file.write(data_chunk)
                    progress.update(len(data_chunk))

    temporary_file.replace(destination)
    return destination


def download_vqa_files(download_directory, include_images):
    selected_files = {
        name: details
        for name, details in vqa_files.items()
        if include_images or not details["large"]
    }

    downloaded_files = []

    for file_name, details in selected_files.items():
        destination = download_directory / details["archive"]

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already downloaded: {destination}")
        else:
            print(f"Downloading {file_name}")
            download_file(details["url"], destination)

        downloaded_files.append(destination)

    return downloaded_files


def safe_member_path(output_directory, member_name):
    output_root = output_directory.resolve()
    member_path = (output_directory / member_name).resolve()

    try:
        member_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"Unsafe archive member: {member_name}") from error

    return member_path


def extract_archive(archive_path, output_directory):
    import zipfile

    output_directory.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            safe_member_path(output_directory, member.filename)

        archive.extractall(output_directory)

    print(f"Extracted: {archive_path.name}")


def extract_downloaded_files(downloaded_files, output_directory):
    for archive_path in downloaded_files:
        extract_archive(archive_path, output_directory)


def read_json(json_path):
    import json

    with json_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def validate_vqa_metadata(output_directory):
    expected_files = {
        "questions": output_directory
        / "v2_OpenEnded_mscoco_val2014_questions.json",
        "annotations": output_directory
        / "v2_mscoco_val2014_annotations.json",
        "complementary_pairs": output_directory
        / "v2_mscoco_val2014_complementary_pairs.json",
    }

    missing_files = [
        str(file_path)
        for file_path in expected_files.values()
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing extracted files:\n{missing_text}")

    questions = read_json(expected_files["questions"]).get("questions", [])
    annotations = read_json(expected_files["annotations"]).get("annotations", [])
    complementary_pairs = read_json(expected_files["complementary_pairs"])

    results = {
        "question_count": len(questions),
        "annotation_count": len(annotations),
        "complementary_pair_count": len(complementary_pairs),
        "question_count_matches": len(questions) == 214354,
        "annotation_count_matches": len(annotations) == 214354,
        "pair_count_matches": len(complementary_pairs) == 95144,
    }

    print()
    print("VQA v2 metadata validation")
    print(f"Questions: {results['question_count']}")
    print(f"Annotations: {results['annotation_count']}")
    print(f"Complementary pairs: {results['complementary_pair_count']}")

    checks_passed = all(
        [
            results["question_count_matches"],
            results["annotation_count_matches"],
            results["pair_count_matches"],
        ]
    )

    if not checks_passed:
        raise ValueError("One or more VQA v2 record counts are unexpected.")

    print("Official record counts verified.")
    return results


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/vqa_v2/archives"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/vqa_v2/extracted"),
    )
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--show-files", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.show_files:
        show_registered_files()
        return

    downloaded_files = download_vqa_files(
        arguments.download_dir,
        arguments.include_images,
    )

    if not arguments.skip_extraction:
        extract_downloaded_files(downloaded_files, arguments.output_dir)

    validation_results = validate_vqa_metadata(arguments.output_dir)

    print()
    print("VQA v2 metadata preparation passed.")


if __name__ == "__main__":
    main()