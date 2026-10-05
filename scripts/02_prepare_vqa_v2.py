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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/vqa_v2/archives"),
    )
    parser.add_argument("--include-images", action="store_true")
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

    print()
    print(f"Downloaded files: {len(downloaded_files)}")


if __name__ == "__main__":
    main()