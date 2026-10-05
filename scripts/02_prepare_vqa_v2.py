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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-files", action="store_true")
    return parser.parse_args()


def main():
    parse_arguments()
    show_registered_files()


if __name__ == "__main__":
    main()