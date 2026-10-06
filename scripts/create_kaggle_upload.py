from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

project_folder = Path(__file__).resolve().parents[1]

output_file = project_folder / "cross_lingual_vlm_local_data_v2.zip"

items_to_include = [
    project_folder
    / "data"
    / "processed"
    / "benchmark"
    / "general_benchmark_canonical.parquet",
    project_folder
    / "data"
    / "processed"
    / "cultural"
    / "banglaverse_500.parquet",
    project_folder / "data" / "raw" / "aokvqa",
    project_folder / "data" / "raw" / "banglaverse",
    project_folder / "data" / "raw" / "textvqa",
    project_folder / "data" / "raw" / "xgqa",
]


def collect_files():
    selected_files = []

    for item in items_to_include:
        if not item.exists():
            print(f"Missing: {item.relative_to(project_folder)}")
            continue

        if item.is_file():
            selected_files.append(item)
        else:
            selected_files.extend(
                file_path
                for file_path in item.rglob("*")
                if file_path.is_file()
            )

    return selected_files


def create_archive(files):
    if output_file.exists():
        output_file.unlink()

    with ZipFile(
        output_file,
        mode="w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for number, file_path in enumerate(files, start=1):
            archive_name = file_path.relative_to(project_folder).as_posix()
            archive.write(file_path, arcname=archive_name)

            if number % 100 == 0 or number == len(files):
                print(f"Added {number}/{len(files)} files")


def verify_archive():
    with ZipFile(output_file, mode="r") as archive:
        names = archive.namelist()
        bad_names = [name for name in names if "\\" in name]

    print(f"Archive: {output_file}")
    print(f"Files: {len(names)}")
    print(f"Invalid paths: {len(bad_names)}")

    if bad_names:
        raise ValueError(
            f"The archive contains invalid paths: {bad_names[:5]}"
        )


def main():
    files = collect_files()

    if not files:
        raise RuntimeError("No files were found for the Kaggle archive")

    create_archive(files)
    verify_archive()


if __name__ == "__main__":
    main()