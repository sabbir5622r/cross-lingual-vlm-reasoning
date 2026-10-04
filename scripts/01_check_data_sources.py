import argparse
from pathlib import Path

import yaml


required_sections = [
    "primary_sources",
    "user_prepared_sources",
    "external_evaluation_sources",
    "canonical_languages",
    "storage",
    "split_policy",
]

source_sections = [
    "primary_sources",
    "user_prepared_sources",
    "external_evaluation_sources",
]


def read_registry(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Registry not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        registry = yaml.safe_load(config_file)

    if not isinstance(registry, dict):
        raise ValueError("The registry must contain a YAML dictionary.")

    return registry


def validate_registry(registry):
    missing_sections = [
        section for section in required_sections if section not in registry
    ]

    if missing_sections:
        missing_text = ", ".join(missing_sections)
        raise ValueError(f"Missing registry sections: {missing_text}")

    problems = []

    for section_name in source_sections:
        sources = registry.get(section_name, {})

        if not isinstance(sources, dict):
            problems.append(f"{section_name} must be a dictionary")
            continue

        for source_name, details in sources.items():
            if not isinstance(details, dict):
                problems.append(f"{section_name}.{source_name} must be a dictionary")
                continue

            needed_fields = ["enabled", "role", "download_mode", "local_directory"]

            for field_name in needed_fields:
                if field_name not in details:
                    problems.append(
                        f"{section_name}.{source_name} is missing {field_name}"
                    )

    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Registry validation failed:\n{problem_text}")


def summarize_registry(registry, config_path):
    print(f"Registry: {config_path}")
    print(f"Schema version: {registry.get('schema_version')}")
    print(f"Project: {registry.get('project_name')}")
    print()

    for section_name in source_sections:
        sources = registry.get(section_name, {})
        enabled = sum(
            1 for details in sources.values() if details.get("enabled", False)
        )
        readable_name = section_name.replace("_", " ").title()
        print(f"{readable_name}: {enabled} enabled out of {len(sources)}")

    language_names = [
        details.get("name", language_code)
        for language_code, details in registry["canonical_languages"].items()
    ]

    print()
    print(f"Canonical languages: {', '.join(language_names)}")
    print("Registry validation passed.")


def gather_sources(registry):
    gathered_sources = []

    for section_name in source_sections:
        for source_name, details in registry.get(section_name, {}).items():
            gathered_sources.append(
                {
                    "section": section_name,
                    "name": source_name,
                    "details": details,
                }
            )

    return gathered_sources


def inspect_local_source(source):
    details = source["details"]
    local_path = Path(details["local_directory"])

    files = []

    if local_path.exists():
        files = [
            file_path
            for file_path in local_path.rglob("*")
            if file_path.is_file() and file_path.name != ".gitkeep"
        ]

    total_bytes = sum(file_path.stat().st_size for file_path in files)

    return {
        "section": source["section"],
        "name": source["name"],
        "enabled": details.get("enabled", False),
        "local_directory": str(local_path),
        "directory_exists": local_path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": [str(file_path) for file_path in files],
    }


def inspect_local_sources(registry):
    return [
        inspect_local_source(source)
        for source in gather_sources(registry)
        if source["details"].get("enabled", False)
    ]


def readable_size(byte_count):
    size = float(byte_count)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


def print_local_summary(local_results):
    print()
    print("Local dataset inspection")

    for result in local_results:
        state = "found" if result["directory_exists"] else "missing"
        print(
            f"{result['name']}: {state}, "
            f"{result['file_count']} files, "
            f"{readable_size(result['total_bytes'])}"
        )

def check_source_page(source, timeout_seconds):
    import requests

    details = source["details"]
    source_page = details.get("source_page")

    if not source_page:
        return {
            "section": source["section"],
            "name": source["name"],
            "source_page": None,
            "status": "not_listed",
            "status_code": None,
            "error": None,
        }

    try:
        response = requests.get(
            source_page,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout_seconds,
            stream=True,
            allow_redirects=True,
        )
        status_code = response.status_code
        response.close()

        status = "available" if 200 <= status_code < 400 else "unavailable"

        return {
            "section": source["section"],
            "name": source["name"],
            "source_page": source_page,
            "status": status,
            "status_code": status_code,
            "error": None,
        }
    except requests.RequestException as error:
        return {
            "section": source["section"],
            "name": source["name"],
            "source_page": source_page,
            "status": "error",
            "status_code": None,
            "error": str(error),
        }


def check_source_pages(registry, timeout_seconds=20):
    return [
        check_source_page(source, timeout_seconds)
        for source in gather_sources(registry)
        if source["details"].get("enabled", False)
    ]


def print_page_summary(page_results):
    print()
    print("Dataset source pages")

    for result in page_results:
        status_code = result["status_code"]

        if status_code is None:
            print(f"{result['name']}: {result['status']}")
        else:
            print(
                f"{result['name']}: {result['status']} "
                f"(HTTP {status_code})"
            )

def check_huggingface_source(source):
    from huggingface_hub import HfApi

    details = source["details"]
    dataset_id = details.get("dataset_id")

    if not dataset_id:
        return None

    try:
        dataset_info = HfApi().dataset_info(dataset_id)

        return {
            "section": source["section"],
            "name": source["name"],
            "dataset_id": dataset_id,
            "status": "available",
            "private": bool(dataset_info.private),
            "gated": dataset_info.gated,
            "file_count": len(dataset_info.siblings or []),
            "last_modified": (
                str(dataset_info.last_modified)
                if dataset_info.last_modified
                else None
            ),
            "error": None,
        }
    except Exception as error:
        return {
            "section": source["section"],
            "name": source["name"],
            "dataset_id": dataset_id,
            "status": "error",
            "private": None,
            "gated": None,
            "file_count": None,
            "last_modified": None,
            "error": str(error),
        }


def check_huggingface_sources(registry):
    results = []

    for source in gather_sources(registry):
        details = source["details"]

        if not details.get("enabled", False):
            continue

        if details.get("download_mode") != "huggingface":
            continue

        result = check_huggingface_source(source)

        if result:
            results.append(result)

    return results


def print_huggingface_summary(huggingface_results):
    print()
    print("Hugging Face datasets")

    if not huggingface_results:
        print("No enabled Hugging Face datasets found.")
        return

    for result in huggingface_results:
        if result["status"] == "available":
            print(
                f"{result['dataset_id']}: available, "
                f"{result['file_count']} repository files"
            )
        else:
            print(f"{result['dataset_id']}: {result['status']}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_sources.yaml"),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
    )
    return parser.parse_args()

def main():
    arguments = parse_arguments()
    registry = read_registry(arguments.config)
    validate_registry(registry)
    summarize_registry(registry, arguments.config)

    local_results = inspect_local_sources(registry)
    print_local_summary(local_results)

    page_results = check_source_pages(registry, arguments.timeout)
    print_page_summary(page_results)

    huggingface_results = check_huggingface_sources(registry)
    print_huggingface_summary(huggingface_results)
    
if __name__ == "__main__":
    main()