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


def read_registry(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Registry not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        registry = yaml.safe_load(config_file)

    if not isinstance(registry, dict):
        raise ValueError("The registry must contain a YAML dictionary.")

    return registry


def check_required_sections(registry):
    missing_sections = [
        section for section in required_sections if section not in registry
    ]

    if missing_sections:
        missing_text = ", ".join(missing_sections)
        raise ValueError(f"Missing registry sections: {missing_text}")


def check_source_entries(registry):
    source_sections = [
        "primary_sources",
        "user_prepared_sources",
        "external_evaluation_sources",
    ]

    problems = []

    for section_name in source_sections:
        sources = registry.get(section_name, {})

        if not isinstance(sources, dict):
            problems.append(f"{section_name} must be a dictionary")
            continue

        for source_name, source_details in sources.items():
            if not isinstance(source_details, dict):
                problems.append(f"{section_name}.{source_name} must be a dictionary")
                continue

            for field_name in ["enabled", "role", "download_mode", "local_directory"]:
                if field_name not in source_details:
                    problems.append(
                        f"{section_name}.{source_name} is missing {field_name}"
                    )

    if problems:
        problem_text = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"Registry validation failed:\n{problem_text}")


def count_sources(registry):
    counts = {}

    for section_name in [
        "primary_sources",
        "user_prepared_sources",
        "external_evaluation_sources",
    ]:
        sources = registry.get(section_name, {})
        enabled_count = sum(
            1 for source in sources.values() if source.get("enabled", False)
        )

        counts[section_name] = {
            "total": len(sources),
            "enabled": enabled_count,
        }

    return counts


def print_summary(registry, source_counts, config_path):
    print(f"Registry: {config_path}")
    print(f"Schema version: {registry.get('schema_version')}")
    print(f"Project: {registry.get('project_name')}")
    print()

    for section_name, counts in source_counts.items():
        readable_name = section_name.replace("_", " ").title()
        print(
            f"{readable_name}: "
            f"{counts['enabled']} enabled out of {counts['total']}"
        )

    language_names = [
        details.get("name", language_code)
        for language_code, details in registry["canonical_languages"].items()
    ]

    print()
    print(f"Canonical languages: {', '.join(language_names)}")
    print("Registry validation passed.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_sources.yaml"),
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    registry = read_registry(arguments.config)
    check_required_sections(registry)
    check_source_entries(registry)
    source_counts = count_sources(registry)
    print_summary(registry, source_counts, arguments.config)


if __name__ == "__main__":
    main()