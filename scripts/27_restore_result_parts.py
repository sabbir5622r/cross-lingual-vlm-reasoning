import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "results/experiments" / args.run / "restore_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))["files"]

    def safe_path(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Restore manifest contains an unsafe path")
        return path

    for record in records:
        target = safe_path(record["file"])
        temporary = target.with_suffix(target.suffix + ".restoring")
        digest = hashlib.sha256()
        with temporary.open("wb") as destination:
            for name in record["parts"]:
                with safe_path(name).open("rb") as source:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                        digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            temporary.unlink()
            raise ValueError(f"Checksum failed: {target}")
        temporary.replace(target)
        print(f"Restored {target}")
    print("Result restoration completed")


if __name__ == "__main__":
    main()
