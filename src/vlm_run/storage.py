import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid checkpoint line {number} in {path}. "
                    "Preserve the file and repair the incomplete final line before resuming."
                ) from error
    return rows


def environment():
    packages = {}
    for name in ["torch", "transformers", "accelerate", "pandas", "numpy", "pillow", "pyarrow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    try:
        import torch

        result["cuda_runtime"] = torch.version.cuda
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        result["gpus"] = []
    return result


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs/vlm_experiments.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_folder(args):
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        raise ValueError("Use letters, numbers, underscores, or hyphens for --run")
    return ROOT / "results/experiments" / args.run
