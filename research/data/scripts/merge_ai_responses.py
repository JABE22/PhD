#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def merge_dir(base_dir: Path):
    if not base_dir.exists() or not base_dir.is_dir():
        raise RuntimeError(f"Directory does not exist or is not a directory: {base_dir}")

    all_path = base_dir / "all_responses.json"

    # Load existing responses (if any)
    if all_path.exists():
        with all_path.open("r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                raise RuntimeError(f"{all_path} exists but is not valid JSON")
        if not isinstance(existing, list):
            raise RuntimeError(f"{all_path} must contain a JSON list")
    else:
        existing = []

    existing_count = len(existing)

    # Load all *_sample_*.json files
    new_samples = []
    for path in base_dir.glob("*_sample_*.json"):
        if path.name == "all_responses.json":
            continue
        with path.open("r") as f:
            obj = json.load(f)
        new_samples.append(obj)

    merged = existing + new_samples

    with all_path.open("w") as f:
        json.dump(merged, f, indent=2)

    print(f"[{base_dir}] Existing responses: {existing_count}")
    print(f"[{base_dir}] New sample files read: {len(new_samples)}")
    print(f"[{base_dir}] Total responses now in all_responses.json: {len(merged)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python research/data/scripts/merge_ai_responses.py <responses_dir> [<responses_dir> ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        base_dir = Path(arg)
        merge_dir(base_dir)

if __name__ == "__main__":
    main()
