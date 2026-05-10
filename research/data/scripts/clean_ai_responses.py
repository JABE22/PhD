#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional

TEST_MAP = {
    "test1": "test1_ontological_innovation",
    "test2": "test2_epistemic_agency",
    "test3": "test3_theory_generation",
    "test4": "test4_category_recognition",
}

# Map test names to their folder structure: (parent_folder, responses_folder)
TEST_FOLDER_STRUCTURE = {
    "test1_ontological_innovation": ("test1_ontological-innovation", "test1_ontological_innovation_responses"),
    "test2_epistemic_agency": ("test2_epistemic-agency", "test2_epistemic_agency_responses"),
    "test3_theory_generation": ("test3_theory-generation", "test3_theory_generation_responses"),
    "test4_category_recognition": ("test4_category-recognition", "test4_category_recognition_responses"),
}

FENCE_START_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
FENCE_END_RE = re.compile(r"\s*```\s*$", re.IGNORECASE)


def normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def normalize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return normalize_string(obj)
    if isinstance(obj, list):
        return [normalize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}
    return obj


def strip_code_fences(text: str) -> str:
    text = FENCE_START_RE.sub("", text)
    text = FENCE_END_RE.sub("", text)
    return text.strip()


def extract_first_json_block(text: str) -> Optional[str]:
    start = None
    opening = None
    closing = None

    for i, ch in enumerate(text):
        if ch == "{":
            start, opening, closing = i, "{", "}"
            break
        if ch == "[":
            start, opening, closing = i, "[", "]"
            break

    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def clean_response_value(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw

    text = strip_code_fences(raw).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        candidate = extract_first_json_block(text)
        if candidate:
            try:
                parsed = json.loads(candidate)
            except Exception:
                parsed = None

    if parsed is None:
        return normalize_string(text)

    parsed = normalize_obj(parsed)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def resolve_test_folder(test_arg: str) -> str:
    return TEST_MAP.get(test_arg, test_arg)


def process_file(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "skipped_invalid_outer_json"

    if not isinstance(data, dict) or "response" not in data:
        return "skipped_no_response_field"

    old_response = data["response"]
    new_response = clean_response_value(old_response)

    if new_response == old_response:
        return "unchanged"

    data["response"] = new_response
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return "updated"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean AI response payloads in research/testN/<test_folder>.")
    parser.add_argument("test", help="test1/test2/test3/test4 or full folder name (e.g., test1_ontological_innovation)")
    parser.add_argument(
        "--only-not-updated",
        action="store_true",
        help="Print only files that were not updated (unchanged/skipped).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only summary counts (no per-file output).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    test_folder = resolve_test_folder(args.test)
    if test_folder not in TEST_FOLDER_STRUCTURE:
        raise SystemExit(f"Unknown test: {test_folder}. Available: {list(TEST_FOLDER_STRUCTURE.keys())}")
    parent_folder, responses_folder = TEST_FOLDER_STRUCTURE[test_folder]
    target_dir = repo_root / "research" / parent_folder / responses_folder

    if not target_dir.exists() or not target_dir.is_dir():
        raise SystemExit(f"Directory not found: {target_dir}")

    files = sorted(target_dir.rglob("*.json"))
    if not files:
        print(f"No JSON files found in: {target_dir}")
        return

    status_by_file: list[tuple[Path, str]] = []
    counts = Counter()

    for fp in files:
        status = process_file(fp)
        status_by_file.append((fp, status))
        counts[status] += 1

        if args.summary_only:
            continue

        if args.only_not_updated:
            if status != "updated":
                print(f"[{status}] {fp}")
        else:
            print(f"[{status}] {fp}")

    print(f"\nDirectory: {target_dir}")
    print(f"Total files: {len(files)}")
    print(f"Updated: {counts['updated']}")
    print(f"Unchanged: {counts['unchanged']}")
    print(f"Skipped (invalid outer JSON): {counts['skipped_invalid_outer_json']}")
    print(f"Skipped (no response field): {counts['skipped_no_response_field']}")


if __name__ == "__main__":
    main()