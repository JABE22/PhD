#!/usr/bin/env python3
"""Build a compact AI-theory catalog CSV from test3 all_responses.json via OpenRouter.

The script:
1) Parses AI theory records from research/test3_theory-generation/ai_responses/all_responses.json.
    Note: each record has a `response` field that is itself a JSON-formatted string.
2) Uses OpenRouter (anthropic/claude-3.7-sonnet) with strict JSON schema response format
    to normalize:
    - theory name formatting
    - abbreviation
    - ~100-word scientific characterization
3) Writes research/data/consciousness_theories_ai.csv.

Usage:
  python research/data/scripts/build_consciousness_theories_ai_csv.py
  python research/data/scripts/build_consciousness_theories_ai_csv.py --max-items 20 --delay 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore


DEFAULT_INPUT = Path("research/test3_theory-generation/ai_responses/all_responses.json")
DEFAULT_OUTPUT = Path("research/data/consciousness_theories_ai.csv")
DEFAULT_MODEL_NAME = "claude-3.7-sonnet"
DEFAULT_MODEL_ID = "anthropic/claude-3.7-sonnet"


def build_openrouter_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER2_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER2_API_KEY/OPENROUTER_API_KEY in environment or .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def clean_markup(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text, flags=re.S)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    return text.strip()


def try_parse_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = [text]
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidates.append(text[i : j + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def extract_json_string_field(text: str, key: str) -> str:
    pattern = rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"')
    except Exception:
        return m.group(1)


def extract_predictions(text: str) -> List[str]:
    m = re.search(r'"predictions"\s*:\s*\[(.*?)\]', text, flags=re.S)
    if not m:
        return []
    body = m.group(1)
    items = re.findall(r'"((?:\\.|[^"\\])*)"', body, flags=re.S)
    out: List[str] = []
    for it in items:
        try:
            out.append(json.loads(f'"{it}"'))
        except Exception:
            out.append(it)
    return [x.strip() for x in out if str(x).strip()]


def parse_response_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    raw = str((rec or {}).get("response", ""))
    text = clean_markup(raw)
    obj = try_parse_json_block(text)

    if obj is not None:
        theory_name = str(obj.get("theory_name", "")).strip()
        core_claim = str(obj.get("core_claim", "")).strip()
        mechanism = str(obj.get("mechanism", "")).strip()
        explanatory_power = str(obj.get("explanatory_power", "")).strip()
        novelty_justification = str(obj.get("novelty_justification", "")).strip()
        preds = obj.get("predictions", [])
        if isinstance(preds, list):
            predictions = [str(p).strip() for p in preds if str(p).strip()]
        elif isinstance(preds, str):
            predictions = [preds.strip()] if preds.strip() else []
        else:
            predictions = []
    else:
        theory_name = extract_json_string_field(text, "theory_name")
        core_claim = extract_json_string_field(text, "core_claim")
        mechanism = extract_json_string_field(text, "mechanism")
        explanatory_power = extract_json_string_field(text, "explanatory_power")
        novelty_justification = extract_json_string_field(text, "novelty_justification")
        predictions = extract_predictions(text)

    if not theory_name:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        theory_name = lines[0][:140] if lines else "Untitled Theory"

    return {
        "theory_name": theory_name,
        "core_claim": core_claim,
        "mechanism": mechanism,
        "predictions": predictions,
        "explanatory_power": explanatory_power,
        "novelty_justification": novelty_justification,
        "raw_text": text,
    }


def normalize_name_format(name: str, abbr: str) -> str:
    name = re.sub(r"\s+", " ", str(name or "").strip())
    abbr = re.sub(r"\s+", "", str(abbr or "").strip())
    if not name:
        return "Untitled Theory"

    # Keep internal acronyms as-is while title-casing plain tokens.
    parts = []
    for tok in name.split(" "):
        if tok.isupper() and len(tok) > 1:
            parts.append(tok)
        else:
            parts.append(tok[:1].upper() + tok[1:] if tok else tok)
    name = " ".join(parts)

    # If name already has a trailing abbreviation in parentheses, keep it as-is.
    m = re.search(r"\(([A-Za-z0-9\-]{2,})\)\s*$", name)
    if m:
        return name

    if abbr:
        long_name_word_count = len([w for w in re.split(r"\W+", name) if w])
        if long_name_word_count >= 3:
            return f"{name} ({abbr})"
    return name


def extract_name_abbreviation(theory_name: str) -> str:
    m = re.search(r"\(([A-Za-z0-9\-]{2,})\)\s*$", str(theory_name or ""))
    return m.group(1).upper() if m else ""


def collapse_duplicate_abbreviation_suffixes(name: str) -> str:
    """Collapse names like 'X (ABC) (ABCD)' into 'X (ABC)' to avoid duplicate suffixes."""
    return re.sub(r"\s*\(([A-Za-z0-9\-]{2,})\)\s*\(([A-Za-z0-9\-]{2,})\)\s*$", r" (\1)", str(name or "").strip())


def synthesize_characterization(parsed: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ["core_claim", "mechanism", "explanatory_power", "novelty_justification"]:
        val = re.sub(r"\s+", " ", str(parsed.get(key, "") or "").strip())
        if val:
            parts.append(val)
    preds = parsed.get("predictions", [])
    if isinstance(preds, list) and preds:
        pred_text = "; ".join(str(p).strip() for p in preds if str(p).strip())
        if pred_text:
            parts.append(f"Predictions include: {pred_text}.")
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if text:
        return text
    raw = re.sub(r"\s+", " ", str(parsed.get("raw_text", "")).strip())
    if raw:
        return raw
    return "No characterization available from source response."


def infer_model_name(rec: Dict[str, Any], default_model_name: str) -> str:
    model = str(rec.get("model", "") or "").strip()
    if model:
        return model
    model_id = str(rec.get("model_id", "") or "").strip()
    if model_id and "/" in model_id:
        return model_id.split("/", 1)[1]
    if model_id:
        return model_id
    return default_model_name


def infer_sample_id(rec: Dict[str, Any], global_idx: int) -> int:
    sample = rec.get("sample_id", None)
    if isinstance(sample, int):
        return sample
    try:
        return int(str(sample).strip())
    except Exception:
        return global_idx


def fallback_abbreviation(theory_name: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", theory_name) if len(w) > 2]
    if len(words) >= 3:
        return "".join(w[0].upper() for w in words[:6])
    if words:
        return words[0][:4].upper()
    return "AI"


def build_llm_prompt(parsed: Dict[str, Any]) -> str:
    parsed_payload = {
        "theory_name": parsed.get("theory_name", ""),
        "core_claim": parsed.get("core_claim", ""),
        "mechanism": parsed.get("mechanism", ""),
        "predictions": parsed.get("predictions", []),
        "explanatory_power": parsed.get("explanatory_power", ""),
        "novelty_justification": parsed.get("novelty_justification", ""),
    }
    parsed_payload_json = json.dumps(parsed_payload, ensure_ascii=False, indent=2)

    preds = parsed.get("predictions", [])
    preds_text = "\n".join(f"- {p}" for p in preds) if preds else "- None"
    return (
        "You are a scientific editor for a consciousness-theory dataset.\n"
        "Return STRICT JSON only. No markdown, no extra keys, no commentary.\n\n"
        "Required output JSON fields and constraints:\n"
        "- theory_name: string, Title Case; if >= 3 words, include one abbreviation in parentheses at end when available.\n"
        "- abbreviation: string, uppercase acronym matching the final parenthetical acronym in theory_name when present.\n"
        "- key_proponents: string; if unknown use exactly 'AI-generated'.\n"
        "- characterization: objective scientific mechanism description, target ~100 words (acceptable 90-110 words).\n\n"
        "Goals:\n"
        "1) Standardize theory name format.\n"
        "2) Write an objective, mechanism-focused characterization around 100 words (target 90-110).\n"
        "3) Do not invent citations or historical facts unless clearly present in source text.\n\n"
        "Parsed source payload (from one item's response JSON string):\n"
        f"{parsed_payload_json}\n\n"
        "Expanded parsed source fields:\n"
        f"Theory Name: {parsed.get('theory_name', '')}\n"
        f"Core Claim: {parsed.get('core_claim', '')}\n"
        f"Mechanism: {parsed.get('mechanism', '')}\n"
        f"Predictions:\n{preds_text}\n"
        f"Explanatory Power: {parsed.get('explanatory_power', '')}\n"
        f"Novelty Justification: {parsed.get('novelty_justification', '')}\n"
    )


def call_openrouter_structured(
    client: OpenAI,
    model_id: str,
    parsed: Dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    base_schema = {
        "name": "ai_theory_record",
        "schema": {
            "type": "object",
            "properties": {
                "theory_name": {"type": "string"},
                "abbreviation": {"type": "string"},
                "key_proponents": {"type": "string"},
                "characterization": {"type": "string"},
            },
            "required": [
                "theory_name",
                "abbreviation",
                "key_proponents",
                "characterization",
            ],
            "additionalProperties": False,
        },
    }

    strict_schema = {
        **base_schema,
        "strict": True,
    }

    relaxed_schema = {
        **base_schema,
    }

    # Keep structured output as first priority, but mirror collect_ai_responses.py fallbacks
    # to handle provider-routing limitations for some model+parameter combinations.
    attempts = [
        {
            "messages": [{"role": "user", "content": build_llm_prompt(parsed)}],
            "response_format": {"type": "json_schema", "json_schema": strict_schema},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"provider": {"require_parameters": True}},
        },
        {
            "messages": [{"role": "user", "content": build_llm_prompt(parsed)}],
            "response_format": {"type": "json_schema", "json_schema": relaxed_schema},
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        {
            "messages": [{
                "role": "user",
                "content": build_llm_prompt(parsed) + "\n\nReturn valid JSON only. No markdown fences.",
            }],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    ]

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            completion = client.chat.completions.create(model=model_id, **kwargs)
            text = (completion.choices[0].message.content or "").strip()
            parsed_json: Optional[Dict[str, Any]] = None
            try:
                parsed_json = json.loads(text)
            except Exception:
                parsed_json = try_parse_json_block(text)
            if parsed_json is None:
                raise ValueError("Model output is not valid JSON")
            return parsed_json
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"OpenRouter formatting call failed: {last_error}")


def coerce_record_fields(formatted: Dict[str, Any], source_name: str) -> Dict[str, str]:
    theory_name = str(formatted.get("theory_name", "")).strip()
    abbreviation = str(formatted.get("abbreviation", "")).strip()
    if not abbreviation:
        abbreviation = fallback_abbreviation(theory_name or source_name)
    abbreviation = re.sub(r"[^A-Za-z0-9\-]", "", abbreviation).upper()
    if not theory_name:
        theory_name = source_name or "Untitled Theory"
    theory_name = collapse_duplicate_abbreviation_suffixes(theory_name)
    embedded_abbr = extract_name_abbreviation(theory_name)
    if embedded_abbr:
        abbreviation = embedded_abbr
    theory_name = normalize_name_format(theory_name, abbreviation)
    embedded_abbr_after = extract_name_abbreviation(theory_name)
    if embedded_abbr_after:
        abbreviation = embedded_abbr_after

    characterization = re.sub(r"\s+", " ", str(formatted.get("characterization", "")).strip())

    return {
        "theory_name": theory_name,
        "abbreviation": abbreviation,
        "key_proponents": str(formatted.get("key_proponents", "")).strip() or "AI-generated",
        "characterization": characterization,
    }


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return [x for x in data if isinstance(x, dict)]


def load_existing_ids(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    ids: Set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ids.add(int(str(row.get("id", "")).strip()))
            except Exception:
                continue
    return ids


def write_csv(path: Path, rows: List[Dict[str, Any]], append: bool) -> None:
    fieldnames = [
        "id",
        "model",
        "sample_id",
        "theory_name",
        "abbreviation",
        "key_proponents",
        "characterization",
        "source_theory_name",
    ]
    file_exists = path.exists()
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_next_unprocessed_index(existing_ids: Set[int], total_records: int) -> int:
    """Return the first 0-based record index not yet present in output ids."""
    for idx in range(total_records):
        if (idx + 1) not in existing_ids:
            return idx
    return total_records


def process_records(
    input_path: Path,
    output_path: Path,
    model_id: str,
    start_index: Optional[int],
    batch_size: Optional[int],
    delay: float,
    temperature: float,
    max_tokens: int,
    append_output: bool,
    use_next_batch: bool,
    default_model_name: str,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    all_records = read_json_list(input_path)
    total_records = len(all_records)

    existing_ids = load_existing_ids(output_path) if append_output else set()

    if use_next_batch:
        start_index = find_next_unprocessed_index(existing_ids, total_records)
        if start_index >= total_records:
            print("All records already processed. Nothing to do.")
            return

    if start_index is None:
        start_index = 0

    if start_index < 0 or start_index >= total_records:
        raise ValueError(f"start-index must be in [0, {max(total_records - 1, 0)}], got {start_index}")

    end_index = total_records if batch_size is None else min(start_index + batch_size, total_records)
    selected_indices = list(range(start_index, end_index))

    print(
        f"Selected records: {len(selected_indices)} (indices {start_index}..{end_index - 1}) out of {total_records}"
    )

    if append_output and existing_ids:
        print(f"Existing output detected with {len(existing_ids)} id(s); duplicate ids will be skipped.")

    client = build_openrouter_client()
    out_rows: List[Dict[str, Any]] = []

    total = len(selected_indices)
    for n, global_idx in enumerate(selected_indices, start=1):
        rec = all_records[global_idx]
        output_id = global_idx + 1

        if append_output and output_id in existing_ids:
            print(f"[{n}/{total}] Skipped existing id={output_id}")
            continue

        parsed = parse_response_record(rec)
        source_name = parsed.get("theory_name", "")

        try:
            formatted = call_openrouter_structured(
                client=client,
                model_id=model_id,
                parsed=parsed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            clean = coerce_record_fields(formatted, source_name=source_name)
        except Exception as exc:
            # Fallback row if one formatting request fails.
            fallback_name = normalize_name_format(source_name or "Untitled Theory", "")
            clean = {
                "theory_name": fallback_name,
                "abbreviation": fallback_abbreviation(fallback_name),
                "key_proponents": "AI-generated",
                "characterization": synthesize_characterization(parsed),
            }
            print(f"[{n}/{total}] Warning: formatting failed, fallback used ({exc})")

        if not clean.get("characterization", "").strip():
            clean["characterization"] = synthesize_characterization(parsed)

        out_rows.append(
            {
                "id": output_id,
                "model": infer_model_name(rec, default_model_name=default_model_name),
                "sample_id": infer_sample_id(rec, global_idx=global_idx),
                "theory_name": clean["theory_name"],
                "abbreviation": clean["abbreviation"],
                "key_proponents": clean["key_proponents"],
                "characterization": clean["characterization"],
                "source_theory_name": source_name,
            }
        )

        print(f"[{n}/{total}] Processed id={output_id}: {source_name}")
        if delay > 0:
            time.sleep(delay)

    write_csv(output_path, out_rows, append=append_output)
    print(f"\nWrote {len(out_rows)} row(s) to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consciousness_theories_ai.csv from all_responses.json via OpenRouter")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Display name for model (default: claude-3.7-sonnet)")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="OpenRouter model id")
    parser.add_argument("--max-items", type=int, default=None, help="Legacy alias for --batch-size (process first N from --start-index)")
    parser.add_argument("--start-index", type=int, default=0, help="0-based index to start processing from")
    parser.add_argument("--batch-size", type=int, default=None, help="Number of records to process in this run")
    parser.add_argument(
        "--next-batch-size",
        type=int,
        default=None,
        help="Auto-detect next unprocessed start index from output CSV and process this many records",
    )
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between API calls in seconds")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=450, help="Max completion tokens for formatter call")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite output CSV instead of appending/skipping existing ids",
    )
    args = parser.parse_args()

    if args.next_batch_size is not None and args.next_batch_size <= 0:
        raise ValueError("--next-batch-size must be > 0")

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    batch_size = args.batch_size if args.batch_size is not None else args.max_items
    use_next_batch = args.next_batch_size is not None
    if use_next_batch:
        batch_size = args.next_batch_size

    print(f"Using OpenRouter model: {args.model_name} ({args.model_id})")
    process_records(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model_id,
        start_index=None if use_next_batch else args.start_index,
        batch_size=batch_size,
        delay=args.delay,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        append_output=not args.overwrite_output,
        use_next_batch=use_next_batch,
        default_model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
