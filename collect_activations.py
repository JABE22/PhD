#!/usr/bin/env python3
"""
Collect hidden states and attentions for mechanistic analysis.

This script reuses prompts from the existing AI response exports under
ai_responses/*/all_responses.json and runs a local open-source causal LM over
those prompts. It stores one artifact per prompt in data/activations so that
section_5-6_mechanistic.ipynb can load them without performing collection.

Example:
    python collect_activations.py \
        --model-name mistralai/Mistral-7B-v0.1 \
        --per-test-limit 4 \
        --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEST_EXPORTS = {
    "test1_ontological_innovation": Path("ai_responses/test1_ontological_innovation/all_responses.json"),
    "test2_epistemic_agency": Path("ai_responses/test2_epistemic_agency/all_responses.json"),
    "test3_theory_generation": Path("ai_responses/test3_theory_generation/all_responses.json"),
    "test4_category_recognition": Path("ai_responses/test4_category_recognition/all_responses.json"),
}


@dataclass(slots=True)
class ActivationCollectionConfig:
    model_name: str
    tests: list[str]
    per_test_limit: int | None = None
    max_prompts: int | None = None
    max_input_length: int = 1024
    max_new_tokens: int = 64
    output_dir: str = "data/activations"
    device: str = "auto"
    dtype: str = "auto"
    save_attentions: bool = False
    overwrite: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect activations for Section 5.6 mechanistic analysis.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Hugging Face causal LM identifier used for activation collection.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        help="Tests to include: all, test1_ontological_innovation, test2_epistemic_agency, test3_theory_generation, test4_category_recognition.",
    )
    parser.add_argument(
        "--per-test-limit",
        type=int,
        default=None,
        help="Maximum number of unique prompts to collect per test.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to collect overall after per-test sampling.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=1024,
        help="Tokenizer truncation length for the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated continuation tokens saved in the artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/activations",
        help="Directory where activation artifacts are written.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--save-attentions",
        action="store_true",
        help="Save attention tensors in addition to hidden states.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt artifacts if they already exist.",
    )
    return parser.parse_args(argv)


def _extract_prompt_text(record: dict) -> str:
    prompt_keys = [
        "prompt",
        "user_prompt",
        "input",
        "question",
        "query",
        "test_prompt",
        "instruction",
        "content",
    ]
    for key in prompt_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        user_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if user_messages:
            content = user_messages[-1].get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    return ""


def _load_records(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        records = payload.get("responses", payload.get("data", []))
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def load_prompt_table(test_names: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for test_name in test_names:
        export_path = DEFAULT_TEST_EXPORTS[test_name]
        if not export_path.exists():
            print(f"Skipping missing export: {export_path}")
            continue

        for index, record in enumerate(_load_records(export_path)):
            prompt = _extract_prompt_text(record)
            if not prompt:
                continue

            sample_id = record.get("sample_id", record.get("id", index))
            rows.append(
                {
                    "test_id": test_name,
                    "prompt_id": str(sample_id),
                    "prompt": prompt,
                    "source_model": record.get("model") or record.get("model_id") or "unknown",
                    "source_provider": record.get("provider", "unknown"),
                    "source_timestamp": record.get("timestamp"),
                    "source_response": record.get("response", ""),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "test_id",
                "prompt_id",
                "prompt",
                "source_model",
                "source_provider",
                "source_timestamp",
                "source_response",
            ]
        )

    prompt_df = pd.DataFrame(rows)
    prompt_df = prompt_df.drop_duplicates(subset=["test_id", "prompt"]).reset_index(drop=True)
    return prompt_df


def resolve_tests(requested_tests: Sequence[str]) -> list[str]:
    if list(requested_tests) == ["all"]:
        return list(DEFAULT_TEST_EXPORTS)

    invalid = sorted(set(requested_tests) - set(DEFAULT_TEST_EXPORTS))
    if invalid:
        raise ValueError(f"Unknown tests: {', '.join(invalid)}")
    return list(requested_tests)


def sample_prompts(prompt_df: pd.DataFrame, per_test_limit: int | None, max_prompts: int | None) -> pd.DataFrame:
    if prompt_df.empty:
        return prompt_df

    sampled_groups = []
    for test_id, group in prompt_df.groupby("test_id", sort=True):
        if per_test_limit is None:
            sampled_groups.append(group)
        else:
            sampled_groups.append(group.head(per_test_limit))

    sampled = pd.concat(sampled_groups, ignore_index=True)
    if max_prompts is not None:
        sampled = sampled.head(max_prompts).reset_index(drop=True)
    return sampled


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype_arg: str, device: str):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(model_name: str, device: str, dtype_arg: str):
    dtype = _resolve_dtype(dtype_arg, device)
    model_kwargs = {"torch_dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer


def collect_for_prompt(
    row: pd.Series,
    model,
    tokenizer,
    device: str,
    max_input_length: int,
    max_new_tokens: int,
    save_attentions: bool,
) -> dict:
    prompt = str(row["prompt"])
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        forward_outputs = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=save_attentions,
            use_cache=False,
        )
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

    generated_ids_cpu = generated_ids[0].detach().cpu()
    prompt_length = int(encoded["input_ids"].shape[1])
    input_ids_cpu = encoded["input_ids"][0].detach().cpu()
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu.tolist())
    generated_token_ids = generated_ids_cpu[prompt_length:]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids.tolist()) if len(generated_token_ids) else []

    return {
        "prompt_id": str(row["prompt_id"]),
        "test_id": str(row["test_id"]),
        "prompt": prompt,
        "source_model": str(row.get("source_model", "unknown")),
        "source_provider": str(row.get("source_provider", "unknown")),
        "source_timestamp": row.get("source_timestamp"),
        "source_response": row.get("source_response", ""),
        "collector_model_name": model.name_or_path,
        "input_ids": input_ids_cpu,
        "input_tokens": input_tokens,
        "hidden_states": tuple(t.detach().cpu() for t in forward_outputs.hidden_states),
        "attentions": tuple(t.detach().cpu() for t in forward_outputs.attentions) if save_attentions else None,
        "generated_ids": generated_ids_cpu,
        "generated_tokens": generated_tokens,
        "generated_text": tokenizer.decode(generated_ids_cpu, skip_special_tokens=True),
        "prompt_token_count": int(len(input_ids_cpu)),
        "generated_token_count": int(len(generated_token_ids)),
    }


def collect_activations(config: ActivationCollectionConfig) -> dict:
    selected_tests = resolve_tests(config.tests)
    prompt_df = load_prompt_table(selected_tests)
    prompt_df = sample_prompts(prompt_df, config.per_test_limit, config.max_prompts)
    if prompt_df.empty:
        raise ValueError("No prompts found in the selected response exports.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.device)
    print(f"Loading model {config.model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device, config.dtype)

    written = 0
    skipped = 0
    for _, row in prompt_df.iterrows():
        output_path = output_dir / f"{row['test_id']}__{row['prompt_id']}.pt"
        if output_path.exists() and not config.overwrite:
            skipped += 1
            continue

        artifact = collect_for_prompt(
            row=row,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_input_length=config.max_input_length,
            max_new_tokens=config.max_new_tokens,
            save_attentions=config.save_attentions,
        )
        torch.save(artifact, output_path)
        written += 1
        print(f"Saved {output_path}")

    summary = {
        "tests": selected_tests,
        "prompts_considered": int(len(prompt_df)),
        "files_written": int(written),
        "files_skipped": int(skipped),
        "output_directory": str(output_dir),
        "device": device,
        "config": asdict(config),
    }

    print("\nActivation collection complete.")
    print(f"  Tests: {', '.join(selected_tests)}")
    print(f"  Prompts considered: {summary['prompts_considered']}")
    print(f"  Files written: {written}")
    print(f"  Files skipped: {skipped}")
    print(f"  Output directory: {output_dir}")
    return summary


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    config = ActivationCollectionConfig(
        model_name=args.model_name,
        tests=list(args.tests),
        per_test_limit=args.per_test_limit,
        max_prompts=args.max_prompts,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        save_attentions=args.save_attentions,
        overwrite=args.overwrite,
    )
    return collect_activations(config)


if __name__ == "__main__":
    main()
