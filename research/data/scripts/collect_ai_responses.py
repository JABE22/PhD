#!/usr/bin/env python3
"""
Collect AI responses for empirical tests.

This script queries multiple AI models via OpenRouter, Perplexity, and Mistral APIs
and saves their responses for analysis in the test notebooks.

Usage:
    python research/data/scripts/collect_ai_responses.py --test test3_theory_generation --models gemini-3.1-pro-preview --n-samples 24 --provider openrouter

Requirements:
    - .env file with API keys:
      OPENROUTER_API_KEY=your_key_here
      PERPLEXITY_API_KEY=your_key_here
      MISTRAL_API_KEY=your_key_here
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv # type: ignore
from openai import OpenAI # type: ignore
from perplexity import Perplexity # type: ignore
#from transformers.convert_slow_tokenizers_checkpoints_to_fast import args

# Load environment variables
load_dotenv()

# API clients
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER2_API_KEY", ""),
)

perplexity_client = Perplexity(
    api_key=os.getenv("PERPLEXITY_API_KEY", "")
)

mistral_client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=os.getenv("MISTRALAI_API_KEY", ""),
)


# Model configurations by provider
PROVIDER_MODEL_CONFIGS = {
    "openrouter": {
        # OpenRouter models (prefix with provider/)
        "gpt-5.2": {"api": "openrouter", "id": "openai/gpt-5.2"},
        "claude-3.7-sonnet": {"api": "openrouter", "id": "anthropic/claude-3.7-sonnet"},
        "gemini-3.1-pro-preview": {"api": "openrouter", "id": "google/gemini-3.1-pro-preview"},
        "llama-3.3-70b-instruct": {"api": "openrouter", "id": "meta-llama/llama-3.3-70b-instruct"},
        "deepseek-v3.2": {"api": "openrouter", "id": "deepseek/deepseek-v3.2"},
        "mistral-large": {"api": "openrouter", "id": "mistralai/mistral-large"},
        "perplexity-sonar-pro": {"api": "openrouter", "id": "perplexity/sonar-pro"},
    },
    "perplexity": {
        "gpt-5.2": {"api": "perplexity-agent", "id": "openai/gpt-5.2"},
        "claude-4.5-sonnet": {"api": "perplexity-agent", "id": "anthropic/claude-sonnet-4-5"},
        "perplexity-sonar": {"api": "perplexity-agent", "id": "perplexity/sonar"},
    },
}


# Token limit for responses (adjust as needed)
MAX_TOKENS = 3000
MAX_WORDS = 750

# Map test names to their folder structure
TEST_FOLDER_MAPPING = {
    "test1_ontological_innovation": ("test1_ontological-innovation", "test1_ontological_innovation_responses"),
    "test2_epistemic_agency": ("test2_epistemic-agency", "test2_epistemic_agency_responses"),
    "test3_theory_generation": ("test3_theory-generation", "test3_theory_generation_responses"),
    "test4_category_recognition": ("test4_category-recognition", "test4_category_recognition_responses"),
}

PROMPTS = {
    "test1_ontological_innovation": """You are participating in a scientific study on AI creativity and conceptual innovation.

BACKGROUND:
We have defined a sensory system with 8 modalities: visual, auditory, tactile, olfactory, gustatory, proprioceptive, vestibular, and interoceptive. Each modality has specific:
- Physical basis (what it detects)
- Information content (what data it provides)
- Functional role (what it enables)

YOUR TASK:
Propose a NINTH sensory modality that would be genuinely novel - not just a refinement or combination of existing ones. Your proposal should include:

1. Modality Name: Clear, descriptive name
2. Physical Basis: What physical phenomenon it detects
3. Information Content: What specific information it provides
4. Functional Role: What capabilities it enables
5. Justification: Why this is genuinely novel (not reducible to existing modalities)

Be creative but rigorous. Think beyond obvious extensions like "electromagnetic sensing" or "radiation detection."

Provide your response in the following JSON format:
{
  "proposal_name": "Name of the ninth modality",
  "physical_basis": "Description of physical detection mechanism",
  "information_content": "Description of information provided",
  "functional_role": "Description of enabled capabilities",
  "justification": "Explanation of why this is genuinely novel"
}
""",

    "test2_epistemic_agency": """You are participating in a scientific study on AI epistemic agency and question-asking capabilities.

BACKGROUND:
Epistemic agency involves not just answering questions, but asking genuinely productive questions that advance inquiry. We are interested in whether AI can generate questions that:
- Transcend given conceptual frameworks
- Identify novel problem spaces
- Challenge underlying assumptions

YOUR TASK:
Given the following philosophical problem:

"What is the relationship between phenomenal consciousness (subjective experience) and access consciousness (information available for reasoning and reporting)?"

Generate 5 ORIGINAL research questions that:
1. Are not easily found in existing literature
2. Challenge or transcend the phenomenal/access distinction itself
3. Open up genuinely new avenues of inquiry
4. Demonstrate deep understanding of the conceptual space

For each question, explain:
- Why it's original (not standard in literature)
- What conceptual framework it transcends or challenges
- What new problem space it opens

Provide your response as a JSON array:
[
  {
    "question": "Your original research question",
    "originality_explanation": "Why this question is original",
    "framework_transcendence": "What framework it transcends/challenges",
    "new_problem_space": "What new inquiry space it opens"
  },
  ...
]
""",

    "test3_theory_generation": """You are participating in a scientific study on AI theory generation capabilities.

BACKGROUND:
We are investigating whether AI can generate genuinely novel theories of consciousness, or whether it primarily recombines existing theoretical elements. Dominant theories include:
- Global Workspace Theory (GWT)
- Integrated Information Theory (IIT)
- Higher-Order Thought Theory (HOT)
- Predictive Processing (PP)
- Attention Schema Theory (AST)

YOUR TASK:
Generate a NOVEL theory of consciousness that:
1. Explains the relationship between neural processes and subjective experience
2. Is not merely a hybrid of existing theories
3. Makes specific, testable predictions
4. Addresses the hard problem of consciousness

Your theory should include:
- Core Claim: Central thesis about consciousness
- Mechanism: How consciousness arises from physical processes
- Predictions: Testable empirical predictions
- Explanatory Power: What phenomena it explains
- Novelty: Why it's not reducible to existing theories

Provide your response in JSON format:
{
  "theory_name": "Name of your theory",
  "core_claim": "Central thesis",
  "mechanism": "How consciousness arises",
  "predictions": ["Prediction 1", "Prediction 2", "Prediction 3"],
  "explanatory_power": "What phenomena it explains",
  "novelty_justification": "Why this is genuinely novel"
}
""",

    "test4_category_recognition": """You are participating in a scientific study on AI category recognition and philosophical sophistication.

BACKGROUND:
Philosophical categories are often contested, vague, or involve important distinctions that are easy to miss. We are testing whether AI:
- Recognizes when categories are philosophically contested
- Avoids category mistakes
- Maintains important distinctions

YOUR TASK:
Analyze the following statement and identify any category mistakes, conflations, or unrecognized contestedness:

"Machine learning systems have beliefs about the world, which they form through training. These beliefs are stored in their weights as knowledge representations. When they make predictions, they are exercising their understanding of the domain. Therefore, advanced AI systems genuinely know things in the same sense that humans do."

Your analysis should:
1. Identify key philosophical categories invoked (beliefs, knowledge, understanding, etc.)
2. Note which categories are philosophically contested
3. Identify any category mistakes or illegitimate conflations
4. Explain what important distinctions are being missed
5. Provide a more nuanced analysis

Provide your response in JSON format:
{
  "categories_identified": ["category1", "category2", ...],
  "contested_categories": [
    {
      "category": "name",
      "why_contested": "explanation"
    }
  ],
  "category_mistakes": [
    {
      "mistake": "description",
      "why_illegitimate": "explanation"
    }
  ],
  "missed_distinctions": [
    {
      "distinction": "X vs Y",
      "importance": "why this matters"
    }
  ],
  "nuanced_analysis": "Your more sophisticated take"
}
""",
}

PERPLEXITY_RESPONSE_SCHEMAS = {
    "test1_ontological_innovation": {
        "type": "json_schema",
        "json_schema": {
            "name": "ontological_innovation",
            "schema": {
                "type": "object",
                "properties": {
                    "proposal_name": {"type": "string"},
                    "physical_basis": {"type": "string"},
                    "information_content": {"type": "string"},
                    "functional_role": {"type": "string"},
                    "justification": {"type": "string"},
                },
                "required": [
                    "proposal_name",
                    "physical_basis",
                    "information_content",
                    "functional_role",
                    "justification",
                ],
                "additionalProperties": False,
            },
        },
    },
    "test2_epistemic_agency": {
        "type": "json_schema",
        "json_schema": {
            "name": "epistemic_agency_questions",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "originality_explanation": {"type": "string"},
                        "framework_transcendence": {"type": "string"},
                        "new_problem_space": {"type": "string"},
                    },
                    "required": [
                        "question",
                        "originality_explanation",
                        "framework_transcendence",
                        "new_problem_space",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    },
    "test3_theory_generation": {
        "type": "json_schema",
        "json_schema": {
            "name": "consciousness_theory",
            "schema": {
                "type": "object",
                "properties": {
                    "theory_name": {"type": "string"},
                    "core_claim": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "predictions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "explanatory_power": {"type": "string"},
                    "novelty_justification": {"type": "string"},
                },
                "required": [
                    "theory_name",
                    "core_claim",
                    "mechanism",
                    "predictions",
                    "explanatory_power",
                    "novelty_justification",
                ],
                "additionalProperties": False,
            },
        },
    },
    "test4_category_recognition": {
        "type": "json_schema",
        "json_schema": {
            "name": "category_recognition_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "categories_identified": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "contested_categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "why_contested": {"type": "string"},
                            },
                            "required": ["category", "why_contested"],
                            "additionalProperties": False,
                        },
                    },
                    "category_mistakes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mistake": {"type": "string"},
                                "why_illegitimate": {"type": "string"},
                            },
                            "required": ["mistake", "why_illegitimate"],
                            "additionalProperties": False,
                        },
                    },
                    "missed_distinctions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "distinction": {"type": "string"},
                                "importance": {"type": "string"},
                            },
                            "required": ["distinction", "importance"],
                            "additionalProperties": False,
                        },
                    },
                    "nuanced_analysis": {"type": "string"},
                },
                "required": [
                    "categories_identified",
                    "contested_categories",
                    "category_mistakes",
                    "missed_distinctions",
                    "nuanced_analysis",
                ],
                "additionalProperties": False,
            },
        },
    },
}


def get_client_and_model_id(provider: str, model_name: str):
    """Get appropriate API client and model ID."""
    if provider not in PROVIDER_MODEL_CONFIGS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_MODEL_CONFIGS.keys())}")

    model_configs = PROVIDER_MODEL_CONFIGS[provider]
    if model_name not in model_configs:
        raise ValueError(f"Unknown model for provider '{provider}': {model_name}. Available: {list(model_configs.keys())}")

    config = model_configs[model_name]
    api = config["api"]
    model_id = config["id"]
    
    if api == "perplexity-agent":
        return perplexity_client, model_id
    elif api == "openrouter":
        return openrouter_client, model_id
    elif api == "mistral":
        return mistral_client, model_id
    else:
        raise ValueError(f"Unknown API: {api}")


def query_model(
    provider: str,
    test_name: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS,
) -> Dict:
    """Query a model (Perplexity Agent, OpenRouter, or Mistral) and return response with metadata."""
    client, model_id = get_client_and_model_id(provider, model_name)
    api = PROVIDER_MODEL_CONFIGS[provider][model_name]["api"]

    def extract_perplexity_response_text(resp) -> str:
        """Extract assistant text from Responses API output items.

        Preferred source per API docs: output[].content[].text where
        output item is assistant message and content type is output_text.
        """
        def try_parse_json_candidate(text: Optional[str]) -> Optional[str]:
            if not text or not isinstance(text, str):
                return None

            raw = text.strip()
            if not raw:
                return None

            try:
                json.loads(raw)
                return raw
            except json.JSONDecodeError:
                pass

            # Try extracting likely JSON substring from mixed text
            candidates = []
            obj_start, obj_end = raw.find("{"), raw.rfind("}")
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                candidates.append(raw[obj_start:obj_end + 1])

            arr_start, arr_end = raw.find("["), raw.rfind("]")
            if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                candidates.append(raw[arr_start:arr_end + 1])

            for candidate in sorted(candidates, key=len, reverse=True):
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

            return None

        def collect_all_text_leaves(node, out: List[str]) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    collect_all_text_leaves(value, out)
            elif isinstance(node, list):
                for item in node:
                    collect_all_text_leaves(item, out)
            elif isinstance(node, str):
                out.append(node)

        # Try dictionary payload first (stable across SDK object wrappers)
        payload = None
        if hasattr(resp, "model_dump"):
            try:
                payload = resp.model_dump()
            except Exception:
                payload = None

        text_sources: List[str] = []

        if isinstance(payload, dict):
            parts = []
            for item in payload.get("output", []) or []:
                if item.get("type") != "message" or item.get("role") != "assistant":
                    continue
                for content_item in item.get("content", []) or []:
                    if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                        text_piece = content_item["text"]
                        parts.append(text_piece)
                        text_sources.append(text_piece)
            if parts:
                joined = "".join(parts)
                text_sources.append(joined)

            # Also collect every string in payload for robust recovery.
            collect_all_text_leaves(payload, text_sources)

        # Fallback to SDK convenience field
        output_text = getattr(resp, "output_text", "") or ""
        if output_text:
            text_sources.append(output_text)

        valid_json_candidates = []
        for source_text in text_sources:
            candidate = try_parse_json_candidate(source_text)
            if candidate:
                valid_json_candidates.append(candidate)

        if valid_json_candidates:
            return max(valid_json_candidates, key=len)

        # If no valid JSON was found anywhere, keep the most informative raw text.
        if text_sources:
            return max(text_sources, key=len)

        return ""

    try:
        # 1) Perplexity Agent API (Perplexity SDK, responses.create)
        if api == "perplexity-agent":
            response_format = PERPLEXITY_RESPONSE_SCHEMAS.get(test_name)
            resp = client.responses.create(
                model=model_id,
                input=prompt,
                response_format=response_format,
                max_output_tokens=max_tokens,
                # temperature=temperature, # Perplexity API may not support temperature - check docs
                # Optional: tools / instructions, e.g. web_search
                )

            if resp.status != "completed":
                return {
                    "model": model_name,
                    "model_id": model_id,
                    "error": f"status={resp.status}, error={getattr(resp, 'error', None)}",
                    "timestamp": datetime.now().isoformat(),
                    "temperature": None,
                }

            usage = getattr(resp, "usage", None)
            return {
                "model": model_name,
                "model_id": model_id,
                "response": extract_perplexity_response_text(resp),
                "status": getattr(resp, "status", None),
                "finish_reason": getattr(resp, "finish_reason", None),
                "usage": {
                    "prompt_tokens": getattr(usage, "input_tokens", None),
                    "completion_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                } if usage else None,
                "timestamp": datetime.now().isoformat(),
                "temperature": None,
            }

        # 2) OpenRouter (OpenAI-compatible chat.completions + structured outputs)
        elif api == "openrouter":
            schema_wrapper = PERPLEXITY_RESPONSE_SCHEMAS.get(test_name)
            base_messages = [{"role": "user", "content": prompt}]
            last_err = None

            # Attempt 1: strict schema + require_parameters (current behavior)
            attempts = []

            if schema_wrapper:
                json_schema = schema_wrapper.get("json_schema", {})
                strict_response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        **json_schema,
                        "strict": True,
                    },
                }
                attempts.append({
                    "messages": base_messages,
                    "response_format": strict_response_format,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": {"provider": {"require_parameters": True}},
                })

                # Attempt 2: schema, but no strict+routing lock
                relaxed_response_format = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
                attempts.append({
                    "messages": base_messages,
                    "response_format": relaxed_response_format,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                })
            # Attempt 3: no response_format, ask for JSON in prompt
            attempts.append({
                "messages": [
                    {
                        "role": "user",
                        "content": prompt + "\n\nReturn valid JSON only. No markdown fences."
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            })

            completion = None
            for kwargs in attempts:
                try:
                    completion = client.chat.completions.create(
                        model=model_id,
                        **kwargs,
                    )
                    break
                except Exception as e:
                    last_err = e
                    continue

            if completion is None:
                raise last_err if last_err else RuntimeError("OpenRouter request failed")

            usage = getattr(completion, "usage", None)
            return {
                "model": model_name,
                "model_id": model_id,
                "response": completion.choices[0].message.content,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                } if usage else None,
                "timestamp": datetime.now().isoformat(),
                "temperature": temperature,
            }

        # 3) Mistral (OpenAI-compatible chat.completions)
        elif api == "mistral":
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            usage = getattr(completion, "usage", None)
            return {
                "model": model_name,
                "model_id": model_id,
                "response": completion.choices[0].message.content,
                "finish_reason": completion.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                } if usage else None,
                "timestamp": datetime.now().isoformat(),
                "temperature": temperature,
            }

        else:
            raise ValueError(f"Unknown API type for model {model_name}: {api}")

    except Exception as e:
        return {
            "model": model_name,
            "model_id": model_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "temperature": temperature,
        }



def collect_responses(
    provider: str,
    test_name: str,
    models: List[str],
    n_samples: int,
    temperature: float = 0.7,
    temperature_schedule: Optional[List[float]] = None,
    max_tokens: int = MAX_TOKENS,
    delay: float = 2.0,
    output_dir: Optional[Path] = None,
):
    """Collect responses from multiple models for a given test."""
    if output_dir is None:
        if test_name not in TEST_FOLDER_MAPPING:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(TEST_FOLDER_MAPPING.keys())}")
        test_folder, responses_folder = TEST_FOLDER_MAPPING[test_name]
        output_dir = Path("research") / test_folder / responses_folder
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prompt
    if test_name not in PROMPTS:
        raise ValueError(f"Unknown test: {test_name}. Available: {list(PROMPTS.keys())}")
    
    prompt = PROMPTS[test_name]
    
    # Save prompt (idempotent)
    with open(output_dir / "prompt.txt", "w") as f:
        f.write(prompt)
    
    print(f"\n{'='*60}")
    print(f"COLLECTING RESPONSES: {test_name}")
    print(f"{'='*60}")
    print(f"Provider: {provider}")
    print(f"Models: {', '.join(models)}")
    print(f"Samples per model: {n_samples}")
    print(f"Temperature: {temperature}")
    if temperature_schedule:
        print(f"Temperature schedule: {temperature_schedule}")
    print(f"Max output tokens: {max_tokens}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    run_responses = []

    def response_is_valid_json(text: Optional[str]) -> bool:
        if not text or not isinstance(text, str):
            return False
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def slot_is_final(file_path: Path) -> bool:
        if not file_path.exists():
            return False
        try:
            with file_path.open("r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        finish_reason = data.get("finish_reason")
        if finish_reason == "stop":
            return True

        if finish_reason is None:
            return response_is_valid_json(data.get("response"))

        return False

    for model_name in models:
        print(f"\n[{model_name}] Evaluating slots 0..{n_samples - 1}...")

        target_indices = []
        for sample_index in range(n_samples):
            filename = f"{model_name}_sample_{sample_index:03d}.json"
            file_path = output_dir / filename
            if not slot_is_final(file_path):
                target_indices.append(sample_index)

        if not target_indices:
            print(f"[{model_name}] All requested slots already finalized (stop or valid JSON with null finish_reason). Skipping.")
            continue

        print(f"[{model_name}] Need to (re)generate {len(target_indices)} slot(s): {target_indices}")

        for query_idx, sample_index in enumerate(target_indices, start=1):
            print(f"  Query {query_idx}/{len(target_indices)} for sample_{sample_index:03d}...", end=" ", flush=True)

            current_temperature = (
                temperature_schedule[sample_index % len(temperature_schedule)]
                if temperature_schedule
                else temperature
            )
            
            response_data = query_model(
                provider=provider,
                test_name=test_name,
                model_name=model_name,
                prompt=prompt,
                temperature=current_temperature,
                max_tokens=max_tokens,
            )
            
            response_data["provider"] = provider
            response_data["test"] = test_name
            response_data["sample_id"] = sample_index
            response_data["prompt"] = prompt
            
            # Save individual response (per-run, per-model files)
            filename = f"{model_name}_sample_{sample_index:03d}.json"
            with open(output_dir / filename, "w") as f:
                json.dump(response_data, f, indent=2)
            
            run_responses.append(response_data)
            
            if "error" in response_data:
                print(f"ERROR: {response_data['error']}")
            else:
                print("✓")
            
            time.sleep(delay)
        
        print(f"[{model_name}] Complete.")
    
    # ---- NEW: append to existing all_responses.json instead of overwriting ----
    all_path = output_dir / "all_responses.json"
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

    merged = existing + run_responses

    with all_path.open("w") as f:
        json.dump(merged, f, indent=2)
    # --------------------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"New responses collected in this run: {len(run_responses)}")
    print(f"Total responses now in all_responses.json: {len(merged)}")
    print(f"Saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return run_responses



def list_available_models(provider: str):
    """List available models for the selected provider."""
    print("\n" + "="*60)
    print(f"AVAILABLE MODELS (provider={provider})")
    print("="*60)

    if provider == "openrouter":
        try:
            openrouter_models = openrouter_client.models.list()
            print(f"\nTotal models available from API: {len(openrouter_models.data)}\n")

            # Group by upstream provider
            by_upstream_provider = {}
            for model in openrouter_models.data:
                upstream_provider = model.id.split('/')[0] if '/' in model.id else 'other'
                if upstream_provider not in by_upstream_provider:
                    by_upstream_provider[upstream_provider] = []
                by_upstream_provider[upstream_provider].append(model.id)

            for upstream_provider, model_ids in sorted(by_upstream_provider.items()):
                print(f"\n{upstream_provider.upper()} ({len(model_ids)} models):")
                for model_id in sorted(model_ids)[:10]:  # Show first 10
                    print(f"  - {model_id}")
                if len(model_ids) > 10:
                    print(f"  ... and {len(model_ids) - 10} more")

        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
    else:
        print("\nPerplexity SDK model listing is not available here; showing configured models only.")
    
    print("\n" + "="*60)
    print("CONFIGURED MODELS IN THIS SCRIPT")
    print("="*60)
    for name, config in PROVIDER_MODEL_CONFIGS[provider].items():
        print(f"  {name:30} -> {config['id']:50} ({config['api']})")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect AI responses for empirical tests"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        choices=list(PROVIDER_MODEL_CONFIGS.keys()),
        help="Service provider to use (default: openrouter)",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=False,
        choices=list(PROMPTS.keys()) + ["all"],
        help="Which test to run (or 'all' for all tests)",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=False,
        help="Comma-separated list of model names",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per model (default: 50)",
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.7",
        help="Sampling temperature as float (e.g., 0.7) or comma-separated schedule (e.g., 0.2,0.5,0.7,0.9)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"Max output tokens per response (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        list_available_models(args.provider)
        return
    
    # Validate arguments
    if not args.test or not args.models:
        parser.print_help()
        print("\nExample usage:")
        print("  python research/data/scripts/collect_ai_responses.py --test test1 --models gpt-4o,claude-3.5-sonnet --n-samples 50")
        print("  python research/data/scripts/collect_ai_responses.py --list-models")
        return
    
    # Parse models
    model_configs = PROVIDER_MODEL_CONFIGS[args.provider]
    if args.models.lower() == "all":
        models = list(model_configs.keys())
    else:
        models = [m.strip() for m in args.models.split(",")]

    # Parse temperature (single value or schedule)
    temperature_schedule = None
    temperature_arg = args.temperature.strip()
    try:
        if "," in temperature_arg:
            temperature_schedule = [
                float(value.strip())
                for value in temperature_arg.split(",")
                if value.strip()
            ]
            if not temperature_schedule:
                raise ValueError("temperature schedule is empty")
            temperature = temperature_schedule[0]
        else:
            temperature = float(temperature_arg)
    except ValueError:
        print(
            "\nError: --temperature must be a float (e.g., 0.7) or a comma-separated list "
            "(e.g., 0.2,0.5,0.7,0.9)"
        )
        return

    # Validate models
    invalid_models = [m for m in models if m not in model_configs]
    if invalid_models:
        print(f"\nError: Unknown models: {invalid_models}")
        print(f"Provider: {args.provider}")
        print(f"Available models: {list(model_configs.keys())}")
        return
    
    # Run collection
    if args.test == "all":
        tests = list(PROMPTS.keys())
    else:
        tests = [args.test]
    
    for test in tests:
        collect_responses(
            provider=args.provider,
            test_name=test,
            models=models,
            n_samples=args.n_samples,
            temperature=temperature,
            temperature_schedule=temperature_schedule,
            max_tokens=args.max_tokens,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
