import os
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# === Perplexity Sonar ===
from openai import OpenAI as PerplexityClient  # per docs: OpenAI-compatible client [web:151]

# === SambaNova ===
from sambanova import SambaNova

# === Mistral ===
from mistralai import Mistral


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str


class UnifiedModelClient:
    """
    Thin wrapper that normalizes chat completions across:
      - Perplexity Sonar (OpenAI-compatible)
      - SambaNova (Llama-4-Maverick-17B-128E-Instruct)
      - Mistral (mistral-medium-latest)
    """

    def __init__(self):
        # Perplexity Sonar client (OpenAI-compatible) [web:151]
        self.perplexity_client = PerplexityClient(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )

        # SambaNova client
        self.sambanova_client = SambaNova(
            api_key=os.getenv("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )

        # Mistral client
        self.mistral_client = Mistral(api_key=os.getenv("MISTRALAI_API_KEY"))

        # Model registry
        self.models = {
            "sonar": ModelConfig(
                name="Perplexity Sonar",
                provider="perplexity",
                model_id="sonar-pro"  # or latest Sonar variant you choose [web:151]
            ),
            "llama4_maverick": ModelConfig(
                name="Llama-4-Maverick-17B-128E-Instruct",
                provider="sambanova",
                model_id="Llama-4-Maverick-17B-128E-Instruct",
            ),
            "mistral_medium": ModelConfig(
                name="mistral-medium-latest",
                provider="mistral",
                model_id="mistral-medium-latest",
            ),
        }

    def _format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def call_model(
        self,
        model_key: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Unified call interface. Returns:
          {
            "model_key": ...,
            "model_name": ...,
            "provider": ...,
            "temperature": ...,
            "top_p": ...,
            "timestamp": ...,
            "prompt": ...,
            "response_text": ...,
            "raw_response": ...
          }
        """
        cfg = self.models[model_key]
        messages = self._format_messages(system_prompt, user_prompt)
        ts = datetime.now().isoformat()

        if cfg.provider == "perplexity":
            # OpenAI-compatible chat completion [web:151]
            resp = self.perplexity_client.chat.completions.create(
                model=cfg.model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content

        elif cfg.provider == "sambanova":
            resp = self.sambanova_client.chat.completions.create(
                model=cfg.model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            text = resp.choices[0].message.content

        elif cfg.provider == "mistral":
            completion_args = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            # Mistral conversation API
            resp = self.mistral_client.beta.conversations.start(
                inputs=[{"role": "user", "content": user_prompt}],
                model=cfg.model_id,
                instructions=system_prompt,
                completion_args=completion_args,
                tools=[],
            )
            # you may need to inspect resp structure and adjust:
            text = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)

        else:
            raise ValueError(f"Unknown provider: {cfg.provider}")

        return {
            "model_key": model_key,
            "model_name": cfg.name,
            "provider": cfg.provider,
            "temperature": temperature,
            "top_p": top_p,
            "timestamp": ts,
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "response_text": text,
            "raw_response": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        }