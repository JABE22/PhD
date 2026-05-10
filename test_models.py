"""
Quick test script to verify connections to:

- Perplexity (sonar-pro, official SDK)
- SambaNova (Llama-4-Maverick-17B-128E-Instruct)
- Mistral (mistral-medium-latest)

Prerequisites:
- pip install -r requirements.txt
- .env file with:
    PERPLEXITY_API_KEY=...
    SAMBANOVA_API_KEY=...
    MISTRALAI_API_KEY=...

Usage:
    python test_models.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def test_perplexity():
    print("\n[Perplexity] Testing sonar-pro (official SDK)...")

    try:
        from perplexity import Perplexity
    except ImportError:
        print("  ✗ perplexity package not installed. Run: pip install perplexity")
        return

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("  ✗ PERPLEXITY_API_KEY missing in .env")
        return

    try:
        # Official SDK: key via env or explicit, both work [web:137][web:152]
        client = Perplexity(api_key=api_key)

        completion = client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "user",
                    "content": "Say: Connection successful (Perplexity).",
                }
            ],
            max_tokens=20,
            temperature=0.0,
        )

        text = completion.choices[0].message.content.strip()
        print("  ✓ Success")
        print("  Response:", text)

        usage = getattr(completion, "usage", None)
        if usage:
            print(
                f"  Tokens: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}"
            )
    except Exception as e:
        print("  ✗ Error:", e)


def test_sambanova():
    print("\n[SambaNova] Testing Llama-4-Maverick-17B-128E-Instruct...")

    try:
        from sambanova import SambaNova
    except ImportError:
        print("  ✗ sambanova package not installed. Run: pip install sambanova")
        return

    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        print("  ✗ SAMBANOVA_API_KEY missing in .env")
        return

    try:
        client = SambaNova(
            api_key=api_key,
            base_url="https://api.sambanova.ai/v1",
        )

        resp = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say: Connection successful (SambaNova)."},
            ],
            max_tokens=20,
            temperature=0.0,
        )

        text = resp.choices[0].message.content.strip()
        print("  ✓ Success")
        print("  Response:", text)

        usage = getattr(resp, "usage", None)
        if usage and hasattr(usage, "total_tokens"):
            print(f"  Tokens: total={usage.total_tokens}")
    except Exception as e:
        print("  ✗ Error:", e)


def test_mistral():
    print("\n[Mistral] Testing mistral-medium-latest...")

    try:
        from mistralai import Mistral
    except ImportError:
        print("  ✗ mistralai package not installed. Run: pip install mistralai")
        return

    api_key = os.getenv("MISTRALAI_API_KEY")
    if not api_key:
        print("  ✗ MISTRALAI_API_KEY missing in .env")
        return

    try:
        client = Mistral(api_key=api_key)

        resp = client.beta.conversations.start(
            inputs=[{"role": "user", "content": "Say: Connection successful (Mistral)."}],
            model="mistral-medium-latest",
            completion_args={"max_tokens": 20, "temperature": 0.0},
        )

        text = None
        if hasattr(resp, "outputs") and resp.outputs:
            entry = resp.outputs[0]
            content = getattr(entry, "content", None)
            text = content if isinstance(content, str) else str(content)
        else:
            text = str(resp)

        print("  ✓ Success")
        print("  Response:", text.strip())
    except Exception as e:
        print("  ✗ Error:", e)


if __name__ == "__main__":
    print("=== Testing model connections ===")
    test_perplexity()
    test_sambanova()
    test_mistral()
    print("\nDone.")
