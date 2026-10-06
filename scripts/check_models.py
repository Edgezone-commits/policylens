"""Check that the configured models can actually be called with your keys.

    python scripts/check_models.py
    python scripts/check_models.py --list     # also list every Gemini model

Run this first when anything LLM-shaped fails. It separates the three
environment problems that otherwise look identical from inside the agent:

  * the key is missing        -- nothing was set in .env
  * the key is rejected       -- set, but wrong or revoked
  * the model is gone         -- key fine, model name retired

The third is not hypothetical and is why this script exists: the free tier's
model names are not stable. gemini-2.5-flash was this project's default until
Google began refusing it for newly created keys with a 404, which from inside
the agent reads as a generic provider error on every single call.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import litellm  # noqa: E402

from src import config  # noqa: E402

logger = logging.getLogger("check_models")

litellm.suppress_debug_info = True


def list_gemini_models(api_key: str) -> list[str]:
    """Names of Gemini models this key may call for text generation.

    Queried over plain HTTP rather than through LiteLLM because LiteLLM has no
    model-listing call -- and a hardcoded list is exactly what goes stale.
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}&pageSize=200"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        logger.error("Gemini rejected the model listing: HTTP %s", error.code)
        return []
    except OSError as error:
        logger.error("Could not reach the Gemini API: %s", error)
        return []

    return sorted(
        model["name"].removeprefix("models/")
        for model in payload.get("models", [])
        if "generateContent" in model.get("supportedGenerationMethods", [])
    )


def check_model(model: str) -> tuple[bool, str]:
    """Try the cheapest possible real call. Returns (ok, detail).

    A real call, not a key-format check: only the provider can tell us whether
    this key may use this model today, which is the question that matters.
    """
    report = litellm.validate_environment(model=model)
    if not report.get("keys_in_environment", True):
        missing = ", ".join(report.get("missing_keys") or []) or "the API key"
        return False, f"{missing} is not set in .env"

    extra = {}
    # Without this, a thinking model spends the whole token budget reasoning
    # about a two-word prompt and returns an empty string -- which looks like a
    # failure and is not one.
    if litellm.supports_reasoning(model=model):
        extra["reasoning_effort"] = config.LLM_REASONING_FOR_SIMPLE_TASKS

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=32,
            **extra,
        )
    except Exception as error:  # noqa: BLE001 -- the point is to report anything
        detail = str(error).replace("\n", " ")
        return False, f"{type(error).__name__}: {detail[:300]}"

    reply = (response.choices[0].message.content or "").strip()
    return True, f"replied {reply[:40]!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="List every Gemini model the GOOGLE_API_KEY can call",
    )
    args = parser.parse_args()
    config.configure_logging()

    print()
    print("=" * 72)
    print("CONFIGURED MODELS")
    print("=" * 72)

    failures = 0
    for label, model in (
        ("primary ", config.LLM_MODEL),
        ("fallback", config.LLM_FALLBACK_MODEL),
    ):
        if not model:
            print(f"{label}  (none configured)")
            continue
        ok, detail = check_model(model)
        print(f"{label}  {'OK  ' if ok else 'FAIL'}  {model}")
        print(f"          {detail}")
        if not ok:
            # The fallback is optional by design, so a missing Groq key is
            # worth reporting but is not a reason to fail the check.
            failures += 1 if label.strip() == "primary" else 0

    if args.list_models:
        key = os.getenv("GOOGLE_API_KEY", "")
        print()
        print("=" * 72)
        print("GEMINI MODELS AVAILABLE TO THIS KEY")
        print("=" * 72)
        if not key:
            print("GOOGLE_API_KEY is not set.")
        else:
            names = list_gemini_models(key)
            for name in names:
                print(f"  {name}")
            print(f"\n{len(names)} model(s). Use one as "
                  f"POLICYLENS_MODEL=gemini/<name> in .env")

    print()
    if failures:
        print("The primary model is not usable. Either fix the key in .env, or "
              "set POLICYLENS_MODEL to a model this key can call "
              "(python scripts/check_models.py --list).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
