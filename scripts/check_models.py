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
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import litellm  # noqa: E402
from litellm import exceptions as llm_exceptions  # noqa: E402

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


# Enough for "ok" and nothing more: the probe is charged against the same
# daily quota as a real question, so it should cost as little as possible.
_PROBE_TOKENS = 32


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

    def call(**overrides: object):
        return litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            **overrides,
        )

    stripped = ""
    try:
        try:
            response = call(max_tokens=_PROBE_TOKENS, **extra)
        except llm_exceptions.BadRequestError:
            # The same retry LLMClient._call_once performs: a provider may
            # REJECT reasoning_effort rather than ignore it, and the spelling
            # is provider-specific ("disable" is Gemini's; Groq accepts
            # "none" but not "disable"). Without mirroring that retry this
            # script reports FAIL for a model the agent uses perfectly well
            # -- and its one job is to say which models are reachable, so a
            # false FAIL would argue for deleting a working fallback.
            if not extra:
                raise
            stripped = (
                " (after dropping reasoning_effort, which this provider"
                " rejects rather than ignores -- the client retries the"
                " same way)"
            )
            # With reasoning_effort gone, thinking is back ON, and a probe
            # budget sized for a two-word answer would be spent entirely on
            # reasoning -- returning the empty string this script exists to
            # tell apart from a real failure. So pay for the headroom here.
            response = call(
                max_tokens=_PROBE_TOKENS + config.LLM_THINKING_TOKEN_HEADROOM
            )
    except Exception as error:  # noqa: BLE001 -- the point is to report anything
        detail = str(error).replace("\n", " ")
        # A quota refusal is the common case and deserves to be legible rather
        # than buried in 300 characters of provider JSON. The daily allowance
        # is the number that decides how many questions today can hold.
        quota = re.search(r'"quotaValue":\s*"(\d+)"', detail)
        if quota:
            return False, (
                f"daily free-tier quota exhausted ({quota.group(1)} requests/day "
                "for this model); the next model in the chain has its own bucket"
            )
        return False, f"{type(error).__name__}: {detail[:300]}"

    reply = (response.choices[0].message.content or "").strip()
    return True, f"replied {reply[:40]!r}{stripped}"


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

    chain = [("primary ", config.LLM_MODEL)] + [
        (f"fallback{position}", model)
        for position, model in enumerate(config.LLM_FALLBACK_MODELS, start=1)
    ]

    usable = 0
    for label, model in chain:
        if not model:
            continue
        ok, detail = check_model(model)
        print(f"{label}  {'OK  ' if ok else 'FAIL'}  {model}")
        print(f"          {detail}")
        usable += 1 if ok else 0

    print()
    # The chain matters more than any single entry: the free-tier daily limit
    # is charged per model, so each usable link is another day's worth of
    # questions. One usable model is a handful of questions a day.
    print(f"{usable} of {len(chain)} model(s) in the chain are usable right now.")

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

    if not usable:
        print("Nothing in the chain can be called. Either fix the key in .env, "
              "or set POLICYLENS_MODEL / POLICYLENS_FALLBACK_MODELS to models "
              "this key can reach (python scripts/check_models.py --list).")

    # Non-zero only when NO model is usable. An exhausted primary with a
    # working fallback is the chain doing its job, not a failed check.
    return 1 if not usable else 0


if __name__ == "__main__":
    raise SystemExit(main())
