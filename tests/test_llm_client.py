"""Tests for the LLM client: parsing, retrying, failing over, throttling.

litellm.completion is monkeypatched throughout, so nothing here touches the
network or needs a key. WHY test this layer at all when the model is the
unpredictable part: the behaviour that keeps a free-tier demo alive is not the
model's, it is this file's -- backing off on a 429, failing over when a key is
rejected, and spacing calls so the per-minute limit is never hit. Those paths
only ever run when something has already gone wrong, which is precisely when
nobody is watching them, so they are tested here instead.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from litellm import exceptions as llm_exceptions

from src.agent import llm as llm_module
from src.agent.llm import (
    LLMClient,
    LLMError,
    LLMResponseFormatError,
    LLMUsage,
    extract_json,
)


def fake_reply(content: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build the minimal object shape LLMClient reads from a litellm reply."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@pytest.fixture
def client() -> LLMClient:
    """A client with the waiting switched off, so tests run instantly."""
    return LLMClient(
        model="test/primary",
        fallback_model="test/fallback",
        throttle_seconds=0.0,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"intent": "DOCUMENT_QA"}',
        '```json\n{"intent": "DOCUMENT_QA"}\n```',
        '```\n{"intent": "DOCUMENT_QA"}\n```',
        'Here is the JSON you asked for:\n{"intent": "DOCUMENT_QA"}',
        '  \n {"intent": "DOCUMENT_QA"}  \n',
    ],
)
def test_extract_json_handles_the_wrappings_models_actually_use(reply):
    assert extract_json(reply) == {"intent": "DOCUMENT_QA"}


def test_extract_json_keeps_nested_objects_intact():
    """A non-greedy regex would stop at the inner brace; find/rfind does not."""
    reply = 'Result:\n{"citations": [{"evidence": 1, "quote": "x"}], "ok": true}\ndone'
    assert extract_json(reply) == {
        "citations": [{"evidence": 1, "quote": "x"}],
        "ok": True,
    }


@pytest.mark.parametrize("reply", ["", "   ", "no json here at all", "[1, 2, 3]", "{"])
def test_extract_json_rejects_what_it_cannot_parse(reply):
    with pytest.raises(LLMResponseFormatError):
        extract_json(reply)


# ---------------------------------------------------------------------------
# Retry and failover
# ---------------------------------------------------------------------------


def test_rate_limit_is_retried_on_the_same_model(monkeypatch, client):
    calls: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) < 3:
            raise llm_exceptions.RateLimitError(
                "429", model="test/primary", llm_provider="test"
            )
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(client, "_backoff_seconds", lambda retry: 0.0)

    response = client.complete("hello")

    assert response.text == "ok"
    assert calls == ["test/primary"] * 3  # retried, never failed over
    assert response.attempts == 3
    assert response.used_fallback is False


def test_a_rejected_key_fails_over_immediately(monkeypatch, client):
    """No point waiting: a bad key will be just as bad in two seconds."""
    calls: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "test/primary":
            raise llm_exceptions.AuthenticationError(
                "bad key", model="test/primary", llm_provider="test"
            )
        return fake_reply("from the fallback")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_has_credentials", staticmethod(lambda model: True))

    response = client.complete("hello")

    assert response.text == "from the fallback"
    assert response.used_fallback is True
    assert calls == ["test/primary", "test/fallback"]  # one try each, no backoff


def test_exhausted_quota_falls_back_after_retrying(monkeypatch, client):
    """The free-tier daily limit, which is what the fallback exists for."""
    calls: list[str] = []

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "test/primary":
            raise llm_exceptions.RateLimitError(
                "quota exhausted", model="test/primary", llm_provider="test"
            )
        return fake_reply("from the fallback")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(client, "_backoff_seconds", lambda retry: 0.0)
    monkeypatch.setattr(LLMClient, "_has_credentials", staticmethod(lambda model: True))

    response = client.complete("hello")

    assert response.used_fallback is True
    assert calls.count("test/primary") == 3  # the first try plus two retries
    assert calls[-1] == "test/fallback"


def test_a_fallback_without_a_key_is_skipped_with_a_clear_error(monkeypatch, client):
    def completion(**kwargs):
        raise llm_exceptions.RateLimitError(
            "429", model=kwargs["model"], llm_provider="test"
        )

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(client, "_backoff_seconds", lambda retry: 0.0)
    monkeypatch.setattr(LLMClient, "_has_credentials", staticmethod(lambda model: False))

    with pytest.raises(LLMError, match="RateLimitError"):
        client.complete("hello")


def test_an_empty_reply_is_treated_as_a_failure(monkeypatch, client):
    """LiteLLM reports it as success; it is useless to every caller."""
    monkeypatch.setattr(
        llm_module.litellm, "completion", lambda **kwargs: fake_reply("   ")
    )
    monkeypatch.setattr(LLMClient, "_has_credentials", staticmethod(lambda model: True))

    with pytest.raises(LLMError):
        client.complete("hello")


def test_json_mode_is_dropped_and_remembered_when_a_provider_rejects_it(
    monkeypatch, client
):
    """A provider that rejects response_format should cost one wasted call, not one per prompt."""
    seen: list[bool] = []

    def completion(**kwargs):
        has_format = "response_format" in kwargs
        seen.append(has_format)
        if has_format:
            raise llm_exceptions.BadRequestError(
                "response_format is not supported",
                model=kwargs["model"],
                llm_provider="test",
            )
        return fake_reply('{"ok": true}')

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    first, _ = client.complete_json("hello", required_keys=("ok",))
    second, _ = client.complete_json("hello again", required_keys=("ok",))

    assert first == second == {"ok": True}
    # Rejected once, then never attempted again for this model.
    assert seen == [True, False, False]


def test_complete_json_reprompts_once_then_gives_up(monkeypatch, client):
    replies = ["not json at all", "still not json"]

    def completion(**kwargs):
        return fake_reply(replies.pop(0) if replies else "{}")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    with pytest.raises(LLMResponseFormatError):
        client.complete_json("hello")
    # Exactly two calls: one reprompt, and no third attempt spending quota to
    # prove the prompt is wrong.
    assert replies == []


def test_complete_json_reprompts_when_a_required_key_is_missing(monkeypatch, client):
    replies = ['{"reason": "forgot the verdict"}', '{"supported": true}']

    def completion(**kwargs):
        return fake_reply(replies.pop(0))

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    payload, _ = client.complete_json("hello", required_keys=("supported",))

    assert payload == {"supported": True}


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def test_calls_are_spaced_by_the_throttle_interval(monkeypatch):
    """Free tiers limit requests per minute, so calls must be spread out."""
    monkeypatch.setattr(
        llm_module.litellm, "completion", lambda **kwargs: fake_reply("ok")
    )
    # The clock is shared across clients, so reset it or an earlier test's
    # timestamp decides this one's result.
    LLMClient._last_call_at = 0.0
    client = LLMClient(model="test/primary", fallback_model="", throttle_seconds=0.2)

    started = time.monotonic()
    client.complete("one")
    client.complete("two")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2


def test_the_throttle_clock_is_shared_between_clients(monkeypatch):
    """Two clients, one API key, one rate limit -- so one clock."""
    monkeypatch.setattr(
        llm_module.litellm, "completion", lambda **kwargs: fake_reply("ok")
    )
    LLMClient._last_call_at = 0.0
    first = LLMClient(model="test/primary", fallback_model="", throttle_seconds=0.2)
    second = LLMClient(model="test/primary", fallback_model="", throttle_seconds=0.2)

    started = time.monotonic()
    first.complete("one")
    second.complete("two")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2


def test_backoff_grows_and_stays_bounded(client):
    delays = [client._backoff_seconds(retry) for retry in range(5)]

    assert delays[0] < delays[1] < delays[2]  # exponential
    assert all(delay <= 30.0 * 1.25 for delay in delays)  # capped, plus jitter
    assert delays[0] >= 2.0  # the first wait is long enough to clear a 429


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


def test_usage_totals_are_aggregated_from_the_per_call_records():
    usage = LLMUsage.from_records(
        [
            {
                "label": "router",
                "attempts": 1,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "used_fallback": False,
            },
            {
                "label": "analyzer",
                "attempts": 3,
                "prompt_tokens": 900,
                "completion_tokens": 150,
                "used_fallback": True,
            },
        ]
    )

    assert usage.calls == 2
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 170
    assert usage.fallback_calls == 1
    assert usage.retries == 2  # attempts counts the first try too
