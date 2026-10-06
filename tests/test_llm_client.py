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

from src import config
from src.agent import llm as llm_module
from src.agent.llm import (
    LLMClient,
    LLMError,
    LLMResponseFormatError,
    LLMUsage,
    extract_json,
)


def fake_reply(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
):
    """Build the minimal object shape LLMClient reads from a litellm reply."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
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
        fallback_models=["test/fallback"],
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


def test_the_chain_walks_past_every_exhausted_model(monkeypatch):
    """The free-tier quota is per model, so a later link has its own budget.

    Measured on a real key: with gemini-3.8-flash refusing at 20 requests/day,
    gemini-3.5-flash answered immediately. Walking the chain is what makes more
    than a handful of questions a day possible.
    """
    chained = LLMClient(
        model="test/exhausted-one",
        fallback_models=["test/exhausted-two", "test/has-budget"],
        throttle_seconds=0.0,
        max_retries=0,
    )
    tried: list[str] = []

    def completion(**kwargs):
        tried.append(kwargs["model"])
        if kwargs["model"] != "test/has-budget":
            raise llm_exceptions.RateLimitError(
                "quota exhausted", model=kwargs["model"], llm_provider="test"
            )
        return fake_reply("answered by the third model")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_has_credentials", staticmethod(lambda model: True))

    response = chained.complete("hello")

    assert response.text == "answered by the third model"
    assert tried == ["test/exhausted-one", "test/exhausted-two", "test/has-budget"]
    assert response.used_fallback is True


def test_a_model_repeated_in_the_chain_is_only_tried_once():
    """An exhausted bucket is still exhausted the second time."""
    chained = LLMClient(
        model="test/primary",
        fallback_models=["test/primary", "test/other"],
        throttle_seconds=0.0,
    )

    assert chained.model_chain == ["test/primary", "test/other"]


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


def test_a_truncated_reply_keeps_growing_the_budget_until_it_fits(
    monkeypatch, client
):
    """A doubled budget has to be USED, not logged and discarded.

    The loop used to allow two attempts total, so the second truncation
    computed a larger budget and then fell out of the loop without ever
    sending it -- and reported a cut-off reply as unparsable JSON. Measured on
    the gratuity question, which needs two doublings to fit six quotes.
    """
    budgets: list[int] = []

    def completion(**kwargs):
        budgets.append(kwargs["max_tokens"])
        # Truncated until the budget has doubled twice.
        if len(budgets) < 3:
            return fake_reply('{"answer": "cut off mid-str', finish_reason="length")
        return fake_reply('{"answer": "six months"}')

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    payload, _ = client.complete_json("question", required_keys=("answer",))

    assert payload == {"answer": "six months"}
    assert budgets == [1024, 2048, 4096]


def test_a_reply_still_truncated_at_the_ceiling_is_reported_as_truncated(
    monkeypatch, client
):
    """"No JSON object found" sends the next reader after the prompt.

    The budget is what ran out, and the number they need is the one it ran out
    at, so the error says both.
    """
    def completion(**kwargs):
        return fake_reply('{"answer": "cut off mid-str', finish_reason="length")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    with pytest.raises(LLMResponseFormatError, match="still cut off at 4096 tokens"):
        client.complete_json("question", required_keys=("answer",))


def test_a_truncation_does_not_spend_the_reprompt_allowance(monkeypatch, client):
    """Growing the budget is a different request; scolding it about JSON is not.

    A reply cut off by the budget did nothing wrong, so the reminder must not
    be attached -- on a real run it invited a shorter answer that abandoned
    the question.
    """
    prompts: list[str] = []

    def completion(**kwargs):
        prompts.append(kwargs["messages"][-1]["content"])
        if len(prompts) == 1:
            return fake_reply('{"answer": "cut off', finish_reason="length")
        return fake_reply('{"answer": "six months"}')

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    client.complete_json("question", required_keys=("answer",))

    # The retry is the same prompt at a bigger budget, with no reminder added.
    assert prompts[0] == prompts[1]



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


def test_a_truncated_reply_is_retried_with_a_bigger_budget_not_a_scolding(
    monkeypatch, client
):
    """Observed live: the reminder made the model abandon the question instead.

    A reply cut off by the token budget is well-formed and simply unfinished,
    so the fix is more room, not a sterner prompt.
    """
    seen: list[dict] = []

    def completion(**kwargs):
        seen.append({"max_tokens": kwargs["max_tokens"], "prompt": kwargs["messages"][-1]["content"]})
        if len(seen) == 1:
            # A JSON object cut off mid-string, exactly as Gemini returned one.
            return fake_reply('{\n  "answer": "Under the Labour Act, the employer',
                              finish_reason="length")
        return fake_reply('{"answer": "Under the Labour Act, the employer deposits 8.33%."}')

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    payload, _ = client.complete_json("question", required_keys=("answer",))

    assert payload["answer"].endswith("8.33%.")
    # The budget doubled...
    assert seen[1]["max_tokens"] == seen[0]["max_tokens"] * 2
    # ...and the prompt was NOT changed to complain about JSON formatting.
    assert seen[1]["prompt"] == seen[0]["prompt"]


def test_a_malformed_reply_is_retried_with_the_reminder(monkeypatch, client):
    """The other half of the pair: bad format does get the sterner prompt."""
    seen: list[str] = []

    def completion(**kwargs):
        seen.append(kwargs["messages"][-1]["content"])
        if len(seen) == 1:
            return fake_reply("Sure! Here is my answer: six months.")
        return fake_reply('{"answer": "six months"}')

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    client.complete_json("question", required_keys=("answer",))

    assert "not valid JSON" in seen[1]
    assert "not valid JSON" not in seen[0]


def test_thinking_models_get_their_own_token_headroom(monkeypatch, client):
    """maxOutputTokens is shared with reasoning, so the answer cap is not enough."""
    budgets: list[int] = []

    def completion(**kwargs):
        budgets.append(kwargs["max_tokens"])
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_supports_reasoning", staticmethod(lambda model: True))

    client.complete("analyse this", max_tokens=1024, simple_task=False)
    client.complete("classify this", max_tokens=1024, simple_task=True)

    # Thinking on: the headroom is added. Thinking off: it is not, because the
    # reasoning_effort=disable makes it dead budget.
    assert budgets[0] == 1024 + config.LLM_THINKING_TOKEN_HEADROOM
    assert budgets[1] == 1024


_DAILY_QUOTA_429 = """
{"error": {"code": 429, "message": "You exceeded your current quota.",
  "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
      {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
       "quotaDimensions": {"model": "gemini-3.8-flash"}, "quotaValue": "20"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo",
     "retryDelay": "66545s"}]}}
"""

_PER_MINUTE_429 = """
{"error": {"code": 429, "message": "Too many requests.",
  "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo",
     "retryDelay": "21s"}]}}
"""


def rate_limited(body: str, model: str = "test/primary"):
    return llm_exceptions.RateLimitError(body, model=model, llm_provider="test")


def test_a_daily_quota_goes_straight_to_the_fallback(monkeypatch, client):
    """A retryDelay of 18 hours is not something to back off and wait for.

    Backing off 2/4/9/17s against a daily quota buys thirty wasted seconds and
    still fails, so the chain should move on at the first refusal.
    """
    calls: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: slept.append(s))

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "test/primary":
            raise rate_limited(_DAILY_QUOTA_429)
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    assert client.complete("hello").text == "ok"
    # One attempt on the spent model, then the fallback -- no backoff at all.
    assert calls == ["test/primary", "test/fallback"]
    assert slept == []


def test_a_per_minute_limit_is_still_retried_with_backoff(monkeypatch, client):
    """The distinction has to cut both ways: a short retryDelay IS worth waiting."""
    calls: list[str] = []
    slept: list[float] = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: slept.append(s))

    attempts = {"n": 0}

    def completion(**kwargs):
        calls.append(kwargs["model"])
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise rate_limited(_PER_MINUTE_429)
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    assert client.complete("hello").text == "ok"
    # Retried on the SAME model, after waiting -- not failed over.
    assert calls == ["test/primary", "test/primary"]
    assert len(slept) == 1


def test_a_spent_model_is_not_tried_again_in_the_same_run(monkeypatch, client):
    """The saving that matters: one answer is several calls, and an eval is hundreds."""
    calls: list[str] = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)

    def completion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "test/primary":
            raise rate_limited(_DAILY_QUOTA_429)
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    client.complete("first")
    client.complete("second")
    client.complete("third")

    # The spent model is attempted once, ever -- not once per prompt.
    assert calls.count("test/primary") == 1
    assert calls.count("test/fallback") == 3


def test_an_entirely_spent_chain_says_so_instead_of_blaming_the_last_error(
    monkeypatch, client
):
    """Nothing was attempted, so "all models failed" would describe nothing."""
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)

    def completion(**kwargs):
        raise rate_limited(_DAILY_QUOTA_429, model=kwargs["model"])

    monkeypatch.setattr(llm_module.litellm, "completion", completion)

    # Both the run that exhausts the chain and every call after it: the
    # first is when a reader most needs to be told what happened.
    with pytest.raises(LLMError, match="every model in the chain") as caught:
        client.complete("first")

    with pytest.raises(LLMError, match="every model in the chain") as caught:
        client.complete("second")

    # Names the spent models and what to do, rather than a stack of 429 JSON.
    assert "test/primary" in str(caught.value)
    assert "test/fallback" in str(caught.value)
    assert "POLICYLENS_FALLBACK_MODELS" in str(caught.value)



def test_a_rejected_reasoning_effort_is_dropped_and_remembered(monkeypatch, client):
    """Groq refuses the "disable" value, so sending it twice buys two wasted calls.

    The router and the citation judge are both simple tasks, so without the
    memo every one of them would pay a rejected call and a retry on the exact
    model the chain falls back to when the quota is tight.
    """
    seen: list[bool] = []

    def completion(**kwargs):
        has_effort = "reasoning_effort" in kwargs
        seen.append(has_effort)
        if has_effort:
            raise llm_exceptions.BadRequestError(
                "'reasoning_effort': value is not one of the allowed values",
                model=kwargs["model"],
                llm_provider="test",
            )
        return fake_reply("DOCUMENT_QA")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_supports_reasoning", staticmethod(lambda model: True))

    assert client.complete("classify this", simple_task=True).text == "DOCUMENT_QA"
    assert client.complete("classify that", simple_task=True).text == "DOCUMENT_QA"

    # Rejected once, then never offered to this model again.
    assert seen == [True, False, False]


def test_dropping_reasoning_effort_restores_the_thinking_budget(monkeypatch, client):
    """Thinking comes back on when the parameter goes, so the budget must too.

    Otherwise the retry "succeeds" and returns an empty string, because the
    reasoning spent a budget that was sized for a model that was not thinking.
    """
    budgets: list[int] = []

    def completion(**kwargs):
        budgets.append(kwargs["max_tokens"])
        if "reasoning_effort" in kwargs:
            raise llm_exceptions.BadRequestError(
                "'reasoning_effort': value is not one of the allowed values",
                model=kwargs["model"],
                llm_provider="test",
            )
        return fake_reply("ok")

    monkeypatch.setattr(llm_module.litellm, "completion", completion)
    monkeypatch.setattr(LLMClient, "_supports_reasoning", staticmethod(lambda model: True))

    client.complete("classify this", max_tokens=1024, simple_task=True)
    # The rejected attempt was sized for a model with thinking off; the retry
    # has to cover the reasoning the provider would not let us switch off.
    assert budgets == [1024, 1024 + config.LLM_THINKING_TOKEN_HEADROOM]

    # And the next simple task skips the doomed attempt, keeping the headroom.
    client.complete("classify that", max_tokens=1024, simple_task=True)
    assert budgets[2:] == [1024 + config.LLM_THINKING_TOKEN_HEADROOM]



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
    client = LLMClient(model="test/primary", fallback_models=[], throttle_seconds=0.2)

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
    first = LLMClient(model="test/primary", fallback_models=[], throttle_seconds=0.2)
    second = LLMClient(model="test/primary", fallback_models=[], throttle_seconds=0.2)

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
