"""The project's only LLM call site.

Everything the agent asks a model goes through LLMClient. WHY funnel it:

  * Free tiers rate-limit per MINUTE as well as per day, so calls must be
    spaced. A throttle is only effective if it cannot be bypassed, which means
    there has to be exactly one door.
  * A 429 mid-demo should not be a crash. Retry-with-backoff and a fallback
    provider belong here, once, rather than in each of the four nodes.
  * Nodes want a dict, not prose. The JSON-repair logic that makes a fenced
    reply usable is fiddly and identical everywhere, so it lives here too.

The model name itself is never written in this file -- it comes from
src/config.py, so swapping provider stays a one-line change.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import exceptions as llm_exceptions

from src import config

logger = logging.getLogger(__name__)

# Drop LiteLLM's provider banners from stdout. configure_logging() already
# quiets its logger; this flag covers what it prints directly.
litellm.suppress_debug_info = True

# Errors worth retrying on the SAME model: transient, and usually cured by
# waiting. A 429 on a free tier is the common case.
_RETRYABLE = (
    llm_exceptions.RateLimitError,
    llm_exceptions.Timeout,
    llm_exceptions.APIConnectionError,
    llm_exceptions.InternalServerError,
    llm_exceptions.ServiceUnavailableError,
)

# Errors retrying cannot fix -- a bad key or a model name that does not exist
# will fail identically forever. These skip straight to the fallback provider
# instead of burning the backoff budget.
_SWITCH_PROVIDER = (
    llm_exceptions.AuthenticationError,
    llm_exceptions.PermissionDeniedError,
    llm_exceptions.NotFoundError,
)


# A 429 means two different things, and they want opposite responses.
#
# A per-MINUTE limit clears in seconds, so backing off is exactly right.
# A per-DAY free-tier quota reports, verbatim:
#     "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
#     "retryDelay": "66545s"
# -- eighteen hours. Backing off 2/4/9/17s against that buys nothing but
# thirty wasted seconds, and since one answer costs several calls, every
# one of them pays it again: a measured 88s for a question that takes 10s
# once the spent model is skipped. So a daily refusal is treated as a dead
# model for the rest of the run, which is what the fallback chain is for.
#
# Keyed on retryDelay rather than the quotaId, because the delay is the
# fact that matters ("could waiting plausibly help?") and other providers
# report it too; the PerDay marker is the fallback when it is absent.
_UNWAITABLE_RETRY_DELAY_SECONDS = 120.0


def _is_spent_for_the_day(error: Exception) -> bool:
    """Is this 429 a daily quota, rather than a per-minute burst limit?"""
    text = str(error)
    delay = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', text)
    if delay:
        return float(delay.group(1)) > _UNWAITABLE_RETRY_DELAY_SECONDS
    return "PerDay" in text


class LLMError(RuntimeError):
    """Raised when every model and retry has been exhausted."""


class LLMResponseFormatError(LLMError):
    """Raised when a reply that had to be JSON could not be parsed as JSON."""


@dataclass
class LLMResponse:
    """One completed call, with the provenance the reasoning path reports."""

    text: str
    model: str
    attempts: int = 1
    used_fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        """Did the model run out of output budget mid-reply?

        Worth distinguishing from any other bad reply: a truncated JSON object
        is not a formatting mistake, so reprompting with "reply with JSON only"
        asks the model to fix something it did not do wrong -- and, observed on
        a real run, invites it to return a shorter answer that abandons the
        question instead. The fix is a bigger budget, not a sterner prompt.
        """
        return self.finish_reason in ("length", "max_tokens", "MAX_TOKENS")


def call_record(response: LLMResponse, label: str) -> dict[str, Any]:
    """Describe one call for the agent state's running list.

    Plain dicts rather than objects because this list travels in the graph
    state, gets returned by the API as JSON, and is written to the query log --
    all three want something already serialisable.
    """
    return {
        "label": label,
        "model": response.model,
        "attempts": response.attempts,
        "used_fallback": response.used_fallback,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_seconds": round(response.latency_seconds, 2),
    }


@dataclass
class LLMUsage:
    """Totals for one agent run, surfaced in the API response.

    WHY count at all: on a free tier the interesting cost is requests, not
    dollars, and "this answer took 5 calls" is the number that explains both
    the latency a user feels and how close a batch evaluation is to its daily
    ceiling.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_calls: int = 0
    retries: int = 0
    per_call: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> LLMUsage:
        """Aggregate the per-call records accumulated in the agent state."""
        return cls(
            calls=len(records),
            prompt_tokens=sum(int(record.get("prompt_tokens", 0)) for record in records),
            completion_tokens=sum(
                int(record.get("completion_tokens", 0)) for record in records
            ),
            fallback_calls=sum(1 for record in records if record.get("used_fallback")),
            # attempts counts the first try too, so retries is attempts - 1.
            retries=sum(int(record.get("attempts", 1)) - 1 for record in records),
            per_call=list(records),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "fallback_calls": self.fallback_calls,
            "retries": self.retries,
            "per_call": self.per_call,
        }


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply.

    Models asked for JSON return it four ways in practice: bare, inside a
    json-tagged markdown fence, inside a bare fence, and with a line of
    commentary in front ("Here is the JSON you asked for:"). Even with
    response_format set, the fallback provider may ignore it. Rather than fail
    a whole answer on punctuation, we find the outermost braces and parse that.

    Raises LLMResponseFormatError if there is no parsable object.
    """
    if not text or not text.strip():
        raise LLMResponseFormatError("model returned an empty reply")

    candidate = text.strip()

    # Strip a markdown fence if the whole reply is wrapped in one.
    fence = re.match(r"^`{3,}(?:json)?\s*(.*?)\s*`{3,}$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span. find/rfind rather than a regex
        # because nested objects make a non-greedy regex stop at the wrong
        # brace.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseFormatError(
                f"no JSON object found in reply: {text[:200]!r}"
            ) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as error:
            raise LLMResponseFormatError(
                f"malformed JSON in reply ({error.msg}): {text[:200]!r}"
            ) from error

    if not isinstance(parsed, dict):
        raise LLMResponseFormatError(
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


class LLMClient:
    """Throttled, retrying, provider-swapping wrapper over litellm.completion."""

    # Class-level, not instance-level: the rate limit belongs to the API KEY,
    # so two LLMClient objects in one process (the agent and the evaluation
    # harness, say) must share one clock, or they would each send at the limit
    # and together exceed it.
    _throttle_lock = threading.Lock()
    _last_call_at: float = 0.0

    def __init__(
        self,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        throttle_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.model = model or config.LLM_MODEL
        self.fallback_models = (
            list(fallback_models)
            if fallback_models is not None
            else list(config.LLM_FALLBACK_MODELS)
        )
        # The chain in call order, with the primary first and duplicates
        # dropped. Deduplicated because a repeated name buys nothing -- the
        # daily quota is per model, so trying an exhausted one twice just
        # spends two more seconds proving it is still exhausted.
        self.model_chain: list[str] = list(
            dict.fromkeys([self.model, *self.fallback_models])
        )
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.throttle_seconds = (
            config.LLM_MIN_SECONDS_BETWEEN_CALLS
            if throttle_seconds is None
            else throttle_seconds
        )
        self.max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
        # Models whose json_object response_format turned out to be rejected.
        # Remembered so one BadRequest does not cost a wasted call on every
        # subsequent prompt.
        self._no_json_mode: set[str] = set()
        # The same, for reasoning_effort. Separate sets because the two are
        # rejected independently: Groq takes response_format but refuses
        # "disable" as a reasoning_effort, and one rejection says nothing
        # about the other.
        self._no_reasoning_effort: set[str] = set()
        # Models whose per-day quota is gone. Process-wide and never
        # cleared: the reset is ~18 hours away, so nothing in one run will
        # outlive it.
        self._spent_today: set[str] = set()

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        simple_task: bool = False,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send one prompt and return the reply.

        `simple_task=True` disables Gemini's "thinking" budget. The router and
        the citation judge are classification, not reasoning: thinking roughly
        doubles their latency and token cost for no measured gain, while the
        analyzer -- which has to read five chunks and compose a cited answer --
        keeps it.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        models = self.model_chain
        attempts = 0
        last_error: Exception | None = None

        for index, model in enumerate(models):
            is_fallback = index > 0
            if model in self._spent_today:
                logger.info(
                    "Skipping %s: its daily quota ran out earlier in this run",
                    model,
                )
                continue
            if is_fallback:
                if not self._has_credentials(model):
                    logger.warning("Skipping fallback %s: its API key is not set", model)
                    continue
                logger.warning(
                    "Falling back to %s (%d of %d in the chain) after %s failed",
                    model,
                    index + 1,
                    len(models),
                    models[index - 1],
                )

            for retry in range(self.max_retries + 1):
                attempts += 1
                try:
                    return self._call(
                        model=model,
                        messages=messages,
                        simple_task=simple_task,
                        json_mode=json_mode,
                        max_tokens=max_tokens or self.max_tokens,
                        attempts=attempts,
                        used_fallback=is_fallback,
                    )
                except _SWITCH_PROVIDER as error:
                    # Nothing to gain by waiting; try the other provider.
                    logger.warning("%s rejected the call: %s", model, error)
                    last_error = error
                    break
                except _RETRYABLE as error:
                    last_error = error
                    if isinstance(
                        error, llm_exceptions.RateLimitError
                    ) and _is_spent_for_the_day(error):
                        self._spent_today.add(model)
                        logger.warning(
                            "%s has used its daily free-tier quota; moving on "
                            "and not trying it again this run",
                            model,
                        )
                        break
                    if retry == self.max_retries:
                        logger.warning(
                            "%s still failing after %d retries: %s",
                            model,
                            self.max_retries,
                            type(error).__name__,
                        )
                        break
                    delay = self._backoff_seconds(retry)
                    logger.warning(
                        "%s on %s; retrying in %.1fs (%d/%d)",
                        type(error).__name__,
                        model,
                        delay,
                        retry + 1,
                        self.max_retries,
                    )
                    time.sleep(delay)
                except llm_exceptions.BadRequestError as error:
                    # A malformed prompt is our bug, not the provider's, and it
                    # fails identically on the fallback. Surface it.
                    raise LLMError(f"{model} rejected the request: {error}") from error

        if models and all(model in self._spent_today for model in models):
            # The whole chain is out of quota. Said plainly, because the fix
            # is to wait or to add a model -- and because the alternative is
            # a page of provider JSON on the run where the quota ran out,
            # which is exactly when this is hardest to read.
            raise LLMError(
                "every model in the chain has used its daily free-tier quota "
                f"({', '.join(sorted(self._spent_today))}). The quota resets "
                "about 18 hours after it ran out; until then, add a model to "
                "POLICYLENS_FALLBACK_MODELS or use a different key."
            ) from last_error

        raise LLMError(
            f"all models failed after {attempts} attempt(s); "
            f"last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        simple_task: bool = False,
        required_keys: tuple[str, ...] = (),
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Send a prompt that must be answered with a JSON object.

        One retry is allowed if the reply will not parse, or parses but lacks a
        required key. WHY retry rather than fail: a malformed reply is usually
        a one-off (a stray preamble, a truncated brace), and the alternative is
        losing an otherwise good answer to a formatting slip. WHY only one: if
        the second reply is also malformed then the prompt is wrong, and a
        third call just spends quota proving it.

        The retry is chosen to match the failure, which matters more than it
        sounds. A reply cut off by the token budget is not a formatting
        mistake, and scolding the model about JSON produced -- on a real run --
        a well-formed reply that had given up on the question. Truncation gets
        a bigger budget; everything else gets the reminder.
        """
        reminder = (
            "\n\nIMPORTANT: your previous reply was not valid JSON. Reply with "
            "the JSON object ONLY -- no prose, no markdown fence."
        )

        budget = max_tokens or self.max_tokens
        remind = False
        last_error: LLMResponseFormatError | None = None

        # Two separate allowances, because the two failures are not alike.
        #
        # A malformed reply gets ONE reprompt: if the second is also malformed
        # the prompt is wrong, and a third call just spends quota proving it.
        #
        # A truncation is not that. It is the same prompt against too small a
        # budget, so doubling it is a DIFFERENT request rather than the same one
        # asked again, and it deserves its own allowance. It previously shared
        # the one above, which meant the last iteration logged "retrying with
        # N tokens", computed the bigger budget, and then fell out of the loop
        # without ever using it -- reporting a cut-off reply as unparsable
        # JSON. Measured on "how much gratuity is a worker entitled to",
        # which the Act spreads across Rule 23, Section 52 and Section 53:
        # six verbatim quotes, one doubling off the analyzer's own cap, and
        # an answer with 6/6 citations verified. From the DEFAULT cap the
        # same question needs three, which is why the analyzer carries its
        # own (see LLM_MAX_TOKENS_FOR_CITED_ANSWER) rather than relying on
        # this loop to climb there one wasted call at a time.
        reprompts_left = 1
        growths_left = 2
        response: LLMResponse | None = None

        while True:
            response = self.complete(
                prompt + reminder if remind else prompt,
                system=system,
                simple_task=simple_task,
                json_mode=True,
                max_tokens=budget,
            )
            try:
                payload = extract_json(response.text)
            except LLMResponseFormatError as error:
                last_error = error
                if response.truncated and growths_left:
                    growths_left -= 1
                    budget *= 2
                    remind = False
                    logger.warning(
                        "Reply was cut off by the token budget; retrying with "
                        "%d tokens (%d growth(s) left)",
                        budget,
                        growths_left,
                    )
                    continue
                if not response.truncated and reprompts_left:
                    reprompts_left -= 1
                    remind = True
                    logger.warning("Unparsable JSON reply: %s", error)
                    continue
                break

            missing = [key for key in required_keys if key not in payload]
            if missing:
                last_error = LLMResponseFormatError(
                    f"reply is missing required key(s): {', '.join(missing)}"
                )
                if reprompts_left:
                    reprompts_left -= 1
                    remind = True
                    logger.warning("%s", last_error)
                    continue
                break

            return payload, response

        if response is not None and response.truncated:
            # Name the real cause. "No JSON object found" sends the next reader
            # after the prompt, when the budget is what ran out -- and the
            # number here is the one they would need.
            last_error = LLMResponseFormatError(
                f"the reply was still cut off at {budget} tokens after "
                f"exhausting the budget growths; raise LLM_MAX_TOKENS or "
                f"the caller's max_tokens"
            )

        raise last_error or LLMResponseFormatError("could not obtain JSON from model")

    # -- internals ----------------------------------------------------------

    def _call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        simple_task: bool,
        json_mode: bool,
        max_tokens: int,
        attempts: int,
        used_fallback: bool,
    ) -> LLMResponse:
        thinks = self._supports_reasoning(model)
        # Can thinking actually be turned off for this model? Not if the
        # provider has already refused reasoning_effort -- Groq rejects the
        # "disable" value outright -- in which case a "simple" task still
        # pays for reasoning and still needs the budget to cover it.
        can_disable_thinking = model not in self._no_reasoning_effort
        thinking_on = thinks and not (simple_task and can_disable_thinking)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            # A thinking model spends this budget on reasoning before it writes
            # a word of the reply, so the headroom is added only when thinking
            # is actually left on (see LLM_THINKING_TOKEN_HEADROOM). Getting
            # this wrong does not raise: the reasoning eats the whole budget
            # and the reply comes back empty.
            "max_tokens": max_tokens
            + (config.LLM_THINKING_TOKEN_HEADROOM if thinking_on else 0),
        }
        if json_mode and model not in self._no_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if simple_task and thinks and can_disable_thinking:
            kwargs["reasoning_effort"] = config.LLM_REASONING_FOR_SIMPLE_TASKS

        self._wait_for_turn()
        started = time.monotonic()
        try:
            raw = litellm.completion(**kwargs)
        except llm_exceptions.BadRequestError:
            # Some providers reject response_format or reasoning_effort rather
            # than ignoring them. Retry once without the optional parameters
            # before concluding the request itself is bad -- and remember it, so
            # the next prompt does not pay the same wasted call.
            optional = [
                key for key in ("response_format", "reasoning_effort") if key in kwargs
            ]
            if not optional:
                raise
            logger.warning(
                "%s rejected %s; retrying without it", model, ", ".join(optional)
            )
            if "response_format" in kwargs:
                self._no_json_mode.add(model)
            if "reasoning_effort" in kwargs:
                self._no_reasoning_effort.add(model)
                # Dropping the parameter turns thinking back ON, so the
                # budget has to grow with it. Without this the retry
                # succeeds and returns an empty reply, which is the harder
                # failure to read: the call looks fine and says nothing.
                if not thinking_on:
                    kwargs["max_tokens"] = (
                        max_tokens + config.LLM_THINKING_TOKEN_HEADROOM
                    )
            for key in optional:
                kwargs.pop(key)
            self._wait_for_turn()
            raw = litellm.completion(**kwargs)

        latency = time.monotonic() - started
        text = (raw.choices[0].message.content or "").strip()
        usage = getattr(raw, "usage", None)
        finish = str(getattr(raw.choices[0], "finish_reason", "stop") or "stop")

        if not text:
            # An empty reply is not an exception in LiteLLM, but it is useless
            # to every caller, so turn it into one here.
            raise LLMError(f"{model} returned an empty reply (finish_reason={finish})")

        response = LLMResponse(
            text=text,
            model=model,
            attempts=attempts,
            used_fallback=used_fallback,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=latency,
            finish_reason=finish,
        )
        logger.debug(
            "LLM %s: %d prompt + %d completion tokens in %.1fs",
            model,
            response.prompt_tokens,
            response.completion_tokens,
            latency,
        )
        return response

    def _wait_for_turn(self) -> None:
        """Block until `throttle_seconds` have passed since the last call.

        The lock is held across the sleep on purpose: the point is to serialise
        outbound calls, so a second thread must wait rather than observe a
        stale timestamp and fire immediately.
        """
        if self.throttle_seconds <= 0:
            return
        with LLMClient._throttle_lock:
            deadline = LLMClient._last_call_at + self.throttle_seconds
            remaining = deadline - time.monotonic()
            if remaining > 0:
                logger.debug("Throttling %.1fs to respect per-minute limits", remaining)
                # Slept in a loop, not once: time.sleep() can return early as
                # measured by time.monotonic(), because on Windows the two read
                # different clocks and disagree by up to ~15ms. One short sleep
                # would put two calls closer together than the per-minute limit
                # allows, which is the one thing this function exists to stop.
                while remaining > 0:
                    time.sleep(remaining)
                    remaining = deadline - time.monotonic()
            LLMClient._last_call_at = time.monotonic()

    def _backoff_seconds(self, retry: int) -> float:
        """Exponential backoff with jitter: roughly 2s, 4s, 8s, 16s.

        Jitter matters even for a single-user tool, because the evaluation
        harness retries many questions that hit the same per-minute window;
        without it they would all wake up together and collide again.
        """
        return min(2.0 ** (retry + 1), 30.0) * (1.0 + random.random() * 0.25)

    @staticmethod
    def _supports_reasoning(model: str) -> bool:
        """Does this model accept a reasoning/thinking budget parameter?"""
        try:
            return bool(litellm.supports_reasoning(model=model))
        except Exception:  # noqa: BLE001 -- unknown model, treat as "no"
            return False

    @staticmethod
    def _has_credentials(model: str) -> bool:
        """Is the API key for this model's provider present in the environment?

        Checked before falling back so the log says "its API key is not set"
        rather than showing an authentication traceback from a provider the
        user never intended to use.
        """
        try:
            report = litellm.validate_environment(model=model)
        except Exception:  # noqa: BLE001 -- unknown model string
            return True  # let the call itself produce the real error
        return bool(report.get("keys_in_environment", True))


_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the process-wide client, so the throttle clock is shared."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
