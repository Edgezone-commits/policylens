"""Tests for the UI's contract with the API.

Streamlit rendering is not unit tested here -- that needs a browser to mean
anything, and `streamlit run` plus AppTest covers it. What IS tested is the
one thing that breaks silently: the UI's table of statuses drifting away from
the statuses the API can actually return.

The failure that motivates this: add a status to AnswerResponse, forget the
UI, and every answer with the new status renders through the ERROR branch --
red box, "the system could not run" -- while the API returned 200 and a
perfectly good answer. Nothing would raise, no test would fail, and the UI
would be confidently telling users the opposite of what happened.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.schemas import AnswerResponse

frontend = pytest.importorskip(
    "frontend.app", reason="the UI's optional dependencies are not installed"
)


def api_statuses() -> set[str]:
    """Every status the API's own response model permits."""
    annotation = AnswerResponse.model_fields["status"].annotation
    return set(get_args(annotation))


def test_the_ui_presents_every_status_the_api_can_return():
    """Otherwise a new status renders as a red "could not run" box."""
    missing = api_statuses() - set(frontend.STATUS_PRESENTATION)

    assert not missing, (
        f"These statuses have no presentation in the UI and would fall back "
        f"to the ERROR branch: {sorted(missing)}"
    )


def test_the_ui_does_not_present_statuses_the_api_cannot_return():
    """A stale entry is dead code that looks like coverage."""
    extra = set(frontend.STATUS_PRESENTATION) - api_statuses()

    assert not extra, f"Statuses the API never returns: {sorted(extra)}"


@pytest.mark.parametrize("status", sorted(api_statuses()))
def test_each_presentation_is_complete(status):
    presentation = frontend.STATUS_PRESENTATION[status]

    for key in ("icon", "label", "blurb", "kind"):
        assert presentation.get(key), f"{status} is missing {key}"


@pytest.mark.parametrize("status", ["INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"])
def test_an_abstention_is_never_presented_as_an_error(status):
    """The project's central UX claim, asserted rather than left to a comment.

    An abstention is the system working. Rendering it through st.error would
    teach users that the one behaviour worth trusting is a malfunction.
    """
    assert frontend.STATUS_PRESENTATION[status]["kind"] == "info"


def test_a_pipeline_failure_is_presented_as_an_error():
    """The other half of that claim: a real failure must not look like a result."""
    assert frontend.STATUS_PRESENTATION["ERROR"]["kind"] == "error"


def test_a_partially_verified_answer_is_flagged_rather_than_celebrated():
    """It carries a citation nobody could confirm, so it is not a plain success."""
    assert frontend.STATUS_PRESENTATION["PARTIALLY_VERIFIED"]["kind"] == "warning"


def test_the_ui_reads_its_api_url_from_config():
    """So POLICYLENS_API_URL moves the UI without editing it."""
    from src import config

    assert frontend.api("/health") == f"{config.API_BASE_URL}/health"
