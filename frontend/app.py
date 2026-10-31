"""Streamlit UI for PolicyLens.

    uvicorn src.api.app:app --port 8000      # terminal 1
    streamlit run frontend/app.py            # terminal 2

The UI reaches the agent only over HTTP. It never imports PolicyLensAgent,
which is deliberate on two counts: the API stays the single contract, so
nothing can be shown here that another client could not also obtain; and the
100MB embedding model stays in one process instead of being loaded again by
every Streamlit script rerun.

The one design rule, which drives every layout choice below: AN ABSTENTION IS
A RESULT, NOT AN ERROR. When the system says the documents do not support an
answer, that is the feature working. So it is rendered as a first-class
outcome with its own panel and its own explanation -- never as a red error
box, never as an empty state, never as a spinner that gave up. A UI that made
honesty look like a malfunction would teach users to distrust the one
behaviour worth trusting.

The second rule: a citation is shown WITH its verdict, always. An unverified
quote is displayed, labelled, with the reason -- not hidden. Hiding it would
leave the user with a confident answer and no way to see which part of it the
system could not stand behind.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

# How each status is presented. The wording matters more than it looks: these
# strings are the whole explanation most users will ever read about why they
# did or did not get an answer.
STATUS_PRESENTATION: dict[str, dict[str, str]] = {
    "ANSWERED": {
        "icon": "✅",
        "label": "Answered",
        "blurb": "Every quote below was checked against the passage it cites.",
        "kind": "success",
    },
    "PARTIALLY_VERIFIED": {
        "icon": "⚠️",
        "label": "Answered, with an unverified citation",
        "blurb": (
            "The answer is below, but at least one quote could not be "
            "confirmed against its source. The failing citation is shown and "
            "labelled rather than removed -- check it before relying on it."
        ),
        "kind": "warning",
    },
    "INSUFFICIENT_EVIDENCE": {
        "icon": "🔍",
        "label": "No supported answer in these documents",
        "blurb": (
            "This is a deliberate outcome, not a failure. The indexed "
            "documents do not contain evidence for this question, so the "
            "system declined to answer rather than produce something "
            "plausible and unsupported."
        ),
        "kind": "info",
    },
    "OUT_OF_SCOPE": {
        "icon": "🚫",
        "label": "Not a question about these documents",
        "blurb": (
            "The question was classified as being about something other than "
            "the indexed documents, so no search was run."
        ),
        "kind": "info",
    },
    "ERROR": {
        "icon": "❌",
        "label": "The system could not run",
        "blurb": (
            "This is different from having no answer: the pipeline itself "
            "failed, so nothing was concluded about the documents."
        ),
        "kind": "error",
    },
}


def api(path: str) -> str:
    return f"{config.API_BASE_URL.rstrip('/')}{path}"


def get_json(path: str, **kwargs: Any) -> Any:
    response = requests.get(api(path), timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def ask(question: str, document: str | None) -> tuple[int, Any]:
    """Put a question to the API. Returns (status_code, body).

    The status code comes back to the caller rather than being raised on,
    because a 404 for an unknown document and a 503 for a dead pipeline want
    different messages, and both are expected outcomes rather than bugs.
    """
    response = requests.post(
        api("/ask"),
        json={"question": question, "document": document},
        timeout=config.API_TIMEOUT_SECONDS,
    )
    return response.status_code, response.json()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_citation(citation: dict[str, Any], index: int) -> None:
    """One citation, with its verdict attached.

    The verdict is in the header, not buried in the body, because a reader
    skimming three citations needs to see which one to distrust without
    opening any of them.
    """
    verified = bool(citation.get("verified"))
    where = citation.get("section") or "section unknown"
    page = citation.get("page_number")
    locator = f"{where}, p.{page}" if page else where

    badge = "✅ verified" if verified else "⚠️ UNVERIFIED"
    header = f"{badge} — [{index}] {citation.get('source_document')} ({locator})"

    with st.expander(header, expanded=not verified):
        # Quoted as a blockquote so it reads as the document's words, not the
        # system's. The distinction is the entire point of a citation.
        st.markdown(f"> {citation.get('quote', '')}")

        method = citation.get("method", "unknown")
        score = citation.get("match_score", 0.0)
        st.caption(f"Check: `{method}`, character match {score:.0f}/100")

        if not verified:
            note = citation.get("note") or "no reason recorded"
            st.warning(
                f"**Why this did not verify:** {note}\n\n"
                "It is shown anyway so you can judge it yourself. Hiding it "
                "would leave the answer looking fully supported."
            )


def render_answer(body: dict[str, Any]) -> None:
    status = body.get("status", "ERROR")
    presentation = STATUS_PRESENTATION.get(status, STATUS_PRESENTATION["ERROR"])

    st.markdown(f"### {presentation['icon']} {presentation['label']}")
    getattr(st, presentation["kind"])(presentation["blurb"])

    answer = body.get("answer") or ""
    if answer:
        st.markdown(answer)

    citations = body.get("citations") or []
    if citations:
        verified = sum(1 for c in citations if c.get("verified"))
        st.markdown(f"#### Citations ({verified} of {len(citations)} verified)")
        for index, citation in enumerate(citations, start=1):
            render_citation(citation, index)
    elif status in ("ANSWERED", "PARTIALLY_VERIFIED"):
        # Should not happen -- the agent withdraws an uncited answer -- so if
        # it does, say so loudly rather than rendering a bare claim.
        st.error(
            "This answer arrived with no citations. Do not rely on it: every "
            "supported answer from this system carries quoted evidence."
        )

    render_metrics(body)

    trace = body.get("trace") or []
    if trace:
        with st.expander("How this answer was reached", expanded=False):
            st.caption(
                "One row per node. Shown because a system that asks to be "
                "trusted should be able to show its work."
            )
            for entry in trace:
                st.markdown(f"**{entry.get('node')}** — {entry.get('summary')}")
                detail = entry.get("detail") or {}
                if detail:
                    st.json(detail, expanded=False)


def render_metrics(body: dict[str, Any]) -> None:
    """The numbers behind the answer."""
    confidence = body.get("confidence") or 0.0
    columns = st.columns(4)
    columns[0].metric("Confidence", f"{confidence:.2f}")
    columns[1].metric("Top similarity", f"{body.get('top_similarity') or 0.0:.2f}")
    columns[2].metric(
        "Citations verified",
        f"{body.get('verified_count', 0)}/"
        f"{body.get('verified_count', 0) + body.get('unverified_count', 0)}",
    )
    columns[3].metric("Latency", f"{body.get('latency_seconds') or 0.0:.1f}s")

    usage = body.get("usage") or {}
    st.caption(
        f"Computed confidence, never self-reported: "
        f"{config.CONFIDENCE_WEIGHT_VERIFIED_RATIO:g}x verified fraction + "
        f"{config.CONFIDENCE_WEIGHT_RETRIEVAL_SCORE:g}x retrieval similarity - "
        f"{config.CONFIDENCE_PENALTY_PER_RETRY:g} per retry. "
        f"Cost: {usage.get('calls', 0)} LLM call(s), "
        f"{usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)} tokens"
        + (f", {body.get('retries')} retry" if body.get("retries") else "")
    )


def render_sidebar() -> list[str]:
    """Health, documents and upload. Returns the indexed document names."""
    st.sidebar.title("PolicyLens")
    st.sidebar.caption(
        "Answers from legal and policy documents, where every claim carries a "
        "quote that has been checked against the passage it cites."
    )

    try:
        health = get_json("/health")
    except requests.RequestException as error:
        st.sidebar.error(
            f"Cannot reach the API at {config.API_BASE_URL}\n\n"
            f"Start it with:\n`uvicorn src.api.app:app --port 8000`\n\n{error}"
        )
        return []

    ok = health.get("status") == "ok"
    st.sidebar.success("API ready") if ok else st.sidebar.warning("API degraded")

    # Named individually, because "degraded" on its own tells the user nothing
    # they can act on -- and these are the two things that actually stop it.
    if not health.get("llm_key_present"):
        st.sidebar.error("No LLM API key found. Set GOOGLE_API_KEY in .env")
    if not health.get("chunks_indexed"):
        st.sidebar.error("No documents indexed. Upload one below.")

    documents = health.get("documents") or []
    st.sidebar.metric("Chunks indexed", health.get("chunks_indexed", 0))
    st.sidebar.metric("Questions logged", health.get("queries_logged", 0))

    with st.sidebar.expander("Indexed documents", expanded=True):
        for document in documents:
            st.write(
                f"- {document['source_document']} "
                f"({document['chunk_count']} chunks)"
            )

    with st.sidebar.expander("Add a document"):
        upload = st.file_uploader("PDF or .txt", type=["pdf", "txt"])
        if upload is not None and st.button("Index it"):
            with st.spinner(f"Indexing {upload.name}..."):
                response = requests.post(
                    api("/documents"),
                    files={"file": (upload.name, upload.getvalue())},
                    timeout=config.API_TIMEOUT_SECONDS,
                )
            if response.status_code == 200:
                result = response.json()
                if result.get("succeeded"):
                    st.success(
                        f"Indexed {result['chunks']} chunks from "
                        f"{result['pages']} page(s)."
                    )
                    st.rerun()
                else:
                    # The scanned-PDF case. Named explicitly, because "0
                    # chunks" with no explanation looks like a broken upload.
                    st.warning(
                        f"Nothing was indexed: {result.get('skipped_reason')}. "
                        "A scanned PDF has no extractable text and would need "
                        "OCR first."
                    )
            else:
                st.error(response.json().get("detail", response.text))

    with st.sidebar.expander("Model chain"):
        st.write(f"**Primary:** `{health.get('model')}`")
        for model in health.get("fallback_models") or []:
            st.write(f"↳ `{model}`")
        st.caption(
            "The free tier's daily quota is charged per model, so each link "
            "is another day's worth of questions on one key."
        )
        st.write(f"**Embeddings:** `{health.get('embedding_model')}`")
        st.write(f"**Relevance threshold:** {health.get('similarity_threshold')}")

    return [document["source_document"] for document in documents]


def render_history() -> None:
    """Recent questions and the abstention rate."""
    try:
        stats = get_json("/stats")
        recent = get_json("/queries", params={"limit": 15})
    except requests.RequestException as error:
        st.error(f"Could not read the query log: {error}")
        return

    columns = st.columns(4)
    columns[0].metric("Questions asked", stats.get("total_queries", 0))
    columns[1].metric("Abstention rate", f"{(stats.get('abstention_rate') or 0):.0%}")
    columns[2].metric(
        "Mean confidence",
        (
            "n/a"
            if stats.get("mean_confidence") is None
            else f"{stats['mean_confidence']:.2f}"
        ),
    )
    columns[3].metric(
        "Mean latency",
        (
            "n/a"
            if stats.get("mean_latency_seconds") is None
            else f"{stats['mean_latency_seconds']:.1f}s"
        ),
    )
    st.caption(
        "The abstention rate is the honesty metric to watch: near zero means "
        "the system is answering things it cannot support, near one means it "
        "is refusing things it could."
    )

    st.markdown("#### Recent questions")
    if not recent:
        st.info("Nothing asked yet.")
        return
    for row in recent:
        presentation = STATUS_PRESENTATION.get(
            row.get("status", ""), STATUS_PRESENTATION["ERROR"]
        )
        with st.expander(
            f"{presentation['icon']} {row.get('question')}  " f"— {row.get('asked_at')}"
        ):
            st.write(row.get("answer") or "_no answer_")
            citations = row.get("citations") or []
            if citations:
                verified = sum(1 for c in citations if c.get("verified"))
                st.caption(f"{verified} of {len(citations)} citations verified")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="PolicyLens", page_icon="📑", layout="wide")

    documents = render_sidebar()

    ask_tab, history_tab = st.tabs(["Ask", "History"])

    with ask_tab:
        st.markdown("#### Ask a question about the indexed documents")

        with st.form("ask"):
            question = st.text_area(
                "Question",
                placeholder="What is the maximum probation period for a worker?",
                height=90,
            )
            columns = st.columns([3, 1])
            scope = columns[0].selectbox(
                "Search in",
                ["All documents", *documents],
                help=(
                    "Restricting the search overrides the router's own choice "
                    "of document."
                ),
            )
            columns[1].markdown("<br>", unsafe_allow_html=True)
            submitted = columns[1].form_submit_button("Ask", type="primary")

        if submitted:
            if len(question.strip()) < 3:
                st.warning("Ask something a little longer.")
                return

            document = None if scope == "All documents" else scope
            with st.spinner(
                "Routing, retrieving, drafting, then checking every quote "
                "against its source..."
            ):
                try:
                    code, body = ask(question.strip(), document)
                except requests.RequestException as error:
                    st.error(f"Could not reach the API: {error}")
                    return

            if code == 200:
                render_answer(body)
            elif code == 503:
                # The pipeline failed. Separated from an abstention on purpose:
                # "I could not run" and "I found no evidence" are different
                # claims, and conflating them would make a dead key look like
                # a documented absence of evidence.
                st.error(
                    f"**The system could not run.** {body.get('detail', '')}\n\n"
                    "This is not an abstention -- nothing was concluded about "
                    "the documents. Check the model chain in the sidebar."
                )
            else:
                st.warning(body.get("detail", f"HTTP {code}"))

    with history_tab:
        render_history()


if __name__ == "__main__":
    main()
