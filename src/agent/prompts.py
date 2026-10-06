"""Every prompt the agent sends, in one file.

WHY together rather than beside each node: prompts are the part of this system
most often edited, and they have to stay consistent with one another -- the
analyzer is told to cite by evidence number, so the verifier's judge prompt
must speak the same language. Keeping them adjacent makes a mismatch visible.

A shared convention runs through all of them: the model never sees chunk ids.
Evidence is labelled [1], [2], ... and the node maps those small integers back
to chunk ids itself. WHY: ids look like "Nepal_Labour_Act_2074.pdf::p8::c42",
and a model asked to echo one will eventually mistype it or invent a plausible
variant -- and a citation whose id does not resolve cannot be verified at all.
A single digit it cannot get subtly wrong.
"""

from __future__ import annotations

from src import config
from src.agent.state import RetrievedChunk

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
# The router is deliberately NOT the abstention gate -- that is the retrieval
# threshold, which is measured. The router only separates "a question about
# these documents" from "something else entirely", and rewrites the question
# into the documents' vocabulary. It is told to err towards DOCUMENT_QA for
# exactly that reason: a wrong OUT_OF_SCOPE refuses a question the corpus could
# have answered, and no evidence is ever consulted to justify it.

ROUTER_SYSTEM = """You classify questions for a document question-answering \
system. You never answer the question itself. You reply with JSON only."""

ROUTER_PROMPT = """The system has these documents indexed:
{document_list}

User question:
{question}

Classify the question and rewrite it for a semantic search over those
documents.

intent -- choose exactly one:
  "DOCUMENT_QA"  the question asks about the subject matter of one or more of
                 these documents.
  "COMPARISON"   the question asks how two or more of these documents differ,
                 agree, or relate to each other.
  "OUT_OF_SCOPE" the question is not about these documents' subject matter at
                 all (e.g. cooking, sport, general programming), or it asks
                 you to do something other than answer from them (e.g. write
                 a poem, translate arbitrary text).

Choose OUT_OF_SCOPE only when you are confident. If a question plausibly
touches a document's subject matter, choose DOCUMENT_QA even if you doubt the
answer is present -- whether the evidence exists is decided later, by search.

target_documents -- the filenames from the list above that the question is
about, copied EXACTLY. Use an empty list if the question does not single any
out, or if the intent is OUT_OF_SCOPE.

search_query -- the question restated in the vocabulary the documents would
use, as a short phrase suited to semantic search. Expand informal wording into
the formal terms a legal or policy document uses, and drop conversational
framing. Keep every meaningful constraint (numbers, named parties, conditions).
Example: "hey how long can my boss keep me on trial before hiring me properly?"
-> "maximum probation period of worker before permanent appointment"

reason -- one sentence on why you chose that intent.

Reply with JSON only:
{{"intent": "...", "target_documents": [...], "search_query": "...", "reason": "..."}}"""


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------
# The two rules that matter are the ones a model breaks by default:
#   1. Answer ONLY from the evidence. A model that has read labour law in
#      pre-training will happily supply a number the chunk never mentions,
#      and it will look right, which is what makes it dangerous.
#   2. Quote VERBATIM. The verifier re-reads the chunk and fuzzy-matches the
#      quote; a tidied-up or paraphrased quote fails that check, so asking for
#      an exact span is what makes verification possible rather than theatre.

ANALYZER_SYSTEM = """You answer questions strictly from supplied excerpts of \
legal and policy documents. You quote exactly. You never use knowledge from \
outside the excerpts. You reply with JSON only."""

ANALYZER_PROMPT = """Question:
{question}

Evidence -- numbered excerpts from the indexed documents:
{evidence}

Answer the question using ONLY these excerpts.

Rules:
1. Use nothing but the excerpts. If you know something from elsewhere that
   contradicts or extends them, leave it out.
2. If the excerpts do not contain the answer, set "insufficient_evidence" to
   true and leave "answer" empty. Do not answer partially from guesswork. This
   is a correct and expected outcome, not a failure.
3. Every factual claim in the answer must be supported by a citation.
4. A citation's "quote" must be copied VERBATIM from the excerpt -- the exact
   characters, including numbers and punctuation. Do not paraphrase, tidy,
   translate or shorten with ellipses. The quote is checked against the
   excerpt automatically, and an altered quote is rejected.
5. Quote at least {min_quote_chars} characters: a complete clause or sentence
   that stands on its own as evidence, not a fragment like "the employer
   shall".
6. Keep the answer under 150 words. State the rule and its conditions; do not
   restate the whole excerpt.
7. If the excerpts conflict with each other, say so and cite both.

Reply with JSON only:
{{"answer": "...",
  "citations": [{{"evidence": 1, "quote": "verbatim text from excerpt 1"}}],
  "insufficient_evidence": false}}"""


# ---------------------------------------------------------------------------
# Citation judge
# ---------------------------------------------------------------------------
# Reached only when the fuzzy match has already failed, so it is asked the
# narrow question the fuzzy check cannot answer: is this the same statement in
# different words, or a different statement? It must not be allowed to drift
# into judging whether the ANSWER is correct -- that is not its job and would
# make a wrong answer verifiable.

JUDGE_SYSTEM = """You check whether a quotation is faithful to a source \
excerpt. You judge only the quotation, never whether any answer is correct. \
You reply with JSON only."""

JUDGE_PROMPT = """Source excerpt (the authoritative text):
\"\"\"
{chunk_text}
\"\"\"

Quotation that claims to come from that excerpt:
\"\"\"
{quote}
\"\"\"

An automatic character-level match already failed, so the quotation is not a
verbatim copy. Decide whether it is nonetheless FAITHFUL: does the excerpt
state the same thing, with the same meaning, numbers and conditions?

supported = true  -- the excerpt does state this, in slightly different words
                     (reordered, reworded, whitespace or punctuation
                     differences, a merged pair of adjacent sentences).
supported = false -- the excerpt does not state this: a number, party,
                     condition, timeframe or scope differs; it is a general
                     claim the excerpt never makes; or it comes from somewhere
                     else entirely.

Be strict about specifics. "fifty percent" for the excerpt's "one hundred
percent" is false. An added qualifier the excerpt does not contain is false.
Judge only this quotation against this excerpt.

Reply with JSON only:
{{"supported": true, "reason": "one short sentence"}}"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered evidence block.

    Each excerpt is headed with its document, page and section. WHY show the
    locator to the model rather than only attaching it afterwards: a comparison
    question needs the model to know which document a passage came from in
    order to compare them at all, and a visible "Section 13" lets the answer
    refer to the provision by name the way a lawyer would.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        locator = chunk.get("source_document", "unknown")
        page = chunk.get("page_number")
        if page:
            locator += f", page {page}"
        section = chunk.get("section")
        if section:
            locator += f", {section}"

        text = (chunk.get("text") or "").strip()
        if len(text) > config.EVIDENCE_MAX_CHARS_PER_CHUNK:
            text = text[: config.EVIDENCE_MAX_CHARS_PER_CHUNK] + " [...]"

        blocks.append(f"[{index}] ({locator})\n{text}")
    return "\n\n".join(blocks)


def build_router_prompt(question: str, documents: list[str]) -> str:
    """Fill the router prompt with the documents actually indexed.

    The list comes from the vector store, never from a constant: a router told
    about a document that is not indexed will route confidently to nothing.
    """
    document_list = (
        "\n".join(f"  - {name}" for name in documents)
        if documents
        else "  (no documents are indexed)"
    )
    return ROUTER_PROMPT.format(document_list=document_list, question=question)


def build_analyzer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Fill the analyzer prompt with the question and numbered evidence."""
    return ANALYZER_PROMPT.format(
        question=question,
        evidence=format_evidence(chunks),
        min_quote_chars=config.MIN_CITATION_QUOTE_CHARS,
    )


def build_judge_prompt(quote: str, chunk_text: str) -> str:
    """Fill the citation-judge prompt for one quote and its source chunk."""
    return JUDGE_PROMPT.format(chunk_text=chunk_text.strip(), quote=quote.strip())
