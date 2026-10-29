"""Tests for the ingestion pipeline: loading, cleaning and chunking.

This layer had no tests at all until now, which was the wrong gap to have.
Everything downstream is built on the metadata a chunk carries -- the verifier
looks a citation's chunk back up BY ID, and the API shows the document, page
and section to a reader who wants to check the source themselves. If a chunk
loses its page number or gets a non-deterministic id, the failure does not
look like an ingestion bug: it looks like a citation that cannot be verified,
or a re-ingest that silently doubles the index.

No PDF is needed. `tests/fixtures/sample_policy.txt` is a policy document in
the same shape as the real corpus -- numbered sections, a chapter heading, a
repeated header -- which is enough to exercise cleaning, section detection and
chunk metadata. The one PDF-specific path (PyPDFLoader) is left to the live
corpus; what is tested here is everything that happens to the text afterwards,
which is where the citation-critical logic lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src import config
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import SUPPORTED_EXTENSIONS, ingest_file
from src.ingestion.loader import (
    clean_page_text,
    find_boilerplate_lines,
    load_text_file,
    remove_boilerplate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_policy.txt"


def page(text: str, number: int = 1, name: str = "policy.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"source_document": name, "page_number": number},
    )


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------


def test_the_fixture_loads_as_one_page_with_citation_metadata():
    pages = load_text_file(FIXTURE)

    assert len(pages) == 1
    # Page 1 for a flat text file: there are no pages, but every citation
    # needs the field, and 1 is honest for a single unit.
    assert pages[0].metadata["page_number"] == 1
    assert pages[0].metadata["source_document"] == "sample_policy.txt"
    assert "Probation" in pages[0].page_content


def test_a_missing_file_fails_loudly_rather_than_returning_nothing():
    """An empty list would look like an empty document and index silently."""
    with pytest.raises(FileNotFoundError):
        load_text_file(FIXTURE.parent / "does_not_exist.txt")


def test_cleaning_collapses_the_whitespace_a_pdf_extractor_leaves_behind():
    messy = "The  employer   shall\n\n\n   pay    the  labour."

    cleaned = clean_page_text(messy)

    assert "  " not in cleaned
    assert "employer shall" in cleaned


def test_cleaning_an_empty_page_yields_an_empty_string():
    """A scanned page extracts to nothing, and must not become a chunk."""
    assert clean_page_text("   \n \t \n ") == ""


def test_a_repeated_header_is_detected_as_boilerplate():
    """It appears on every page, so it is navigation, not evidence.

    Left in, it would be indexed dozens of times and could be retrieved and
    quoted as though it answered something.
    """
    header = "ACME UNIVERSITY POLICY -- CONFIDENTIAL"
    pages = [
        page(f"{header}\n1. Short title: this policy is called the Acme policy.", 1),
        page(f"{header}\n2. Definitions: in this policy, unless stated.", 2),
        page(f"{header}\n3. Probation: six months from appointment.", 3),
        page(f"{header}\n4. Notice: thirty days before termination.", 4),
    ]

    boilerplate = find_boilerplate_lines(pages)

    assert header in boilerplate


def test_boilerplate_detection_needs_enough_pages_to_judge_by():
    """On a three-page document "repeated" carries no information.

    Guarding this matters: with two pages, any line appearing on both clears a
    50% share, so a short document's real content would be stripped as a
    header.
    """
    header = "ACME UNIVERSITY POLICY -- CONFIDENTIAL"
    pages = [page(f"{header}\n1. Short title.", n) for n in (1, 2, 3)]

    assert find_boilerplate_lines(pages) == set()


def test_a_line_appearing_once_is_not_boilerplate():
    """Removing a one-off line would delete real content."""
    pages = [
        page("13. Probation: a labour may be kept on probation for six months.", 1),
        page("14. Notice: the employer shall give notice before terminating.", 2),
    ]

    boilerplate = find_boilerplate_lines(pages)

    assert not any("Probation" in line for line in boilerplate)


def test_removing_boilerplate_keeps_the_substance_of_the_page():
    header = "ACME UNIVERSITY POLICY -- CONFIDENTIAL"
    pages = [
        page(f"{header}\n13. Probation: six months from the date of appointment.", 1),
        page(f"{header}\n14. Notice: thirty days before termination.", 2),
        page(f"{header}\n15. Gratuity: deposited every month.", 3),
        page(f"{header}\n16. Records: kept for five years.", 4),
    ]

    stripped = remove_boilerplate(pages, find_boilerplate_lines(pages))

    for document in stripped:
        assert header not in document.page_content
    assert "Probation" in stripped[0].page_content
    assert "Notice" in stripped[1].page_content


# ---------------------------------------------------------------------------
# Chunking -- the metadata every citation depends on
# ---------------------------------------------------------------------------


def test_every_chunk_carries_what_a_citation_needs():
    chunks = chunk_documents(load_text_file(FIXTURE))

    assert chunks, "the fixture produced no chunks"
    for chunk in chunks:
        metadata = chunk.metadata
        # Without any one of these, a citation cannot be shown to a reader or
        # looked back up by the verifier.
        assert metadata["source_document"] == "sample_policy.txt"
        assert metadata["page_number"] == 1
        assert isinstance(metadata["chunk_index"], int)
        assert metadata["chunk_id"]


def test_chunk_ids_are_unique_within_a_document():
    """Two chunks sharing an id would make the verifier check the wrong text."""
    chunks = chunk_documents(load_text_file(FIXTURE))

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_deterministic_so_re_ingesting_overwrites():
    """Random ids would double the index on every re-ingest, silently."""
    first = chunk_documents(load_text_file(FIXTURE))
    second = chunk_documents(load_text_file(FIXTURE))

    assert [c.metadata["chunk_id"] for c in first] == [
        c.metadata["chunk_id"] for c in second
    ]


def test_chunk_index_follows_reading_order_across_pages():
    """Pages can arrive out of order; the index must not."""
    pages = [
        page("14. Notice. " + "The employer shall give notice in writing. " * 12, 2),
        page("13. Probation. " + "A labour may be kept on probation. " * 12, 1),
    ]

    chunks = chunk_documents(pages)

    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == sorted(indices)
    # The page-1 content sorts first even though it was supplied second.
    assert chunks[0].metadata["page_number"] == 1


def test_a_chunk_is_labelled_with_the_section_it_starts_in():
    text = (
        "13. Probation Period: A labour may be kept on probation for a "
        "maximum period of six months from the date of appointment, and the "
        "employer may terminate the contract during that period."
    )

    chunks = chunk_documents([page(text)])

    assert chunks[0].metadata.get("section") == "Section 13"


def test_fragments_too_short_to_be_evidence_are_dropped():
    """A ten-character chunk would fuzzy-match almost any legal prose.

    See MIN_CHUNK_CHARS: a partial match of three or four words scores 100
    against a passage that says nothing about the question, so such a chunk
    could be stamped "verified" while supporting nothing.
    """
    chunks = chunk_documents([page("13. Short."), page("14. Also short.", 2)])

    assert chunks == []


def test_chunking_respects_the_configured_size():
    long_page = page("The employer shall pay the labour on time. " * 200)

    chunks = chunk_documents([long_page], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        # The splitter cuts on separators, so it can overshoot slightly rather
        # than breaking mid-word; the bound is the point, not the exact number.
        assert len(chunk.page_content) <= 300 + 100


def test_overlap_keeps_a_sentence_from_being_lost_at_a_boundary():
    """A fact split across two chunks must survive in at least one of them."""
    sentence = "Gratuity shall be eight point three three percent of basic pay."
    filler = "The employer shall maintain records of attendance. " * 20
    chunks = chunk_documents(
        [page(filler + sentence + filler)], chunk_size=400, chunk_overlap=120
    )

    assert any("eight point three three percent" in c.page_content for c in chunks)


def test_an_empty_page_list_chunks_to_nothing_without_raising():
    assert chunk_documents([]) == []


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class RecordingStore:
    """Captures what would be indexed, without an embedding model."""

    def __init__(self) -> None:
        self.chunks: list[Document] = []

    def add_chunks(self, chunks: list[Document]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)


def test_the_supported_extensions_are_the_ones_the_loader_handles():
    assert {".pdf", ".txt"} == SUPPORTED_EXTENSIONS


def test_ingesting_the_fixture_reports_pages_and_chunks():
    store = RecordingStore()

    result = ingest_file(FIXTURE, store)  # type: ignore[arg-type]

    assert result.succeeded
    assert result.source_document == "sample_policy.txt"
    assert result.pages == 1
    assert result.chunks == len(store.chunks) > 0
    assert result.skipped_reason is None


def test_an_unsupported_extension_is_skipped_with_a_reason(tmp_path):
    """Reported rather than raised: a directory ingest should not stop dead."""
    odd = tmp_path / "notes.docx"
    odd.write_bytes(b"PK\x03\x04")
    store = RecordingStore()

    result = ingest_file(odd, store)  # type: ignore[arg-type]

    assert not result.succeeded
    assert result.skipped_reason
    assert store.chunks == []


def test_a_document_with_no_extractable_text_indexes_nothing(tmp_path):
    """The scanned-PDF case: it must not sit in the index looking ingested."""
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n  \n ", encoding="utf-8")
    store = RecordingStore()

    result = ingest_file(blank, store)  # type: ignore[arg-type]

    assert not result.succeeded
    assert result.chunks == 0
    assert result.skipped_reason
    assert store.chunks == []


def test_a_document_of_only_short_fragments_is_reported_not_silently_empty(tmp_path):
    """Text was extracted but nothing survived the minimum-length rule."""
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("1. A.\n2. B.\n3. C.\n", encoding="utf-8")
    store = RecordingStore()

    result = ingest_file(tiny, store)  # type: ignore[arg-type]

    assert not result.succeeded
    assert result.skipped_reason
    assert config.MIN_CHUNK_CHARS > 0
