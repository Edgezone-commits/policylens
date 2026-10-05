"""Load PDFs into page-level documents, then clean the extracted text.

WHY cleaning is a real step and not paranoia: PDF text extraction is lossy.
The Nepal Labour Act translation used in development comes out of pypdf with
inline BCP-47 language tags glued onto words ("en-INThe Labour Act") and a
"www.lawcommission.gov.np 12" header repeated on all 82 pages. Left alone,
that boilerplate lands in every chunk, where it dilutes the embedding (every
chunk looks slightly like every other chunk) and wastes the LLM's context.

All cleaning here is generic, not tuned to one file, because the system has to
work with any uploaded PDF.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# PDF structure tags that leak into extracted text as language markers, e.g.
# "en-IN", "ne-NP". Real prose essentially never contains this pattern, and
# when glued to a word it corrupts the token ("en-INThe"), so removing it is
# strictly an improvement.
_LANGUAGE_TAG_RE = re.compile(r"\b[a-z]{2}-[A-Z]{2}\b")

# Collapse runs of whitespace. PDF extraction pads with spaces to mimic visual
# layout; those runs carry no meaning for an embedding model.
_WHITESPACE_RE = re.compile(r"[ \t ]+")

# Three or more newlines -> a paragraph break. Keeps structure without the gaps.
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# A line must appear on at least this share of pages to count as boilerplate.
# 0.5 is deliberately conservative: a genuine heading repeated on half the
# pages of a legal document is far less likely than a running header.
_BOILERPLATE_PAGE_SHARE = 0.5

# Below this many pages, the frequency test is not statistically meaningful --
# in a 2-page document a line on both pages may simply be content.
_BOILERPLATE_MIN_PAGES = 4


def clean_page_text(text: str) -> str:
    """Normalise the text of a single extracted page.

    Deliberately conservative: it removes extraction artefacts but never
    rewords content, because the verifier later checks quoted citations
    character-by-character against this text. Aggressive cleaning here would
    make honest citations fail verification.
    """
    text = _LANGUAGE_TAG_RE.sub(" ", text)

    # Drop soft hyphens and zero-width characters, which split words invisibly.
    text = text.replace("­", "").replace("​", "")

    # Re-join words broken across a line by a hyphen ("employ-\nment").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Repair hyphenated words that extraction split with a stray space, e.g.
    # "sub -section" -> "sub-section", "Chapter -5" -> "Chapter-5". The Labour
    # Act has 244 of these and they matter twice over: they corrupt the
    # embedding, and they would make an LLM's honest quote of "sub-section"
    # fail a character-level check against the stored chunk.
    #
    # Deliberately narrow: it only fires when there is a space BEFORE the
    # hyphen and none AFTER it. That is the signature of the artefact, whereas
    # a dash used as punctuation ("the labour - who was absent - shall") has
    # spaces on both sides and is left untouched.
    text = re.sub(r"(\w) +-(\w)", r"\1-\2", text)

    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)

    # Tidy each line, then drop lines that are now empty.
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def find_boilerplate_lines(pages: list[Document]) -> set[str]:
    """Identify running headers/footers by how often a line repeats.

    WHY frequency rather than position: a header is not reliably the first
    extracted line (column order varies), but it is reliably on most pages.
    Page numbers are stripped from the line before counting so that
    "www.lawcommission.gov.np 1" and "...gov.np 2" are seen as one header.
    """
    if len(pages) < _BOILERPLATE_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        # A line only counts once per page, so a word repeated within one page
        # cannot masquerade as boilerplate.
        seen = {
            _strip_trailing_page_number(line)
            for line in page.page_content.split("\n")
            if line.strip()
        }
        counts.update(seen)

    threshold = len(pages) * _BOILERPLATE_PAGE_SHARE
    boilerplate = {
        line
        for line, count in counts.items()
        # Long lines are almost certainly real content even if repeated, so we
        # only strip short ones -- headers and footers are short by nature.
        if count >= threshold and len(line) <= 120
    }

    if boilerplate:
        logger.info(
            "Detected %d repeated header/footer line(s) to strip: %s",
            len(boilerplate),
            sorted(boilerplate)[:3],
        )
    return boilerplate


def _strip_trailing_page_number(line: str) -> str:
    """Normalise a line by removing a leading/trailing bare page number."""
    line = line.strip()
    line = re.sub(r"^\d{1,4}\s+", "", line)
    line = re.sub(r"\s+\d{1,4}$", "", line)
    return line.strip()


def remove_boilerplate(pages: list[Document], boilerplate: set[str]) -> list[Document]:
    """Strip the detected running headers/footers from every page."""
    if not boilerplate:
        return pages

    for page in pages:
        kept = [
            line
            for line in page.page_content.split("\n")
            if _strip_trailing_page_number(line) not in boilerplate
        ]
        page.page_content = "\n".join(kept).strip()
    return pages


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load one PDF into a list of cleaned, page-level Documents.

    Each returned Document carries the metadata the citation system depends on:
      - source_document: the filename, shown to the user in a citation
      - page_number:     1-indexed, because that is what a human reader sees
                         (pypdf counts from 0, which would make every citation
                         off by one)

    Returns an empty list if the PDF has no extractable text, which is how a
    scanned PDF presents itself. We do not attempt OCR -- see the README's
    Limitations section.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s", pdf_path.name)
    pages = PyPDFLoader(str(pdf_path)).load()

    cleaned: list[Document] = []
    for page in pages:
        text = clean_page_text(page.page_content)
        if not text:
            continue  # blank or image-only page
        cleaned.append(
            Document(
                page_content=text,
                metadata={
                    "source_document": pdf_path.name,
                    # pypdf is 0-indexed; humans and citations are 1-indexed.
                    "page_number": int(page.metadata.get("page", 0)) + 1,
                },
            )
        )

    cleaned = remove_boilerplate(cleaned, find_boilerplate_lines(cleaned))
    # Boilerplate removal can empty a page that held nothing but a header.
    cleaned = [page for page in cleaned if page.page_content.strip()]

    total_chars = sum(len(page.page_content) for page in cleaned)
    if not cleaned:
        logger.warning(
            "No extractable text in %s -- it is probably a scanned PDF, which "
            "would need OCR before indexing.",
            pdf_path.name,
        )
    else:
        logger.info(
            "Loaded %d page(s) with text from %s (%d characters)",
            len(cleaned),
            pdf_path.name,
            total_chars,
        )
    return cleaned


def load_text_file(text_path: str | Path) -> list[Document]:
    """Load a plain .txt document as a single page.

    Needed because not every useful policy document is a PDF -- the Apache
    License 2.0, one of our evaluation documents, is published as plain text.
    It also gives the test suite a fixture that does not require a PDF.
    """
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")

    text = clean_page_text(text_path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        logger.warning("No text in %s", text_path.name)
        return []

    logger.info("Loaded %s (%d characters)", text_path.name, len(text))
    return [
        Document(
            page_content=text,
            # Page 1 for a flat text file: there are no pages, but the citation
            # schema needs the field, and "1" is honest for a single unit.
            metadata={"source_document": text_path.name, "page_number": 1},
        )
    ]
