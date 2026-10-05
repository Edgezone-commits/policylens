"""Split page-level documents into embedding-sized chunks with metadata.

Chunking is where retrieval quality is won or lost. A chunk is both the unit
that gets embedded (so it must be topically tight) and the unit shown to the
user as evidence (so it must be readable on its own).
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

# Matches a numbered section heading in legal prose, e.g.
#   "11. Probation Period:"      "54. Leave:"      "3. Definitions:"
# WHY capture this: the evaluation measures retrieval recall against a ground
# truth expressed as "Section 11", and users judge a citation far more readily
# by section than by page. Page number alone is a weak locator in an 82-page
# Act. This is best-effort -- a document with no numbered sections simply gets
# no section metadata, and nothing downstream breaks.
_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d{1,3})\.\s+([A-Z][^\n:]{2,80})(?::|\s*$)", re.MULTILINE
)

# Matches a chapter heading, e.g. "Chapter-1" or "Chapter 10".
_CHAPTER_HEADING_RE = re.compile(r"^\s*Chapter\s*[-\s]?(\d{1,3})\b", re.MULTILINE)

# Matches a parenthesised rule reference, e.g. "Gratuity (Rule 23)".
# WHY a separate pattern: subordinate legislation is numbered by rule, not
# section, and the Labour Rules document heads each provision this way rather
# than with a leading "23.". Without this it produced no locator at all, so a
# citation to the Rules could only say "page 7" -- useless for checking a
# cross-document comparison against the Act.
_RULE_HEADING_RE = re.compile(r"\(Rules?\s*(\d{1,3})", re.IGNORECASE)


def _build_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    """Create the text splitter.

    WHY RecursiveCharacterTextSplitter rather than a fixed-width split: it
    tries separators in order, so it breaks at a paragraph boundary when it
    can, a sentence boundary when it cannot, and only falls back to mid-word
    as a last resort. That keeps legal clauses intact far more often than a
    blind 1000-character cut, which would routinely slice a sentence in half
    and leave both halves unquotable.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ordered most-structural to least. The default list does not include
        # the semicolon, which legal text uses heavily to separate clauses.
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        # Records each chunk's character offset within its page, which lets us
        # attribute a chunk to the section heading that precedes it.
        add_start_index=True,
    )


def _find_headings(text: str) -> list[tuple[int, str]]:
    """Return (offset, section_label) for every heading found in `text`."""
    headings: list[tuple[int, str]] = []
    for match in _SECTION_HEADING_RE.finditer(text):
        headings.append((match.start(), f"Section {match.group(1)}"))
    for match in _CHAPTER_HEADING_RE.finditer(text):
        headings.append((match.start(), f"Chapter {match.group(1)}"))
    for match in _RULE_HEADING_RE.finditer(text):
        headings.append((match.start(), f"Rule {match.group(1)}"))
    headings.sort(key=lambda item: item[0])
    return headings


def _section_for_offset(
    headings: list[tuple[int, str]], offset: int, carried: str | None
) -> str | None:
    """Find the innermost heading at or before `offset`.

    `carried` is the last heading seen on an earlier page: a section routinely
    spans a page break, so a chunk with no heading above it on its own page
    still belongs to the section that started on the previous page.
    """
    current = carried
    for heading_offset, label in headings:
        if heading_offset <= offset:
            current = label
        else:
            break
    return current


def chunk_documents(
    pages: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split pages into chunks, each carrying the metadata citations need.

    Every chunk gets:
      - source_document: filename, for the citation
      - page_number:     1-indexed page, for the citation
      - chunk_index:     position in the whole document, for stable ordering
      - chunk_id:        unique, deterministic id -- the verifier uses it to
                         look a citation's chunk back up by name
      - section:         best-effort "Section N" label, may be absent

    Chunk ids are deterministic (not random UUIDs) so that re-ingesting the
    same document overwrites its chunks instead of silently duplicating them.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    splitter = _build_splitter(chunk_size, chunk_overlap)

    chunks: list[Document] = []
    carried_section: str | None = None
    dropped = 0  # fragments below MIN_CHUNK_CHARS

    # Sorted so chunk_index and the carried section follow true reading order,
    # which the loader does not strictly guarantee.
    pages = sorted(pages, key=lambda page: page.metadata.get("page_number", 0))

    for page in pages:
        headings = _find_headings(page.page_content)
        for piece in splitter.split_documents([page]):
            # Drop fragments too short to ever be usable evidence (see
            # MIN_CHUNK_CHARS in config for why they are actively harmful).
            if len(piece.page_content.strip()) < config.MIN_CHUNK_CHARS:
                dropped += 1
                continue

            start_index = int(piece.metadata.get("start_index", 0))
            end_index = start_index + len(piece.page_content)

            # The section the chunk STARTS in...
            section = _section_for_offset(headings, start_index, carried_section)
            # ...and every section whose heading falls INSIDE the chunk.
            #
            # WHY both: a 1000-character chunk routinely straddles a section
            # boundary. The chunk holding the Labour Act's probation answer
            # begins inside Section 12 and contains the whole of Section 13,
            # so labelling it "Section 12" alone would be actively misleading
            # in a citation -- and would make the Phase 6 recall metric, which
            # matches retrieved chunks against ground-truth section numbers,
            # report a false miss for a chunk that genuinely holds the answer.
            spanned = [section] if section else []
            spanned += [
                label
                for offset, label in headings
                if start_index <= offset < end_index
            ]
            # Deduplicate while preserving reading order.
            spanned = list(dict.fromkeys(spanned))

            source = page.metadata["source_document"]
            page_number = page.metadata["page_number"]
            chunk_index = len(chunks)

            metadata = {
                "source_document": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
                # Readable and unique: "LabourAct.pdf::p8::c42"
                "chunk_id": f"{source}::p{page_number}::c{chunk_index}",
            }
            # Chroma metadata values must be str/int/float/bool -- a list is
            # rejected -- so the spanned sections are stored comma-joined and
            # callers membership-test the string.
            if section:
                metadata["section"] = section
            if spanned:
                metadata["sections"] = ",".join(spanned)

            chunks.append(Document(page_content=piece.page_content, metadata=metadata))

        if headings:
            carried_section = headings[-1][1]

    if chunks:
        sizes = [len(chunk.page_content) for chunk in chunks]
        with_section = sum(1 for chunk in chunks if "section" in chunk.metadata)
        logger.info(
            "Created %d chunks (mean %d chars, min %d, max %d); "
            "%d/%d have a section label; dropped %d fragment(s) under %d chars",
            len(chunks),
            sum(sizes) // len(sizes),
            min(sizes),
            max(sizes),
            with_section,
            len(chunks),
            dropped,
            config.MIN_CHUNK_CHARS,
        )
    else:
        logger.warning("Chunking produced no chunks -- was the input empty?")

    return chunks
