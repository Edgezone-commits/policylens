"""Download the sample policy documents used for development and evaluation.

    python scripts/download_sample_docs.py

These are publicly available legal documents. Each URL was verified live, but
third-party hosts move files, so every download that fails prints the manual
instructions needed to fetch it by hand -- the script never leaves you guessing
why a document is missing.

NOTE: downloaded files land in data/, which is gitignored. Source documents are
deliberately never committed: they are large, and the index is reproducible
from them with scripts/ingest_docs.py.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

logger = logging.getLogger("download_sample_docs")

# Pretend to be a browser: several government and law-firm hosts reject
# requests with no User-Agent outright.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PolicyLens/0.1)"}
_TIMEOUT = 60


@dataclass
class SampleDocument:
    """One downloadable sample document."""

    filename: str
    description: str
    # Several URLs per document: hosts for Nepali legal translations are not
    # especially stable, so we try mirrors in order before giving up.
    urls: list[str]
    manual_instructions: str
    # Why this document is in the evaluation corpus at all.
    role: str
    expected_min_bytes: int = 10_000
    aliases: list[str] = field(default_factory=list)


SAMPLE_DOCUMENTS: list[SampleDocument] = [
    SampleDocument(
        filename="Nepal_Labour_Act_2074.pdf",
        description="Nepal Labour Act, 2074 (2017) -- official English translation",
        urls=[
            "https://antislaverylaw.ac.uk/wp-content/uploads/2019/08/The-Labour-Act-2017.pdf",
        ],
        manual_instructions=(
            "Search the Nepal Law Commission site (lawcommission.gov.np) for\n"
            "   'Labour Act 2074 English', or the ILO NATLEX record at\n"
            "   https://natlex.ilo.org/dyn/natlex2/r/natlex/fe/details?p3_isn=105434\n"
            "   Save the English PDF as data/raw/Nepal_Labour_Act_2074.pdf"
        ),
        role="Primary document: 82 pages, the source for most test questions.",
        # An ILO NATLEX download of the same Act, which may already be present.
        aliases=["NEP225978.pdf", "The-Labour-Act-2017.pdf"],
    ),
    SampleDocument(
        # Named "_Summary_PKF" deliberately. This is a third-party
        # rule-by-rule summary by PKF/Trunco, NOT the official statute, and the
        # filename appears verbatim in every citation the system produces. A
        # citation reading "Nepal_Labour_Rules_2075.pdf" would imply the user
        # was being quoted the official Rules, which would be misleading.
        #
        # We would prefer the official English Rules, but they are not
        # available as machine-readable text: the Government of Nepal edition
        # (ceslam.org/wp-content/uploads/2025/01/GoN_2018_LabourRules.pdf) is a
        # 104-page scanned image with zero extractable characters and would
        # need an OCR stage PolicyLens does not have. See the README's
        # Limitations section -- this is the concrete case behind the claim
        # that scanned PDFs are unsupported.
        filename="Nepal_Labour_Rules_2075_Summary_PKF.pdf",
        description=(
            "Nepal Labour Rules, 2075 (2018) -- rule-by-rule summary "
            "(third-party, PKF/Trunco; not the official statute)"
        ),
        urls=[
            "https://pkf.trunco.com.np/files/publications/1657093966_Labor%20Rules%202018_20181125025258_20181126031446.pdf",
        ],
        manual_instructions=(
            "Search for 'Nepal Labour Rules 2075 2018 English PDF'. Several\n"
            "   Nepali law firms publish English summaries. The official text\n"
            "   is on the Nepal Law Commission site but only as a scan, which\n"
            "   this pipeline cannot read without OCR.\n"
            "   Save it as data/raw/Nepal_Labour_Rules_2075_Summary_PKF.pdf"
        ),
        role=(
            "Second document, for cross-document questions: the Rules "
            "implement the Act, so the two overlap on gratuity, provident "
            "fund, working hours and leave. Being a summary it supports "
            "comparison questions but is weaker ground truth than the Act -- "
            "noted in docs/EVALUATION.md."
        ),
        aliases=["Nepal_Labour_Rules_2075.pdf"],
    ),
    SampleDocument(
        filename="Apache_License_2.0.txt",
        description="Apache License 2.0 -- plain text",
        urls=["https://www.apache.org/licenses/LICENSE-2.0.txt"],
        manual_instructions=(
            "Open https://www.apache.org/licenses/LICENSE-2.0.txt and save it\n"
            "   as data/raw/Apache_License_2.0.txt"
        ),
        role=(
            "A short, stable, unambiguous document. Useful as a control: it "
            "shares no vocabulary with Nepali labour law, so questions about "
            "one must NOT retrieve chunks from the other."
        ),
        expected_min_bytes=5_000,
    ),
]


def _looks_like_pdf(path: Path) -> bool:
    """Check the magic bytes.

    WHY: a failed download often returns an HTML error page with HTTP 200. That
    file would then be 'successfully downloaded' and ingested as garbage. The
    header check catches it at download time instead of at query time.
    """
    with path.open("rb") as handle:
        return handle.read(5) == b"%PDF-"


def download_document(doc: SampleDocument, target_dir: Path) -> bool:
    """Try each mirror for one document. Returns True if the file is present."""
    target = target_dir / doc.filename

    if target.exists() and target.stat().st_size >= doc.expected_min_bytes:
        logger.info("Already present, skipping: %s", doc.filename)
        return True

    # The same document may already be in data/raw/ under its original
    # download name (e.g. an ILO NATLEX id). Reuse it rather than re-download.
    for alias in doc.aliases:
        candidate = target_dir / alias
        if candidate.exists() and candidate.stat().st_size >= doc.expected_min_bytes:
            logger.info("Found existing copy %s -> renaming to %s", alias, doc.filename)
            candidate.rename(target)
            return True

    for url in doc.urls:
        try:
            logger.info("Downloading %s", doc.filename)
            logger.debug("  from %s", url)
            response = requests.get(
                url, headers=_HEADERS, timeout=_TIMEOUT, stream=True
            )
            response.raise_for_status()

            # Write to a temp name first so an interrupted download cannot
            # leave a truncated file that looks valid to the ingest script.
            partial = target.with_suffix(target.suffix + ".part")
            with partial.open("wb") as handle:
                for block in response.iter_content(chunk_size=65_536):
                    handle.write(block)

            size = partial.stat().st_size
            if size < doc.expected_min_bytes:
                logger.warning("  too small (%d bytes), likely an error page", size)
                partial.unlink(missing_ok=True)
                continue

            if doc.filename.endswith(".pdf") and not _looks_like_pdf(partial):
                logger.warning("  not a PDF (probably an HTML error page)")
                partial.unlink(missing_ok=True)
                continue

            partial.replace(target)
            logger.info("  saved %s (%.1f KB)", doc.filename, size / 1024)
            return True

        except requests.RequestException as exc:
            logger.warning("  failed: %s", exc)

    return False


def main() -> int:
    config.configure_logging()
    config.ensure_directories()
    target_dir = config.RAW_DOCS_DIR

    print(f"\nDownloading sample documents into {target_dir}\n")

    succeeded: list[SampleDocument] = []
    failed: list[SampleDocument] = []
    for doc in SAMPLE_DOCUMENTS:
        print(f"-> {doc.description}")
        (succeeded if download_document(doc, target_dir) else failed).append(doc)
        print()

    print("=" * 72)
    print(f"Downloaded or already present: {len(succeeded)}/{len(SAMPLE_DOCUMENTS)}")
    for doc in succeeded:
        print(f"  [ok] {doc.filename}")

    if failed:
        print(f"\nCould not download {len(failed)} document(s) automatically.")
        print("Please fetch these by hand:\n")
        for doc in failed:
            print(f"  [--] {doc.filename}")
            print(f"       {doc.description}")
            print(f"       Why it is needed: {doc.role}")
            print(f"       How to get it: {doc.manual_instructions}\n")

    print("=" * 72)
    print("\nNext step -- index whatever you have:")
    print("  python scripts/ingest_docs.py data/raw/\n")
    print("Note: PolicyLens works with ANY text-based PDF. These samples are")
    print("only what the included evaluation set was written against.\n")

    # Exit 0 even on partial failure: the script's job is to report clearly,
    # and a missing third-party mirror is not a broken script.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
