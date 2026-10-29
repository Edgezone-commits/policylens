"""Tests that pin two properties of HOW this package imports.

Neither is about behaviour, and that is exactly why they need tests: both are
invisible at runtime, both are undone by an edit that looks like a tidy-up,
and the cost of undoing them is paid by everyone who runs the suite rather
than by whoever made the change.

  1. Importing src.retrieval.embeddings must not import sentence_transformers,
     because that pulls in torch and transformers and costs ~57s on the
     development machine. The suite injects a fake embedder and needs none of
     it; before the import was made lazy, the whole suite waited a minute
     at collection for a model no test uses.

  2. src.agent.llm must set LITELLM_LOCAL_MODEL_COST_MAP before importing
     litellm, which otherwise fetches its pricing map over the NETWORK while
     being imported. This project claims its tests need no key and no
     network; a dependency that reaches the internet to be imported makes
     that false in a way nothing else here would catch.
"""

from __future__ import annotations

import subprocess
import sys

import src.agent.llm  # noqa: F401 -- imported for its import-time side effect


def test_importing_the_embedding_wrapper_does_not_import_torch():
    """In a subprocess, because by the time this test runs in the full suite
    another module may legitimately have loaded the real model already.

    Asserted on sentence_transformers AND torch: the wrapper could stop
    importing one while still dragging in the other.
    """
    probe = (
        "import sys;"
        "import src.retrieval.embeddings;"
        "loaded=[name for name in ('sentence_transformers','torch','transformers')"
        " if name in sys.modules];"
        "print(','.join(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        "src.retrieval.embeddings pulled in "
        f"{result.stdout.strip()} at import time. Keep the "
        "`from sentence_transformers import SentenceTransformer` inside "
        "EmbeddingModel.__init__ -- see the comment there for the 57 seconds "
        "it costs."
    )


def test_the_llm_module_pins_litellms_cost_map_to_the_local_copy():
    """Set by src.agent.llm at import, which the import above has done."""
    import os

    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"


def test_the_cost_map_is_set_before_litellm_is_imported():
    """The order is the whole point: litellm reads the variable while being
    imported, so setting it afterwards would be a no-op that still looks
    right in a diff."""
    from pathlib import Path

    source = Path("src/agent/llm.py").read_text(encoding="utf-8")
    env_line = source.index("LITELLM_LOCAL_MODEL_COST_MAP")
    import_line = source.index("\nimport litellm")

    assert env_line < import_line
