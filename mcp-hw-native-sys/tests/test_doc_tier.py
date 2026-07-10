"""Tests for doc-tier resolution (mcp_hwnative_sys.knowledge.resolve_doc_tier)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.knowledge import resolve_doc_tier


def test_canonical_sibling_repo_doc():
    assert resolve_doc_tier("pypto/docs/en/dev/00-ecosystem.md") == "canonical"


def test_enriched_notes():
    assert resolve_doc_tier("pypto-3.0-notes/architecture/PTOAS.md") == "enriched"


def test_ephemeral_pr_plans():
    assert resolve_doc_tier("pypto-3.0-notes/pr_plans/status_prs.md") == "ephemeral"


def test_design_top_level_documents():
    assert resolve_doc_tier("pypto_top_level_documents/sharded_tensor.md") == "design"


def test_mcp_owned_content():
    assert resolve_doc_tier("content/ascend/which_platform.md") == "mcp-owned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
