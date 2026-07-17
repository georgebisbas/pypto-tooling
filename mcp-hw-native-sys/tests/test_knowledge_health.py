"""Tests for the knowledge_health tool (mcp_hwnative_sys.knowledge)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.knowledge import knowledge_health_impl


def test_pass_count_key_is_pypto_scoped():
    # knowledge_health used to surface a generic-looking "pass_count" that was
    # actually always pypto's number (no other repo's pass pipeline is ever
    # scraped) — must be explicitly labeled instead.
    result = knowledge_health_impl()
    assert "pypto_pass_count" in result
    assert "pass_count" not in result
    assert "pypto_passes_index_warning" in result
    assert "passes_index_warning" not in result


def test_reports_pto_isa_and_ptoas_coverage():
    # Real workspace has the generated indices built (140/499 entries) --
    # this must self-report nonzero counts rather than requiring a manual
    # audit to notice the abstraction index barely covers either repo.
    result = knowledge_health_impl()
    assert "coverage" in result
    assert result["coverage"]["pto_isa_indexed"] > 0
    assert result["coverage"]["ptoas_indexed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
