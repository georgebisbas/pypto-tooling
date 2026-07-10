"""Tests for status_prs.md parsing (mcp_hwnative_sys.program_status)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.program_status import parse_status_prs_markdown

_STATUS_MD = """# Program status

<!-- SYNC:ACTIVE_START -->
### pypto
| PR | Title | Branch | Author | CI | Notes |
|----|-------|--------|--------|----|-------|
| [#1942](http://x/1942) | Ring allreduce | `feat/ring` | alice | green | ready |
| [#1900](http://x/1900) | Cross-core skew | `feat/skew` | bob | red | wip |
### simpler
| [#42](http://x/42) | Runtime fix | `fix/rt` | carol | green | ok |
<!-- SYNC:ACTIVE_END -->
"""


def test_parses_open_prs_with_repo_attribution():
    data = parse_status_prs_markdown(_STATUS_MD)
    prs = {row["pr"]: row for row in data["open_prs"]}
    assert set(prs) == {"#1942", "#1900", "#42"}
    assert prs["#1942"]["repo"] == "pypto"
    assert prs["#1942"]["branch"] == "feat/ring"
    assert prs["#42"]["repo"] == "simpler"


def test_header_and_separator_rows_skipped():
    data = parse_status_prs_markdown(_STATUS_MD)
    # Exactly the three data rows, no "PR"/"----" header noise.
    assert len(data["open_prs"]) == 3


def test_empty_document_yields_defaults():
    data = parse_status_prs_markdown("# empty\n")
    assert data["open_prs"] == []
    assert data["dashboard"] == {"open": 0, "merged_all": 0, "merged_90d": 0, "closed": 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
