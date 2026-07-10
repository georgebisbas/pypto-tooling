"""Tests for markdown section extraction (mcp_hwnative_sys.doc_sections)."""

from __future__ import annotations

import pytest

from mcp_hwnative_sys.doc_sections import extract_section, list_sections

_DOC = """# Title

Intro text.

## Setup

Setup body line 1.
Setup body line 2.

### Sub step

Nested body.

## Usage

Usage body.
"""


def test_list_sections_levels_and_titles():
    sections = list_sections(_DOC)
    titles = [(s["level"], s["title"]) for s in sections]
    assert (1, "Title") in titles
    assert (2, "Setup") in titles
    assert (3, "Sub step") in titles
    assert (2, "Usage") in titles


def test_extract_section_stops_at_next_sibling():
    body = extract_section(_DOC, "Setup")
    assert body is not None
    assert "Setup body line 1." in body
    assert "Nested body." in body  # deeper heading is included
    assert "Usage body." not in body  # next sibling-level heading ends it


def test_extract_section_case_insensitive_fuzzy():
    assert extract_section(_DOC, "usage") is not None
    assert extract_section(_DOC, "sub") is not None  # word-overlap fuzzy match


def test_extract_section_missing_returns_none():
    assert extract_section(_DOC, "nonexistent heading") is None


def test_extract_section_empty_hint_returns_none():
    assert extract_section(_DOC, "   ") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
