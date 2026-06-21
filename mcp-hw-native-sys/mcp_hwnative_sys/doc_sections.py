from __future__ import annotations

import re
from typing import Any


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def list_sections(content: str) -> list[dict[str, Any]]:
    """Return markdown headings with level and line number."""
    sections: list[dict[str, Any]] = []
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        title = match.group(2).strip()
        sections.append({"level": level, "title": title, "start": match.start()})
    return sections


def _heading_matches(title: str, section_hint: str) -> bool:
    normalized_title = _normalize_heading(title)
    normalized_hint = _normalize_heading(section_hint)
    if normalized_title == normalized_hint:
        return True
    if normalized_hint in normalized_title or normalized_title in normalized_hint:
        return True
    # Word overlap for fuzzy match
    hint_words = set(normalized_hint.split())
    title_words = set(normalized_title.split())
    if hint_words and hint_words <= title_words:
        return True
    return False


def extract_section(content: str, section_hint: str) -> str | None:
    """Extract markdown body from a heading through the next sibling-or-higher heading."""
    if not section_hint.strip():
        return None

    headings = list(_HEADING_RE.finditer(content))
    if not headings:
        return None

    start_index: int | None = None
    start_level = 0
    for index, match in enumerate(headings):
        title = match.group(2).strip()
        if _heading_matches(title, section_hint):
            start_index = index
            start_level = len(match.group(1))
            break

    if start_index is None:
        return None

    start_pos = headings[start_index].start()
    end_pos = len(content)
    for match in headings[start_index + 1 :]:
        if len(match.group(1)) <= start_level:
            end_pos = match.start()
            break

    return content[start_pos:end_pos].rstrip()


def build_section_toc(content: str, max_entries: int = 40) -> str:
    """Compact table of contents for large docs."""
    lines = ["## Section index", ""]
    for item in list_sections(content)[:max_entries]:
        indent = "  " * (item["level"] - 1)
        lines.append(f"{indent}- {item['title']}")
    return "\n".join(lines)
