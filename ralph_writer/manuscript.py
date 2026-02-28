from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ralph_writer.utils import count_words, line_number_of_index

SECTION_PATTERN = re.compile(
    r"<!-- SECTION: (?P<name>[^\n]+?) -->\n(?P<content>.*?)\n<!-- END SECTION: (?P=name) -->",
    re.DOTALL,
)


def parse_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for match in SECTION_PATTERN.finditer(text):
        section_name = match.group("name").strip()
        content = match.group("content").strip("\n")
        start = match.start()
        end = match.end()
        sections.append(
            {
                "name": section_name,
                "content": content,
                "start": start,
                "end": end,
                "start_line": line_number_of_index(text, start),
                "end_line": line_number_of_index(text, end),
                "words": count_words(content),
            }
        )
    return sections


def read_manuscript(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return path.read_text(encoding="utf-8")


def write_manuscript(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def get_manuscript_info_data(manuscript_path: Path) -> dict[str, Any]:
    text = read_manuscript(manuscript_path)
    sections = parse_sections(text)
    lines = text.splitlines()
    return {
        "total_words": count_words(text),
        "total_lines": len(lines),
        "section_count": len(sections),
        "sections": [
            {
                "name": s["name"],
                "words": s["words"],
                "line_range": [s["start_line"], s["end_line"]],
            }
            for s in sections
        ],
    }
