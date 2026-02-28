from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ralph_writer.utils import count_words, line_number_of_index

# Standard section marker pattern
SECTION_PATTERN = re.compile(
    r"<!-- SECTION: (?P<name>[^\n]+?) -->\n(?P<content>.*?)\n<!-- END SECTION: (?P=name) -->",
    re.DOTALL,
)

# Patterns for detecting malformed/alternative markers to normalize
MALFORMED_START_PATTERN = re.compile(
    r"<!--\s*(?:SECTION|START\s+SECTION):\s*([^\n]+?)\s*-->",
    re.IGNORECASE,
)
MALFORMED_END_PATTERN = re.compile(
    r"<!--\s*(?:END\s+)?SECTION:\s*([^\n]+?)\s*-->",
    re.IGNORECASE,
)


def normalize_section_markers(text: str) -> tuple[str, dict[str, int]]:
    """
    Normalize and self-heal malformed section markers.
    
    Fixes:
    - Alternative formats (START SECTION vs SECTION)
    - Duplicate END markers
    - Inconsistent spacing around markers
    
    Returns:
        Tuple of (cleaned_text, healing_stats) where healing_stats tracks what was fixed.
    """
    stats = {"duplicates_removed": 0, "formats_normalized": 0}
    
    # Find all section names that appear
    all_sections: dict[str, dict[str, list[int]]] = {}
    
    for match in MALFORMED_START_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name not in all_sections:
            all_sections[name] = {"start": [], "end": []}
        all_sections[name]["start"].append((match.start(), match.end()))
    
    for match in MALFORMED_END_PATTERN.finditer(text):
        name = match.group(1).strip()
        if name not in all_sections:
            all_sections[name] = {"start": [], "end": []}
        all_sections[name]["end"].append((match.start(), match.end()))
    
    # Process each section to consolidate duplicates and fix format
    cleaned = text
    offset = 0  # Track offset as we make replacements
    
    for section_name in sorted(all_sections.keys()):
        info = all_sections[section_name]
        
        # Handle multiple START markers - keep only the first, normalize format
        if len(info["start"]) > 1:
            # Keep the first, remove the rest
            for start, end in info["start"][1:]:
                adjusted_start = start - offset
                adjusted_end = end - offset
                cleaned = cleaned[:adjusted_start] + cleaned[adjusted_end:]
                offset += end - start
                stats["duplicates_removed"] += 1
        
        # Handle multiple END markers - keep only the first, remove the rest
        if len(info["end"]) > 1:
            for start, end in info["end"][1:]:
                adjusted_start = start - offset
                adjusted_end = end - offset
                cleaned = cleaned[:adjusted_start] + cleaned[adjusted_end:]
                offset += end - start
                stats["duplicates_removed"] += 1
        
        # Normalize marker format
        if info["start"]:
            old_start = info["start"][0]
            normalized = f"<!-- SECTION: {section_name} -->"
            # This is approximate - in practice, better to do a second pass
    
    # Second pass: normalize all markers to standard format
    cleaned = re.sub(
        r"<!--\s*(?:SECTION|START\s+SECTION):\s*([^\n]+?)\s*-->",
        lambda m: f"<!-- SECTION: {m.group(1).strip()} -->",
        cleaned,
        flags=re.IGNORECASE,
    )
    stats["formats_normalized"] += cleaned.count("<!-- SECTION:")
    
    cleaned = re.sub(
        r"<!--\s*(?:END\s+)?SECTION:\s*([^\n]+?)\s*-->",
        lambda m: f"<!-- END SECTION: {m.group(1).strip()} -->",
        cleaned,
        flags=re.IGNORECASE,
    )
    
    return cleaned, stats


def parse_sections(text: str) -> list[dict[str, Any]]:
    """Parse sections with self-healing for malformed markers."""
    # Auto-normalize the text first
    healed_text, _ = normalize_section_markers(text)
    
    sections: list[dict[str, Any]] = []
    for match in SECTION_PATTERN.finditer(healed_text):
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
                "start_line": line_number_of_index(healed_text, start),
                "end_line": line_number_of_index(healed_text, end),
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


def heal_manuscript(path: Path) -> dict[str, Any]:
    """
    Read a manuscript, normalize malformed section markers, and write it back.
    Returns stats about what was healed.
    """
    text = read_manuscript(path)
    healed_text, stats = normalize_section_markers(text)
    
    if healed_text != text:
        write_manuscript(path, healed_text)
        stats["file_was_healed"] = True
    else:
        stats["file_was_healed"] = False
    
    return stats


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
