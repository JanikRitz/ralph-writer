from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ralph_writer.utils import count_words, line_number_of_index

# Standard section marker pattern (matches normalized format with SECTION name)
SECTION_PATTERN = re.compile(
    r"<!-- SECTION: (?P<name>[^\n]+?) -->\n(?P<content>.*?)\n<!-- END SECTION: (?P=name) -->",
    re.DOTALL,
)

# Patterns for detecting malformed/alternative markers to normalize
# Matches: <!-- SECTION: name -->, <!-- START SECTION: name -->, <!-- CHAPTER: name -->, <!-- START CHAPTER: name -->, etc.
MALFORMED_START_PATTERN = re.compile(
    r"<!--\s*(?:SECTION|START\s+(?:SECTION|CHAPTER)|CHAPTER):\s*([^\n]+?)\s*-->",
    re.IGNORECASE,
)
# Matches END markers: <!-- END SECTION: name -->, <!-- END CHAPTER: name -->, etc.
# REQUIRES "END" to be present (not optional) to avoid matching opening markers
MALFORMED_END_PATTERN = re.compile(
    r"<!--\s*END\s+(?:SECTION|CHAPTER):\s*([^\n]+?)\s*-->",
    re.IGNORECASE,
)


def normalize_section_markers(text: str) -> tuple[str, dict[str, int]]:
    """
    Normalize and self-heal malformed section markers.
    
    Fixes:
    - Alternative formats (START SECTION vs SECTION, CHAPTER vs SECTION)
    - Orphaned END markers (reconstructs missing START markers before them)
    - Duplicate markers
    - Inconsistent spacing around markers
    
    Returns:
        Tuple of (cleaned_text, healing_stats) where healing_stats tracks what was fixed.
    """
    stats = {"duplicates_removed": 0, "formats_normalized": 0, "orphaned_markers_fixed": 0}
    
    cleaned = text
    
    # First: Fix orphaned END markers by prepending their matching START markers
    # Pattern: Find END SECTION/CHAPTER markers that aren't preceded by a START marker
    def fix_orphaned_end_marker(match):
        end_marker = match.group(0)
        section_name = match.group(1)
        # Extract what comes before this END marker
        start_pos = match.start()
        text_before = cleaned[:start_pos]
        
        # Check if there's a corresponding START marker before this END
        # Look for the last section start in the text before
        start_pattern = f"<!-- SECTION: {re.escape(section_name)} -->"
        if start_pattern not in text_before:
            # This is an orphaned END marker - need to prepend a START
            stats["orphaned_markers_fixed"] += 1
            return f"<!-- SECTION: {section_name} -->\n{end_marker}"
        return end_marker
    
    # Find all END markers and check if they're orphaned
    orphaned_pattern = r"<!--\s*END\s+(?:SECTION|CHAPTER):\s*([^\n]+?)\s*-->"
    
    # We need to process this carefully to detect orphaned markers
    # Simple approach: look for END markers that aren't preceded by START
    fixed_cleaned = cleaned
    for match in re.finditer(orphaned_pattern, cleaned, flags=re.IGNORECASE):
        section_name = match.group(1).strip()
        start_pos = match.start()
        text_before = cleaned[:start_pos]
        
        # Check if there's a corresponding START marker before this END
        # Look for either:
        # - <!-- SECTION: name --> or
        # - <!-- START SECTION: name --> or
        # - <!-- CHAPTER: name --> or
        # - <!-- START CHAPTER: name -->
        start_markers = [
            f"<!-- SECTION: {section_name} -->",
            f"<!-- START SECTION: {section_name} -->",
            f"<!-- CHAPTER: {section_name} -->",
            f"<!-- START CHAPTER: {section_name} -->",
        ]
        
        found_start = any(marker in text_before for marker in start_markers)
        
        if not found_start:
            # This is an orphaned END marker - prepend a START
            end_marker_str = match.group(0)
            fixed_cleaned = fixed_cleaned.replace(
                end_marker_str, 
                f"<!-- SECTION: {section_name} -->\n{end_marker_str}",
                1  # Only replace the first occurrence
            )
            stats["orphaned_markers_fixed"] += 1
    
    cleaned = fixed_cleaned
    
    # Second pass: normalize all markers to standard format
    cleaned = re.sub(
        r"<!--\s*(?:SECTION|START\s+(?:SECTION|CHAPTER)|CHAPTER):\s*([^\n]+?)\s*-->",
        lambda m: f"<!-- SECTION: {m.group(1).strip()} -->",
        cleaned,
        flags=re.IGNORECASE,
    )
    stats["formats_normalized"] += cleaned.count("<!-- SECTION:")
    
    cleaned = re.sub(
        r"<!--\s*END\s+(?:SECTION|CHAPTER):\s*([^\n]+?)\s*-->",
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
