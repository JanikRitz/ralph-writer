from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ralph_writer.utils import count_words, line_number_of_index

MARKER_KINDS = ("STORYLINE", "CHAPTER", "BEAT", "SECTION")

_KIND_SYNONYMS = {
    "STORY LINE": "STORYLINE",
    "STORYLINE": "STORYLINE",
    "CHAPTER": "CHAPTER",
    "BEAT": "BEAT",
    "SECTION": "SECTION",
}

# Generic marker regexes (accepts START variant and relaxed spacing/casing).
MALFORMED_START_PATTERN = re.compile(
    r"<!--\s*(?:START\s+)?(?P<kind>STORY\s*LINE|STORYLINE|CHAPTER|BEAT|SECTION):\s*(?P<name>[^\n]+?)\s*-->",
    re.IGNORECASE,
)
MALFORMED_END_PATTERN = re.compile(
    r"<!--\s*END\s+(?P<kind>STORY\s*LINE|STORYLINE|CHAPTER|BEAT|SECTION):\s*(?P<name>[^\n]+?)\s*-->",
    re.IGNORECASE,
)

BLOCK_PATTERN_TEMPLATE = (
    r"<!-- {kind}: (?P<name>[^\n]+?) -->\n"
    r"(?P<content>.*?)\n"
    r"<!-- END {kind}: (?P=name) -->"
)

SECTION_PATTERN = re.compile(BLOCK_PATTERN_TEMPLATE.format(kind="SECTION"), re.DOTALL)
BEAT_PATTERN = re.compile(BLOCK_PATTERN_TEMPLATE.format(kind="BEAT"), re.DOTALL)
CHAPTER_PATTERN = re.compile(BLOCK_PATTERN_TEMPLATE.format(kind="CHAPTER"), re.DOTALL)
STORYLINE_PATTERN = re.compile(BLOCK_PATTERN_TEMPLATE.format(kind="STORYLINE"), re.DOTALL)


def _canonical_kind(kind: str) -> str:
    normalized = re.sub(r"\s+", " ", kind.strip().upper())
    return _KIND_SYNONYMS.get(normalized, "SECTION")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]


def _parse_blocks(pattern: re.Pattern[str], text: str, *, base_offset: int = 0) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        name = match.group("name").strip()
        content = match.group("content").strip("\n")
        start = base_offset + match.start()
        end = base_offset + match.end()
        blocks.append(
            {
                "name": name,
                "content": content,
                "start": start,
                "end": end,
            }
        )
    return blocks


def normalize_section_markers(text: str) -> tuple[str, dict[str, int]]:
    """Normalize and self-heal manuscript markers.

    Supported canonical markers:
    - `STORYLINE`
    - `CHAPTER`
    - `BEAT`
    - `SECTION` (legacy)
    """
    stats = {"duplicates_removed": 0, "formats_normalized": 0, "orphaned_markers_fixed": 0}

    cleaned = text

    # Normalize start markers into canonical forms.
    cleaned, start_replacements = MALFORMED_START_PATTERN.subn(
        lambda m: f"<!-- {_canonical_kind(m.group('kind'))}: {m.group('name').strip()} -->",
        cleaned,
    )
    stats["formats_normalized"] += start_replacements

    # Normalize end markers into canonical forms.
    cleaned, end_replacements = MALFORMED_END_PATTERN.subn(
        lambda m: f"<!-- END {_canonical_kind(m.group('kind'))}: {m.group('name').strip()} -->",
        cleaned,
    )
    stats["formats_normalized"] += end_replacements

    # Heal orphaned END markers by injecting matching START markers.
    orphaned_pattern = re.compile(
        r"<!--\s*END\s+(?P<kind>STORYLINE|CHAPTER|BEAT|SECTION):\s*(?P<name>[^\n]+?)\s*-->"
    )
    for match in list(orphaned_pattern.finditer(cleaned)):
        kind = match.group("kind").strip()
        name = match.group("name").strip()
        text_before = cleaned[: match.start()]
        expected_start = f"<!-- {kind}: {name} -->"
        if expected_start not in text_before:
            end_marker = match.group(0)
            cleaned = cleaned.replace(end_marker, f"{expected_start}\n{end_marker}", 1)
            stats["orphaned_markers_fixed"] += 1

    return cleaned, stats


def parse_story_hierarchy(text: str) -> dict[str, Any]:
    """Parse manuscript into storyline → chapters → beats → paragraphs hierarchy.

    REQUIRES explicit STORYLINE, CHAPTER, and BEAT markers.
    Returns empty structure if proper hierarchy is not present.
    """
    healed_text, _ = normalize_section_markers(text)

    storylines = _parse_blocks(STORYLINE_PATTERN, healed_text)
    used_explicit_hierarchy = bool(storylines)

    if not storylines:
        # No storylines found - return empty structure
        return {
            "storyline_count": 0,
            "chapter_count": 0,
            "beat_count": 0,
            "paragraph_count": 0,
            "storylines": [],
        }

    hierarchy_storylines: list[dict[str, Any]] = []
    chapter_count = 0
    beat_count = 0
    paragraph_count = 0

    for storyline in storylines:
        storyline_content = storyline["content"]
        storyline_content_offset = storyline["start"] + len(
            f"<!-- STORYLINE: {storyline['name']} -->\n"
        ) if used_explicit_hierarchy else storyline["start"]

        chapters = _parse_blocks(CHAPTER_PATTERN, storyline_content, base_offset=storyline_content_offset)
        if not chapters:
            # No chapters found in storyline - skip this storyline
            continue

        hierarchy_chapters: list[dict[str, Any]] = []
        chapter_count += len(chapters)

        for chapter in chapters:
            chapter_content = chapter["content"]
            chapter_content_offset = chapter["start"] + len(
                f"<!-- CHAPTER: {chapter['name']} -->\n"
            ) if "<!-- CHAPTER:" in healed_text[chapter["start"] : chapter["end"]] else chapter["start"]

            beats = _parse_blocks(BEAT_PATTERN, chapter_content, base_offset=chapter_content_offset)
            if not beats:
                # No beats found in chapter - skip this chapter
                continue

            hierarchy_beats: list[dict[str, Any]] = []
            beat_count += len(beats)

            for beat in beats:
                paragraphs = _split_paragraphs(beat["content"])
                paragraph_count += len(paragraphs)
                hierarchy_beats.append(
                    {
                        "name": beat["name"],
                        "words": count_words(beat["content"]),
                        "line_range": [
                            line_number_of_index(healed_text, beat["start"]),
                            line_number_of_index(healed_text, beat["end"]),
                        ],
                        "paragraph_count": len(paragraphs),
                        "paragraphs": paragraphs,
                    }
                )

            hierarchy_chapters.append(
                {
                    "name": chapter["name"],
                    "words": count_words(chapter["content"]),
                    "line_range": [
                        line_number_of_index(healed_text, chapter["start"]),
                        line_number_of_index(healed_text, chapter["end"]),
                    ],
                    "beat_count": len(hierarchy_beats),
                    "beats": hierarchy_beats,
                }
            )

        hierarchy_storylines.append(
            {
                "name": storyline["name"],
                "words": count_words(storyline["content"]),
                "line_range": [
                    line_number_of_index(healed_text, storyline["start"]),
                    line_number_of_index(healed_text, storyline["end"]),
                ],
                "chapter_count": len(hierarchy_chapters),
                "chapters": hierarchy_chapters,
            }
        )

    return {
        "storyline_count": len(hierarchy_storylines),
        "chapter_count": chapter_count,
        "beat_count": beat_count,
        "paragraph_count": paragraph_count,
        "storylines": hierarchy_storylines,
    }


def parse_sections(text: str) -> list[dict[str, Any]]:
    """Parse BEAT blocks as sections for backward compatibility.

    DEPRECATED: This function only exists for legacy tool compatibility.
    New code should use parse_story_hierarchy() instead.
    Only parses BEAT blocks - does NOT create default sections.
    """
    healed_text, _ = normalize_section_markers(text)

    sections: list[dict[str, Any]] = []
    # Only parse BEAT blocks - no fallbacks
    for match in BEAT_PATTERN.finditer(healed_text):
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
    hierarchy = parse_story_hierarchy(text)
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
        "storyline_count": hierarchy["storyline_count"],
        "chapter_count": hierarchy["chapter_count"],
        "beat_count": hierarchy["beat_count"],
        "paragraph_count": hierarchy["paragraph_count"],
        "storylines": hierarchy["storylines"],
    }


def _find_beat_in_text(text: str, storyline_name: str, chapter_name: str, beat_name: str) -> tuple[int, int] | None:
    """Find character positions of a beat within its chapter and storyline. Returns (start, end) or None."""
    # Find storyline boundaries
    sl_pattern = re.compile(
        f"<!-- STORYLINE: {re.escape(storyline_name)} -->\n(.*?)\n<!-- END STORYLINE: {re.escape(storyline_name)} -->",
        re.DOTALL
    )
    sl_match = sl_pattern.search(text)
    if not sl_match:
        return None
    sl_content_start = sl_match.start(1)
    sl_content = sl_match.group(1)
    
    # Find chapter boundaries within storyline content
    ch_pattern = re.compile(
        f"<!-- CHAPTER: {re.escape(chapter_name)} -->\n(.*?)\n<!-- END CHAPTER: {re.escape(chapter_name)} -->",
        re.DOTALL
    )
    ch_match = ch_pattern.search(sl_content)
    if not ch_match:
        return None
    ch_content_start = sl_content_start + ch_match.start(1)
    ch_content = ch_match.group(1)
    
    # Find beat boundaries within chapter content
    bt_pattern = re.compile(
        f"<!-- BEAT: {re.escape(beat_name)} -->\n(.*?)\n<!-- END BEAT: {re.escape(beat_name)} -->",
        re.DOTALL
    )
    bt_match = bt_pattern.search(ch_content)
    if not bt_match:
        return None
    bt_start = ch_content_start + bt_match.start()
    bt_end = ch_content_start + bt_match.end()
    
    return bt_start, bt_end


def add_beat_to_manuscript(text: str, storyline_name: str, chapter_name: str, beat_name: str, content: str) -> str:
    """Add a beat to a chapter, creating storyline/chapter/beat hierarchy as needed."""
    text, _ = normalize_section_markers(text)
    
    # Check if storyline exists
    sl_pattern = re.compile(
        f"<!-- STORYLINE: {re.escape(storyline_name)} -->\n(.*?)\n<!-- END STORYLINE: {re.escape(storyline_name)} -->",
        re.DOTALL
    )
    sl_match = sl_pattern.search(text)
    
    if sl_match:
        sl_end = sl_match.end(1)
        sl_content = sl_match.group(1)
        
        # Check if chapter exists
        ch_pattern = re.compile(
            f"<!-- CHAPTER: {re.escape(chapter_name)} -->\n(.*?)\n<!-- END CHAPTER: {re.escape(chapter_name)} -->",
            re.DOTALL
        )
        ch_match = ch_pattern.search(sl_content)
        
        if ch_match:
            # Chapter exists, insert beat before END CHAPTER
            ch_end_marker = f"<!-- END CHAPTER: {chapter_name} -->"
            ch_end_pos = text.rfind(ch_end_marker, sl_match.start())
            beat_block = f"<!-- BEAT: {beat_name} -->\n{content.strip()}\n<!-- END BEAT: {beat_name} -->\n\n"
            return text[:ch_end_pos] + beat_block + text[ch_end_pos:]
        else:
            # Chapter doesn't exist, create it with the beat
            ch_block = (
                f"<!-- CHAPTER: {chapter_name} -->\n"
                f"<!-- BEAT: {beat_name} -->\n{content.strip()}\n<!-- END BEAT: {beat_name} -->\n"
                f"<!-- END CHAPTER: {chapter_name} -->\n\n"
            )
            sl_end_marker = f"<!-- END STORYLINE: {storyline_name} -->"
            sl_end_pos = text.rfind(sl_end_marker)
            return text[:sl_end_pos] + ch_block + text[sl_end_pos:]
    else:
        # Storyline doesn't exist, create full hierarchy
        sl_block = (
            f"<!-- STORYLINE: {storyline_name} -->\n"
            f"<!-- CHAPTER: {chapter_name} -->\n"
            f"<!-- BEAT: {beat_name} -->\n{content.strip()}\n<!-- END BEAT: {beat_name} -->\n"
            f"<!-- END CHAPTER: {chapter_name} -->\n"
            f"<!-- END STORYLINE: {storyline_name} -->\n"
        )
        if text.strip():
            return text.rstrip("\n") + "\n\n" + sl_block
        return sl_block


def update_beat_content(text: str, storyline_name: str, chapter_name: str, beat_name: str, new_content: str) -> tuple[str, bool]:
    """Replace the content of a beat, preserving markers. Returns (updated_text, success)."""
    text, _ = normalize_section_markers(text)
    positions = _find_beat_in_text(text, storyline_name, chapter_name, beat_name)
    
    if not positions:
        return text, False
    
    start, end = positions
    # Extract just the inner markers to find where content starts/ends
    beat_block = text[start:end]
    # Find the opening marker end and closing marker start
    open_marker = f"<!-- BEAT: {beat_name} -->\n"
    close_marker = f"\n<!-- END BEAT: {beat_name} -->"
    open_end = beat_block.find(open_marker) + len(open_marker)
    close_start = beat_block.rfind(close_marker)
    
    new_beat_block = f"<!-- BEAT: {beat_name} -->\n{new_content.strip()}\n<!-- END BEAT: {beat_name} -->"
    return text[:start] + new_beat_block + text[start + end:], True


def append_to_beat(text: str, storyline_name: str, chapter_name: str, beat_name: str, content_to_add: str) -> tuple[str, bool]:
    """Append content to an existing beat. Returns (updated_text, success)."""
    text, _ = normalize_section_markers(text)
    positions = _find_beat_in_text(text, storyline_name, chapter_name, beat_name)
    
    if not positions:
        return text, False
    
    start, end = positions
    beat_block = text[start:end]
    close_marker = f"\n<!-- END BEAT: {beat_name} -->"
    close_pos = beat_block.rfind(close_marker)
    
    new_beat_block = beat_block[:close_pos] + f"\n\n{content_to_add.strip()}" + beat_block[close_pos:]
    return text[:start] + new_beat_block + text[end:], True
