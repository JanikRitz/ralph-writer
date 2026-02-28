"""Tool system for Ralph Writer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ralph_writer.manuscript import get_manuscript_info_data, parse_sections, read_manuscript, write_manuscript, heal_manuscript
from ralph_writer.utils import count_words, read_json, write_json


def summarize_keys(ai_state: dict[str, Any]) -> dict[str, str]:
    """Summarize AI state keys with type/size hints."""
    result: dict[str, str] = {}
    for key, value in ai_state.items():
        if isinstance(value, dict):
            result[key] = f"object ({len(value)} keys)"
        elif isinstance(value, list):
            result[key] = f"array ({len(value)} items)"
        elif isinstance(value, str):
            result[key] = f"string ({count_words(value)} words)"
        elif isinstance(value, bool):
            result[key] = "boolean"
        elif isinstance(value, (int, float)):
            result[key] = "number"
        elif value is None:
            result[key] = "null"
        else:
            result[key] = type(value).__name__
    return result


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI-compatible tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "List available notes keys with compact type/size hints. Use this to plan, analyze, track characters, world rules, plot beats, and revision notes.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_notes",
                "description": "Read one notes key value. Use this to retrieve character definitions, world rules, plot outlines, or any previous analysis and planning.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "Notes key"}},
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_notes",
                "description": "Write JSON-serializable data to one notes key. Store character definitions, world rules, plot outlines, beat tracking, revision notes, and all planning/analysis here—never in the manuscript.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Notes key"},
                        "data": {
                            "type": ["object", "array", "string", "number", "boolean", "null"],
                            "description": "JSON value to store",
                        },
                    },
                    "required": ["key", "data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_notes",
                "description": "Delete one key from notes.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "Notes key"}},
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_manuscript_info",
                "description": "Get manuscript TOC, word counts, and section line ranges without full content.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_manuscript_section",
                "description": "Read one named manuscript section's content.",
                "parameters": {
                    "type": "object",
                    "properties": {"section_name": {"type": "string", "description": "Section name"}},
                    "required": ["section_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_manuscript_tail",
                "description": "Read the last N words from manuscript for continuation context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "How many words from the end",
                            "default": 300,
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_manuscript",
                "description": "Search manuscript text and return matching lines with surrounding context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text or regex to find"},
                        "context_lines": {
                            "type": "integer",
                            "description": "Lines of context around each match",
                            "default": 2,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "append_to_manuscript",
                "description": "Append prose to the end of manuscript.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string", "description": "Text to append"}},
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_section",
                "description": "Create a new named section with markers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Section name"},
                        "content": {"type": "string", "description": "Section content"},
                    },
                    "required": ["name", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_section",
                "description": "Replace content of an existing named section while preserving markers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Section name"},
                        "content": {"type": "string", "description": "Replacement content"},
                    },
                    "required": ["name", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_section",
                "description": "Delete a named section and its markers.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Section name"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "change_phase",
                "description": "Transition to a valid next phase according to state machine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_phase": {"type": "string", "description": "Target phase"},
                        "reason": {"type": "string", "description": "Why transition now"},
                    },
                    "required": ["new_phase", "reason"],
                },
            },
        },
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    manuscript_path: Path,
    state_machine: dict[str, dict[str, Any]],
) -> Any:
    """Execute a named tool with given arguments."""
    ai_state = state.setdefault("ai_state", {})

    def available_sections() -> list[str]:
        # Auto-heal manuscript before reading sections
        heal_manuscript(manuscript_path)
        return [s["name"] for s in parse_sections(read_manuscript(manuscript_path))]

    try:
        if name == "list_notes":
            return summarize_keys(ai_state)

        if name == "read_notes":
            key = arguments["key"]
            if key not in ai_state:
                return {"error": f"notes key '{key}' not found", "available": sorted(ai_state.keys())}
            return ai_state[key]

        if name == "write_notes":
            key = arguments["key"]
            ai_state[key] = arguments.get("data")
            write_json(state_path, state)
            return f"wrote notes key '{key}' (persisted to file)"

        if name == "delete_notes":
            key = arguments["key"]
            if key not in ai_state:
                return {"error": f"notes key '{key}' not found", "available": sorted(ai_state.keys())}
            del ai_state[key]
            write_json(state_path, state)
            return f"deleted notes key '{key}' (persisted to file)"

        if name == "get_manuscript_info":
            return get_manuscript_info_data(manuscript_path)

        if name == "read_manuscript_section":
            section_name = arguments["section_name"]
            text = read_manuscript(manuscript_path)
            sections = parse_sections(text)
            for section in sections:
                if section["name"] == section_name:
                    return section["content"]
            return {"error": f"section '{section_name}' not found", "available": [s["name"] for s in sections]}

        if name == "read_manuscript_tail":
            word_count = int(arguments.get("word_count", 300))
            text = read_manuscript(manuscript_path).strip()
            words = text.split()
            tail = " ".join(words[-max(word_count, 1) :])
            return tail

        if name == "search_manuscript":
            query = arguments["query"]
            context_lines = max(0, int(arguments.get("context_lines", 2)))
            text = read_manuscript(manuscript_path)
            lines = text.splitlines()
            matches = []
            for idx, line in enumerate(lines):
                if re.search(query, line, flags=re.IGNORECASE):
                    start = max(0, idx - context_lines)
                    end = min(len(lines), idx + context_lines + 1)
                    snippet = "\n".join(lines[start:end])
                    matches.append({"line": idx + 1, "context": snippet})
            return {"query": query, "matches": matches[:20], "count": len(matches)}

        if name == "append_to_manuscript":
            content = arguments["content"].strip("\n")
            existing = read_manuscript(manuscript_path)
            if existing and not existing.endswith("\n\n"):
                existing = existing.rstrip("\n") + "\n\n"
            updated = existing + content + "\n"
            write_manuscript(manuscript_path, updated)
            return f"appended {count_words(content)} words (total: {count_words(updated)})"

        if name == "create_section":
            section_name = arguments["name"].strip()
            content = arguments["content"].strip("\n")
            # Heal manuscript before reading to ensure consistent state
            heal_manuscript(manuscript_path)
            text = read_manuscript(manuscript_path)
            sections = parse_sections(text)
            if section_name in [s["name"] for s in sections]:
                return {
                    "error": f"section '{section_name}' already exists",
                    "available": [s["name"] for s in sections],
                }
            block = (
                f"<!-- SECTION: {section_name} -->\n"
                f"{content}\n"
                f"<!-- END SECTION: {section_name} -->\n"
            )
            if text and not text.endswith("\n\n"):
                text = text.rstrip("\n") + "\n\n"
            write_manuscript(manuscript_path, text + block)
            return f"created section '{section_name}' ({count_words(content)} words)"

        if name == "replace_section":
            section_name = arguments["name"].strip()
            content = arguments["content"].strip("\n")
            # Heal manuscript before reading to ensure consistent state
            heal_manuscript(manuscript_path)
            text = read_manuscript(manuscript_path)
            sections = parse_sections(text)
            target = next((s for s in sections if s["name"] == section_name), None)
            if not target:
                return {"error": f"section '{section_name}' not found", "available": [s["name"] for s in sections]}
            replacement = (
                f"<!-- SECTION: {section_name} -->\n"
                f"{content}\n"
                f"<!-- END SECTION: {section_name} -->"
            )
            updated = text[: target["start"]] + replacement + text[target["end"] :]
            write_manuscript(manuscript_path, updated)
            return f"replaced section '{section_name}' ({count_words(content)} words)"

        if name == "delete_section":
            section_name = arguments["name"].strip()
            # Heal manuscript before reading to ensure consistent state
            heal_manuscript(manuscript_path)
            text = read_manuscript(manuscript_path)
            sections = parse_sections(text)
            target = next((s for s in sections if s["name"] == section_name), None)
            if not target:
                return {"error": f"section '{section_name}' not found", "available": [s["name"] for s in sections]}
            updated = (text[: target["start"]] + text[target["end"] :]).strip("\n") + "\n"
            write_manuscript(manuscript_path, updated)
            return f"deleted section '{section_name}'"

        if name == "change_phase":
            new_phase = arguments["new_phase"].strip()
            reason = arguments["reason"].strip()
            current_phase = state.get("phase", "CHARACTER_CREATION")

            if new_phase not in state_machine:
                return {"error": f"unknown phase '{new_phase}'", "available": sorted(state_machine.keys())}

            allowed = state_machine.get(current_phase, {}).get("transitions", [])
            if new_phase not in allowed:
                return {
                    "error": f"invalid transition from {current_phase} to {new_phase}",
                    "available": allowed,
                }

            state["phase"] = new_phase
            write_json(state_path, state)
            return f"phase changed from {current_phase} to {new_phase}: {reason} (persisted to file)"

        return {"error": f"unknown function '{name}'"}
    except Exception as exc:
        if name in {"read_manuscript_section", "replace_section", "delete_section"}:
            return {"error": str(exc), "available": available_sections()}
        return {"error": str(exc)}
