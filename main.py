from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ralph_writer.config.models import LoadedPhaseConfig, RuntimeSettings
from ralph_writer.images import (
	build_vision_message_blocks,
	extract_image_refs,
	validate_image_paths,
)
from ralph_writer.manuscript import (
	get_manuscript_info_data,
	parse_sections,
	read_manuscript,
	write_manuscript,
)
from ralph_writer.utils import (
	count_words,
	estimate_tokens_messages,
	estimate_tokens_text,
	read_json,
	truncate_text,
	write_json,
)


PROJECTS_DIR = Path("projects")
CONFIG: dict[str, Any] = {}  # Will be populated from config.yaml


def load_phase_config(config_path: Path = Path("config.yaml")) -> LoadedPhaseConfig:
	"""Load phase configuration from YAML file.
	"""
	if not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_path}")
	
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)
	
	default_phase = config.get("default_phase", "CHARACTER_CREATION")
	system_prompt_template = config.get("system_prompt", "")
	phases_data = config.get("phases", {})
	settings = config.get("settings", {})
	
	state_machine: dict[str, dict[str, Any]] = {}
	phase_guide: dict[str, str] = {}
	
	for phase_name, phase_config in phases_data.items():
		state_machine[phase_name] = {
			"description": phase_config.get("description", ""),
			"transitions": phase_config.get("transitions", []),
		}
		phase_guide[phase_name] = phase_config.get("guide", "")

	return LoadedPhaseConfig(
		state_machine=state_machine,
		phase_guide=phase_guide,
		default_phase=default_phase,
		system_prompt_template=system_prompt_template,
		settings=settings,
	)


# Load phase configuration
_PHASE_CONFIG = load_phase_config()
STATE_MACHINE = _PHASE_CONFIG.state_machine
PHASE_GUIDE = _PHASE_CONFIG.phase_guide
DEFAULT_PHASE = _PHASE_CONFIG.default_phase
SYSTEM_PROMPT_TEMPLATE = _PHASE_CONFIG.system_prompt_template

# Apply settings: config.yaml -> ENV variable -> default value
CONFIG.update(RuntimeSettings.from_sources(_PHASE_CONFIG.settings).to_dict())


def get_default_state(phase: str | None = None) -> dict[str, Any]:
	"""Create default state using the configured default phase."""
	if phase is None:
		phase = DEFAULT_PHASE
	return {
		"phase": phase,
		"manuscript_file": "",
		"initial_seed": "",
		"user_feedback": "",
		"previous_summary": "",
		"ai_state": {},
		"image_paths": [],
	}

console = Console()


def now_iso() -> str:
	return datetime.now().isoformat(timespec="seconds")


def safe_project_name(name: str) -> str:
	cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
	cleaned = cleaned.replace(" ", "_")
	return cleaned or "untitled_project"


def choose_or_create_project() -> tuple[str, Path, Path, Path]:
	PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
	existing = sorted([p for p in PROJECTS_DIR.iterdir() if p.is_dir()])

	if existing:
		choices = ", ".join(p.name for p in existing)
		console.print(f"[cyan]Available projects:[/cyan] {choices}")
		selected = Prompt.ask("Project name to open (or new name to create)")
	else:
		selected = Prompt.ask("Project name")

	project_name = safe_project_name(selected)
	project_dir = PROJECTS_DIR / project_name
	state_path = project_dir / "state.json"
	stats_path = project_dir / "stats.json"
	manuscript_path = project_dir / "manuscript.md"

	if state_path.exists():
		return project_name, state_path, stats_path, manuscript_path

	seed = Prompt.ask("Initial story seed (use #path/to/image.png or #\"path with spaces.png\" for images)")
	project_dir.mkdir(parents=True, exist_ok=True)
	init_state = get_default_state()
	init_state["manuscript_file"] = str(manuscript_path).replace("\\", "/")
	init_state["initial_seed"] = seed
	init_state["user_feedback"] = seed
	
	# Extract and validate images from seed
	image_refs = extract_image_refs(seed)
	if image_refs:
		valid_images, errors = validate_image_paths(image_refs, project_dir)
		if errors:
			for err in errors:
				console.print(f"[yellow]Warning: {err}[/yellow]")
		init_state["image_paths"] = valid_images
		if valid_images:
			console.print(f"[green]Loaded {len(valid_images)} image(s) as context:[/green]")
			for img_ref in valid_images:
				img_path = project_dir / img_ref
				if img_path.exists():
					size_kb = img_path.stat().st_size / 1024
					console.print(f"  • {img_ref} ({size_kb:.1f} KB)")
				else:
					console.print(f"  • {img_ref}")
	else:
		init_state["image_paths"] = []
	write_json(state_path, init_state)
	write_json(
		stats_path,
		{"loops": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_time_seconds": 0.0, "total_tool_calls": 0},
	)
	write_manuscript(manuscript_path, "")

	return project_name, state_path, stats_path, manuscript_path


def get_project_paths(project_name: str) -> tuple[str, Path, Path, Path]:
	name = safe_project_name(project_name)
	project_dir = PROJECTS_DIR / name
	state_path = project_dir / "state.json"
	stats_path = project_dir / "stats.json"
	manuscript_path = project_dir / "manuscript.md"
	return name, state_path, stats_path, manuscript_path


def summarize_keys(ai_state: dict[str, Any]) -> dict[str, str]:
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


def compact_json(value: Any, max_chars: int) -> str:
	try:
		serialized = json.dumps(value, ensure_ascii=False)
	except Exception:
		serialized = str(value)
	return truncate_text(serialized, max_chars)


def build_system_prompt(state: dict[str, Any], manuscript_info: dict[str, Any]) -> str:
	phase = state.get("phase", "CHARACTER_CREATION")
	phase_def = STATE_MACHINE.get(phase, {})
	transitions = phase_def.get("transitions", [])
	phase_desc = phase_def.get("description", "Unknown phase")
	guide = PHASE_GUIDE.get(phase, "")
	previous_summary = str(state.get("previous_summary", ""))[: CONFIG["summary_max_chars"]]
	ai_state_keys = list(state.get("ai_state", {}).keys())

	sections = manuscript_info.get("sections", [])
	section_names = ", ".join(s["name"] for s in sections) if sections else "none"
	keys_text = ", ".join(ai_state_keys) if ai_state_keys else "none"
	transitions_text = ", ".join(transitions) if transitions else "none"

	return SYSTEM_PROMPT_TEMPLATE.format(
		initial_seed=state.get("initial_seed", ""),
		phase=phase,
		phase_desc=phase_desc,
		transitions_text=transitions_text,
		total_words=manuscript_info.get("total_words", 0),
		section_count=manuscript_info.get("section_count", 0),
		section_names=section_names,
		keys_text=keys_text,
		guide=guide,
		previous_summary=previous_summary or "none",
	)


def tool_definitions() -> list[dict[str, Any]]:
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


def execute_function(
	name: str,
	arguments: dict[str, Any],
	state: dict[str, Any],
	state_path: Path,
	manuscript_path: Path,
) -> Any:
	ai_state = state.setdefault("ai_state", {})

	def available_sections() -> list[str]:
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

			if new_phase not in STATE_MACHINE:
				return {"error": f"unknown phase '{new_phase}'", "available": sorted(STATE_MACHINE.keys())}

			allowed = STATE_MACHINE.get(current_phase, {}).get("transitions", [])
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


def show_status(project_name: str, state: dict[str, Any], manuscript_info: dict[str, Any]) -> None:
	phase = state.get("phase", "CHARACTER_CREATION")
	phase_def = STATE_MACHINE.get(phase, {"description": "Unknown", "transitions": []})
	transitions = phase_def.get("transitions", [])
	transitions_text = ", ".join(transitions) if transitions else "none"

	status_text = (
		f"Project: [bold]{project_name}[/bold]\n"
		f"Phase: [yellow]{phase}[/yellow]\n"
		f"Description: {phase_def.get('description', '')}\n"
		f"Allowed transitions: {transitions_text}\n"
		f"Manuscript words: {manuscript_info.get('total_words', 0)}"
	)
	console.print(Panel(status_text, title="Ralph Writer Status", border_style="blue"))


def show_stats_table(stats: dict[str, Any], max_entries = 5) -> None:
	loops = stats.get("loops", [])
	recent =  loops[-max_entries:] if max_entries > 0 else loops

	table = Table(title="Recent Loops", show_lines=False)
	table.add_column("#", justify="right")
	table.add_column("Phase")
	table.add_column("Status")
	table.add_column("In")
	table.add_column("Out")
	table.add_column("Tools")
	table.add_column("Seconds", justify="right")

	start_num = len(loops) - len(recent) + 1
	for i, row in enumerate(recent):
		phase_list = row.get("phases", [])
		if isinstance(phase_list, list):
			phase_display = " -> ".join(str(phase) for phase in phase_list if str(phase).strip())
		else:
			phase_display = ""
		if not phase_display:
			phase_display = str(row.get("phase", ""))

		table.add_row(
			str(start_num + i),
			phase_display,
			str(row.get("status", "")),
			str(row.get("in_tokens", 0)),
			str(row.get("out_tokens", 0)),
			str(row.get("tool_calls", 0)),
			f"{row.get('duration_seconds', 0):.2f}",
		)

	console.print(table)
	total_in = stats.get("total_input_tokens", 0)
	total_out = stats.get("total_output_tokens", 0)
	total_time = float(stats.get("total_time_seconds", 0.0))
	total_tools = stats.get("total_tool_calls", 0)
	if total_time > 60:
		time_text = f"{total_time / 60:.1f} min"
	else:
		time_text = f"{total_time:.1f} sec"
	console.print(
		f"[green]Totals:[/green] in={total_in} out={total_out} burn={total_in + total_out} tools={total_tools} time={time_text}"
	)


def append_stats(stats_path: Path, loop_row: dict[str, Any]) -> dict[str, Any]:
	stats = read_json(
		stats_path,
		{"loops": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_time_seconds": 0.0, "total_tool_calls": 0},
	)
	stats["loops"].append(loop_row)
	stats["total_input_tokens"] += int(loop_row.get("in_tokens", 0))
	stats["total_output_tokens"] += int(loop_row.get("out_tokens", 0))
	stats["total_time_seconds"] += float(loop_row.get("duration_seconds", 0.0))
	stats["total_tool_calls"] += int(loop_row.get("tool_calls", 0))
	write_json(stats_path, stats)
	return stats

def build_user_message(state: dict[str, Any], project_dir: Path) -> list[dict[str, Any]]:
	"""Build user message with optional vision content blocks.
	
	Returns list of content blocks for API consumption.
	"""
	seed = str(state.get("initial_seed", "")).strip()
	feedback = str(state.get("user_feedback", "")).strip()
	
	if feedback:
		message_text = f"Initial seed:\n{seed}\n\nLatest user feedback:\n{feedback}"
	else:
		message_text = f"Initial seed:\n{seed}\n\nNo additional user feedback. Continue autonomously."
	
	# Get image paths from state
	image_paths = state.get("image_paths", [])
	
	# Build content blocks
	return build_vision_message_blocks(message_text, image_paths, project_dir)


def run_iteration(
	client: OpenAI,
	project_name: str,
	state_path: Path,
	stats_path: Path,
	manuscript_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
	state = read_json(state_path, get_default_state())
	manuscript_info = get_manuscript_info_data(manuscript_path)
	show_status(project_name, state, manuscript_info)

	start_time = time.time()
	total_in_tokens = 0
	total_out_tokens = 0
	total_tool_calls = 0
	phases_with_tool_calls: list[str] = []
	final_text = ""
	status = "Success"

	system_prompt = build_system_prompt(state, manuscript_info)
	project_dir = Path(state.get("manuscript_file", "")).parent
	user_content_blocks = build_user_message(state, project_dir)
	
	# Show feedback if images are being provided
	image_paths = state.get("image_paths", [])
	if image_paths:
		console.print(f"[cyan]Providing {len(image_paths)} image(s) as context[/cyan]")
	
	messages: list[dict[str, Any]] = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content_blocks},
	]

	while True:
		if CONFIG["max_context_tokens"] > 0:
			current_context_tokens = estimate_tokens_messages(CONFIG["model"], messages)
			if current_context_tokens >= CONFIG["max_context_tokens"]:
				status = (
					f"Stopped early: context limit reached "
					f"({current_context_tokens}/{CONFIG['max_context_tokens']} tokens)"
				)
				console.print(f"[yellow]{status}[/yellow]")
				break

		if CONFIG["stream_console_updates"]:
			console.print("[dim]Thinking...[/dim]")

		streamed_tool_calls: dict[int, dict[str, str]] = {}
		streamed_tool_names_announced: set[int] = set()

		try:
			with client.chat.completions.stream(
				model=CONFIG["model"],
				messages=messages,
				tools=tool_definitions(),
				tool_choice="auto",
			) as stream:
				if CONFIG["stream_console_updates"]:
					console.print("[magenta]Assistant:[/magenta] ", end="")

				for event in stream:
					event_type = getattr(event, "type", "")

					if event_type == "content.delta":
						delta = getattr(event, "delta", "") or ""
						if delta:
							final_text += delta
							if CONFIG["stream_console_updates"]:
								console.print(delta, end="", markup=False, highlight=False)

					elif event_type in {
						"tool_calls.function.arguments.delta",
						"tool_calls.function.arguments.done",
					}:
						index = int(getattr(event, "index", 0) or 0)
						entry = streamed_tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})

						tool_call_id = getattr(event, "tool_call_id", "") or ""
						if tool_call_id:
							entry["id"] = tool_call_id

						name = getattr(event, "name", "") or ""
						if name:
							entry["name"] = name

						delta = getattr(event, "delta", "") or ""
						if delta:
							entry["arguments"] += delta

						if (
							CONFIG["stream_console_updates"]
							and entry["name"]
							and index not in streamed_tool_names_announced
						):
							console.print(f"\n[cyan]Tool planned:[/cyan] {entry['name']}")
							streamed_tool_names_announced.add(index)

				completion = stream.get_final_completion()
		except Exception as exc:
			status = f"Failed: {exc}"
			if CONFIG["stream_console_updates"]:
				console.print()
			break

		if CONFIG["stream_console_updates"]:
			console.print()

		usage = completion.usage
		if usage:
			total_in_tokens += int(usage.prompt_tokens or 0)
			total_out_tokens += int(usage.completion_tokens or 0)

		message = completion.choices[0].message
		assistant_text = message.content or final_text
		if not final_text:
			final_text = assistant_text

		normalized_tool_calls: list[dict[str, str]] = []
		if message.tool_calls:
			for tc in message.tool_calls:
				normalized_tool_calls.append(
					{
						"id": tc.id,
						"name": tc.function.name,
						"arguments": tc.function.arguments or "{}",
					}
				)
		elif streamed_tool_calls:
			for idx in sorted(streamed_tool_calls):
				entry = streamed_tool_calls[idx]
				normalized_tool_calls.append(
					{
						"id": entry.get("id", "") or f"streamed_tool_{idx}",
						"name": entry.get("name", ""),
						"arguments": entry.get("arguments", "") or "{}",
					}
				)

		assistant_payload: dict[str, Any] = {
			"role": "assistant",
			"content": assistant_text,
		}

		if normalized_tool_calls:
			assistant_payload["tool_calls"] = [
				{
					"id": tc["id"],
					"type": "function",
					"function": {
						"name": tc["name"],
						"arguments": tc["arguments"],
					},
				}
				for tc in normalized_tool_calls
			]

		messages.append(assistant_payload)

		if not normalized_tool_calls:
			break

		stop_due_to_tool_limit = False
		stop_due_to_phase_change = False
		for tool_call in normalized_tool_calls:
			if CONFIG["max_tool_calls_per_iteration"] > 0 and total_tool_calls >= CONFIG["max_tool_calls_per_iteration"]:
				status = (
					f"Stopped early: tool call limit reached "
					f"({total_tool_calls}/{CONFIG['max_tool_calls_per_iteration']})"
				)
				console.print(f"[yellow]{status}[/yellow]")
				stop_due_to_tool_limit = True
				break

			total_tool_calls += 1
			fn_name = tool_call["name"]
			raw_args = tool_call["arguments"] or "{}"
			try:
				args = json.loads(raw_args)
				if not isinstance(args, dict):
					args = {}
			except Exception:
				args = {}

			phase_before_call = state.get("phase", "CHARACTER_CREATION")
			if not phases_with_tool_calls or phases_with_tool_calls[-1] != phase_before_call:
				phases_with_tool_calls.append(phase_before_call)
			result = execute_function(fn_name, args, state, state_path, manuscript_path)
			phase_after_call = state.get("phase", "CHARACTER_CREATION")
			result_text = json.dumps(result, ensure_ascii=False)
			args_preview = compact_json(args if args else raw_args, CONFIG["tool_args_max_chars"])
			result_preview = compact_json(result, CONFIG["tool_result_max_chars"])

			console.print(f"[cyan]Tool #{total_tool_calls}:[/cyan] {fn_name}")
			console.print(f"  [bold]args[/bold]: {args_preview}")
			console.print(f"  [bold]result[/bold]: {result_preview}")

			messages.append(
				{
					"role": "tool",
					"tool_call_id": tool_call["id"],
					"name": fn_name,
					"content": result_text,
				}
			)

			if (
				CONFIG["stop_after_phase_change"]
				and fn_name == "change_phase"
				and phase_after_call != phase_before_call
			):
				status = (
					f"Stopped after phase change: {phase_before_call} -> {phase_after_call}"
				)
				console.print(f"[yellow]{status}[/yellow]")
				stop_due_to_phase_change = True
				break

		if stop_due_to_tool_limit:
			break
		if stop_due_to_phase_change:
			break

	if total_in_tokens == 0 and total_out_tokens == 0:
		total_in_tokens = estimate_tokens_messages(CONFIG["model"], messages)
		total_out_tokens = estimate_tokens_text(CONFIG["model"], final_text)

	state["previous_summary"] = final_text[: CONFIG["summary_max_chars"]]
	state["user_feedback"] = ""
	write_json(state_path, state)

	loop_row = {
		"timestamp": now_iso(),
		"phase": state.get("phase", "UNKNOWN"),
		"phases": phases_with_tool_calls,
		"status": status,
		"in_tokens": total_in_tokens,
		"out_tokens": total_out_tokens,
		"duration_seconds": round(time.time() - start_time, 3),
		"tool_calls": total_tool_calls,
	}
	stats = append_stats(stats_path, loop_row)
	show_stats_table(stats)

	if status.startswith("Failed"):
		time.sleep(3)

	return state, loop_row


def should_stop(state: dict[str, Any]) -> bool:
	phase = state.get("phase")
	if CONFIG["stop_only_on_complete"]:
		return phase == "READY_FOR_HUMAN"
	return phase == "READY_FOR_HUMAN"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ralph Writer")
	parser.add_argument(
		"--info",
		dest="info_project",
		help="Print status for an existing project and exit",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.info_project:
		project_name, state_path, stats_path, manuscript_path = get_project_paths(args.info_project)
		if not state_path.exists():
			console.print(f"[red]Project not found:[/red] {project_name}")
			PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
			existing = sorted([p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()])
			if existing:
				console.print(f"[cyan]Available projects:[/cyan] {', '.join(existing)}")
			return

		state = read_json(state_path, get_default_state())
		manuscript_info = get_manuscript_info_data(manuscript_path)
		show_status(project_name, state, manuscript_info)

		if stats_path.exists():
			stats = read_json(
				stats_path,
				{
					"loops": [],
					"total_input_tokens": 0,
					"total_output_tokens": 0,
					"total_time_seconds": 0.0,
					"total_tool_calls": 0,
				},
			)
			show_stats_table(stats, max_entries=0) # Show all entries
		return

	client = OpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"])
	project_name, state_path, stats_path, manuscript_path = choose_or_create_project()

	console.print(
		Panel(
			f"Model: {CONFIG['model']}\nBase URL: {CONFIG['base_url']}\n"
			f"Auto-pilot: {CONFIG['auto_pilot']}\n"
			f"Stop after phase change: {CONFIG['stop_after_phase_change']}",
			title="Session Config",
			border_style="green",
		)
	)

	while True:
		try:
			state, _ = run_iteration(client, project_name, state_path, stats_path, manuscript_path)
		except KeyboardInterrupt:
			console.print("\n[yellow]Interrupted by user.[/yellow]")
			break
		except Exception as exc:
			console.print(f"[red]Fatal iteration error:[/red] {exc}")
			time.sleep(3)
			continue

		if should_stop(state):
			console.print("[bold green]Reached READY_FOR_HUMAN. Session complete.[/bold green]")
			break

		if CONFIG["auto_pilot"]:
			time.sleep(2)
			continue

		feedback = Prompt.ask("Optional feedback (blank to continue)", default="").strip()
		if feedback:
			current_state = read_json(state_path, get_default_state())
			current_state["user_feedback"] = feedback
			write_json(state_path, current_state)


if __name__ == "__main__":
	main()
