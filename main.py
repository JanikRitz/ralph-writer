from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table


PROJECTS_DIR = Path("projects")
CONFIG: dict[str, Any] = {
	"base_url": os.getenv("RALPH_BASE_URL", "http://192.168.0.31:1234/v1"),
	"model": os.getenv("RALPH_MODEL", "qwen3.5-27b-heretic"),
	"api_key": os.getenv("OPENAI_API_KEY", "lm-studio"),
	"auto_pilot": os.getenv("RALPH_AUTO_PILOT", "true").lower() == "true",
	"stop_only_on_complete": os.getenv("RALPH_STOP_ONLY_ON_COMPLETE", "true").lower() == "true",
	"max_tool_turns": int(os.getenv("RALPH_MAX_TOOL_TURNS", "10")),
	"max_context_tokens": int(os.getenv("RALPH_MAX_CONTEXT_TOKENS", "16000")),
	"max_tool_calls_per_iteration": int(os.getenv("RALPH_MAX_TOOL_CALLS_PER_ITERATION", "0")),
	"summary_max_chars": 800,
}


def load_phase_config(config_path: Path = Path("config.yaml")) -> tuple[dict[str, Any], dict[str, str], str, str]:
	"""Load phase configuration from YAML file.
	
	Returns:
		tuple: (STATE_MACHINE dict, PHASE_GUIDE dict, default_phase string, SYSTEM_PROMPT_TEMPLATE string)
	"""
	if not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_path}")
	
	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle)
	
	default_phase = config.get("default_phase", "CHARACTER_CREATION")
	system_prompt_template = config.get("system_prompt", "")
	phases_data = config.get("phases", {})
	
	state_machine: dict[str, dict[str, Any]] = {}
	phase_guide: dict[str, str] = {}
	
	for phase_name, phase_config in phases_data.items():
		state_machine[phase_name] = {
			"description": phase_config.get("description", ""),
			"transitions": phase_config.get("transitions", []),
		}
		phase_guide[phase_name] = phase_config.get("guide", "")
	
	return state_machine, phase_guide, default_phase, system_prompt_template


# Load phase configuration
STATE_MACHINE, PHASE_GUIDE, DEFAULT_PHASE, SYSTEM_PROMPT_TEMPLATE = load_phase_config()


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
	}

SECTION_PATTERN = re.compile(
	r"<!-- SECTION: (?P<name>[^\n]+?) -->\n(?P<content>.*?)\n<!-- END SECTION: (?P=name) -->",
	re.DOTALL,
)

console = Console()


def now_iso() -> str:
	return datetime.now().isoformat(timespec="seconds")


def safe_project_name(name: str) -> str:
	cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
	cleaned = cleaned.replace(" ", "_")
	return cleaned or "untitled_project"


def read_json(path: Path, default: Any) -> Any:
	if not path.exists():
		return default
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def write_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=False, indent=2)


def count_words(text: str) -> int:
	return len(re.findall(r"\S+", text))


def line_number_of_index(text: str, index: int) -> int:
	return text[:index].count("\n") + 1


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

	seed = Prompt.ask("Initial story seed")
	project_dir.mkdir(parents=True, exist_ok=True)
	init_state = get_default_state()
	init_state["manuscript_file"] = str(manuscript_path).replace("\\", "/")
	init_state["initial_seed"] = seed
	init_state["user_feedback"] = seed
	write_json(state_path, init_state)
	write_json(
		stats_path,
		{"loops": [], "total_input_tokens": 0, "total_output_tokens": 0, "total_time_seconds": 0.0, "total_tool_calls": 0},
	)
	write_manuscript(manuscript_path, "")

	return project_name, state_path, stats_path, manuscript_path


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


def estimate_tokens_text(model: str, text: str) -> int:
	try:
		encoding = tiktoken.encoding_for_model(model)
	except Exception:
		encoding = tiktoken.get_encoding("cl100k_base")
	return len(encoding.encode(text))


def estimate_tokens_messages(model: str, messages: list[dict[str, Any]]) -> int:
	flattened = "\n".join(json.dumps(m, ensure_ascii=False, default=str) for m in messages)
	return estimate_tokens_text(model, flattened)


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


def show_stats_table(stats: dict[str, Any]) -> None:
	loops = stats.get("loops", [])
	recent = loops[-5:]

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
		table.add_row(
			str(start_num + i),
			str(row.get("phase", "")),
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

def build_user_message(state: dict[str, Any]) -> str:
	seed = str(state.get("initial_seed", "")).strip()
	feedback = str(state.get("user_feedback", "")).strip()
	if feedback:
		return f"Initial seed:\n{seed}\n\nLatest user feedback:\n{feedback}"
	return f"Initial seed:\n{seed}\n\nNo additional user feedback. Continue autonomously."


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
	final_text = ""
	status = "Success"

	system_prompt = build_system_prompt(state, manuscript_info)
	user_message = build_user_message(state)
	messages: list[dict[str, Any]] = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_message},
	]

	for _ in range(CONFIG["max_tool_turns"]):
		if CONFIG["max_context_tokens"] > 0:
			current_context_tokens = estimate_tokens_messages(CONFIG["model"], messages)
			if current_context_tokens >= CONFIG["max_context_tokens"]:
				status = (
					f"Stopped early: context limit reached "
					f"({current_context_tokens}/{CONFIG['max_context_tokens']} tokens)"
				)
				console.print(f"[yellow]{status}[/yellow]")
				break

		try:
			response = client.chat.completions.create(
				model=CONFIG["model"],
				messages=messages,
				tools=tool_definitions(),
				tool_choice="auto",
			)
		except Exception as exc:
			status = f"Failed: {exc}"
			break

		usage = response.usage
		if usage:
			total_in_tokens += int(usage.prompt_tokens or 0)
			total_out_tokens += int(usage.completion_tokens or 0)

		message = response.choices[0].message
		assistant_text = message.content or ""
		if assistant_text:
			final_text = assistant_text
			console.print(Panel(assistant_text, title="Assistant", border_style="magenta"))

		assistant_payload: dict[str, Any] = {
			"role": "assistant",
			"content": assistant_text,
		}

		if message.tool_calls:
			assistant_payload["tool_calls"] = [
				{
					"id": tc.id,
					"type": tc.type,
					"function": {
						"name": tc.function.name,
						"arguments": tc.function.arguments,
					},
				}
				for tc in message.tool_calls
			]

		messages.append(assistant_payload)

		if not message.tool_calls:
			break

		stop_due_to_tool_limit = False
		for tool_call in message.tool_calls:
			if CONFIG["max_tool_calls_per_iteration"] > 0 and total_tool_calls >= CONFIG["max_tool_calls_per_iteration"]:
				status = (
					f"Stopped early: tool call limit reached "
					f"({total_tool_calls}/{CONFIG['max_tool_calls_per_iteration']})"
				)
				console.print(f"[yellow]{status}[/yellow]")
				stop_due_to_tool_limit = True
				break

			total_tool_calls += 1
			fn_name = tool_call.function.name
			raw_args = tool_call.function.arguments or "{}"
			try:
				args = json.loads(raw_args)
				if not isinstance(args, dict):
					args = {}
			except Exception:
				args = {}

			result = execute_function(fn_name, args, state, state_path, manuscript_path)
			result_text = json.dumps(result, ensure_ascii=False)

			console.print(
				Panel(
					f"[bold]Tool:[/bold] {fn_name}\n[bold]Args:[/bold] {json.dumps(args, ensure_ascii=False)}\n"
					f"[bold]Result:[/bold] {result_text}",
					title="Tool Call",
					border_style="cyan",
				)
			)

			messages.append(
				{
					"role": "tool",
					"tool_call_id": tool_call.id,
					"name": fn_name,
					"content": result_text,
				}
			)

		if stop_due_to_tool_limit:
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


def main() -> None:
	client = OpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"])
	project_name, state_path, stats_path, manuscript_path = choose_or_create_project()

	console.print(
		Panel(
			f"Model: {CONFIG['model']}\nBase URL: {CONFIG['base_url']}\n"
			f"Auto-pilot: {CONFIG['auto_pilot']}",
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
