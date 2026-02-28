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
from ralph_writer.core import Project, ProjectState
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
from ralph_writer.tools import execute_tool, get_tool_definitions
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


def choose_or_create_project() -> Project:
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

	if (project_dir / "state.json").exists():
		return Project(project_name, project_dir)

	seed = Prompt.ask("Initial story seed (use #path/to/image.png or #\"path with spaces.png\" for images)")
	project_dir.mkdir(parents=True, exist_ok=True)
	init_state = get_default_state()
	init_state["manuscript_file"] = str(project_dir / "manuscript.md").replace("\\", "/")
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

	return Project.load_or_create(project_name, project_dir, **init_state)


def get_project(project_name: str) -> Project:
	"""Load a project by name."""
	name = safe_project_name(project_name)
	project_dir = PROJECTS_DIR / name
	if (project_dir / "state.json").exists():
		return Project(name, project_dir)
	return None



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
	project: Project,
) -> tuple[dict[str, Any], dict[str, Any]]:
	state = project.state.to_dict()
	manuscript_info = get_manuscript_info_data(project.manuscript_path)
	show_status(project.name, state, manuscript_info)

	start_time = time.time()
	total_in_tokens = 0
	total_out_tokens = 0
	total_tool_calls = 0
	phases_with_tool_calls: list[str] = []
	final_text = ""
	status = "Success"

	system_prompt = build_system_prompt(state, manuscript_info)
	project_dir = project.project_dir
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
				tools=get_tool_definitions(),
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
			result = execute_tool(fn_name, args, state, project.state_path, project.manuscript_path, STATE_MACHINE)
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

	project.state.previous_summary = final_text[: CONFIG["summary_max_chars"]]
	project.state.user_feedback = ""

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
	stats = project.append_stats(loop_row)
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
		project = get_project(args.info_project)
		if not project:
			console.print(f"[red]Project not found:[/red] {args.info_project}")
			PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
			existing = sorted([p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()])
			if existing:
				console.print(f"[cyan]Available projects:[/cyan] {', '.join(existing)}")
			return

		state = project.state.to_dict()
		manuscript_info = get_manuscript_info_data(project.manuscript_path)
		show_status(project.name, state, manuscript_info)

		stats = project.get_stats()
		show_stats_table(stats, max_entries=0) # Show all entries
		return

	client = OpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"])
	project = choose_or_create_project()

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
			state, _ = run_iteration(client, project)
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
			project.state.user_feedback = feedback


if __name__ == "__main__":
	main()
