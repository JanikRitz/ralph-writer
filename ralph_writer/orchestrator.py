"""Orchestration layer for managing iteration and session workflows."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console

from ralph_writer.core import IterationResult, Project
from ralph_writer.images import build_vision_message_blocks
from ralph_writer.manuscript import get_manuscript_info_data
from ralph_writer.mcp_client import MCPToolRegistry, is_mcp_tool
from ralph_writer.tools import execute_tool, get_tool_definitions, resolve_phase_tools
from ralph_writer.utils import (
	estimate_tokens_messages,
	estimate_tokens_text,
	truncate_text,
)


console = Console()


def now_iso() -> str:
	"""Return current time as ISO string."""
	return datetime.now().isoformat(timespec="seconds")


def show_phase_history(phase_history: list[dict[str, Any]], max_entries: int = 10) -> None:
	"""Display phase history table."""
	from rich.table import Table

	if not phase_history:
		console.print("[dim]No phase history recorded yet.[/dim]")
		return

	recent = phase_history[-max_entries:] if max_entries > 0 else phase_history

	table = Table(title="Phase History", show_lines=False)
	table.add_column("#", justify="right")
	table.add_column("Phase")
	table.add_column("Entered")
	table.add_column("Duration")
	table.add_column("Loops", justify="right")
	table.add_column("Exit Reason")

	start_num = len(phase_history) - len(recent) + 1
	for i, entry in enumerate(recent):
		phase = entry.get("phase", "-")
		entered_at = entry.get("entered_at", "-")
		exited_at = entry.get("exited_at")
		loops = entry.get("loops", 0)
		exit_reason = entry.get("exit_reason", "-")

		# Calculate duration
		if entered_at != "-":
			try:
				enter_dt = datetime.fromisoformat(entered_at)
				if exited_at:
					exit_dt = datetime.fromisoformat(exited_at)
					delta = exit_dt - enter_dt
					minutes = int(delta.total_seconds() / 60)
					if minutes > 0:
						duration = f"{minutes}m"
					else:
						duration = f"{int(delta.total_seconds())}s"
				else:
					duration = "[yellow]active[/yellow]"
					exit_reason = "-"
			except Exception:
				duration = "-"
		else:
			duration = "-"

		# Format entered time (show just date and time)
		if entered_at != "-":
			try:
				dt = datetime.fromisoformat(entered_at)
				entered_display = dt.strftime("%m-%d %H:%M")
			except Exception:
				entered_display = entered_at
		else:
			entered_display = entered_at

		# Truncate exit reason if too long
		if exit_reason and len(exit_reason) > 50:
			exit_reason = exit_reason[:47] + "..."

		table.add_row(
			str(start_num + i),
			phase,
			entered_display,
			duration,
			str(loops),
			exit_reason or "-",
		)

	console.print(table)


def compact_json(value: Any, max_chars: int) -> str:
	"""Compact JSON representation with truncation."""
	try:
		serialized = json.dumps(value, ensure_ascii=False)
	except Exception:
		serialized = str(value)
	return truncate_text(serialized, max_chars)


def build_system_prompt(
	state: dict[str, Any],
	manuscript_info: dict[str, Any],
	state_machine: dict[str, dict[str, Any]],
	phase_guide: dict[str, str],
	phase_rules: dict[str, list[str]],
	system_prompt_template: str,
	config: dict[str, Any],
) -> str:
	"""Build system prompt for the API call."""
	phase = state.get("phase", "CHARACTER_CREATION")
	phase_def = state_machine.get(phase, {})
	transitions = phase_def.get("transitions", [])
	phase_desc = phase_def.get("description", "Unknown phase")
	guide = phase_guide.get(phase, "")
	previous_summary = str(state.get("previous_summary", ""))[: config["summary_max_chars"]]
	ai_state_keys = list(state.get("ai_state", {}).keys())

	sections = manuscript_info.get("sections", [])
	section_names = ", ".join(s["name"] for s in sections) if sections else "none"
	keys_text = ", ".join(ai_state_keys) if ai_state_keys else "none"
	transitions_text = ", ".join(transitions) if transitions else "none"

	# Format phase rules
	rules = phase_rules.get(phase, [])
	if rules:
		rules_text = "\n".join(f"  {i+1}. {rule}" for i, rule in enumerate(rules))
	else:
		rules_text = "  None"

	return system_prompt_template.format(
		initial_seed=state.get("initial_seed", ""),
		phase=phase,
		phase_desc=phase_desc,
		transitions_text=transitions_text,
		total_words=manuscript_info.get("total_words", 0),
		section_count=manuscript_info.get("section_count", 0),
		section_names=section_names,
		keys_text=keys_text,
		guide=guide,
		phase_rules=rules_text,
		previous_summary=previous_summary or "none",
	)


def show_status(
	project_name: str,
	state: dict[str, Any],
	manuscript_info: dict[str, Any],
	state_machine: dict[str, dict[str, Any]],
) -> None:
	"""Display current session status."""
	from rich.panel import Panel

	phase = state.get("phase", "CHARACTER_CREATION")
	phase_def = state_machine.get(phase, {"description": "Unknown", "transitions": []})
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


def show_stats_table(stats: dict[str, Any], max_entries: int = 5) -> None:
	"""Display statistics table."""
	from rich.table import Table

	loops = stats.get("loops", [])
	recent = loops[-max_entries:] if max_entries > 0 else loops

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
		# Support both old and new format
		start_phase = row.get("start_phase")
		end_phase = row.get("end_phase")
		phase_transitions = row.get("phase_transitions", [])
		
		if start_phase and end_phase:
			# New format: show start -> end
			if start_phase == end_phase:
				phase_display = start_phase
			else:
				phase_display = f"{start_phase} → {end_phase}"
				# Add transition count if there were intermediate phases
				if len(phase_transitions) > 1:
					phase_display += f" ({len(phase_transitions)} steps)"
		else:
			# Old format fallback: try phases list, then single phase
			phase_list = row.get("phases", [])
			if isinstance(phase_list, list) and phase_list:
				phase_display = " → ".join(str(phase) for phase in phase_list if str(phase).strip())
			else:
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
	"""Build user message with optional vision content blocks."""
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


class SessionOrchestrator:
	"""Manages iteration and session workflows for a project."""

	def __init__(
		self,
		project: Project,
		client: OpenAI,
		config: dict[str, Any],
		state_machine: dict[str, dict[str, Any]],
		phase_guide: dict[str, str],
		phase_rules: dict[str, list[str]],
		tool_groups: dict[str, list[str]],
		mcp_servers: dict[str, dict[str, Any]],
		system_prompt_template: str,
		mcp_registry: MCPToolRegistry | None = None,
	):
		"""Initialize orchestrator.
		
		Args:
			project: Project instance
			client: OpenAI client
			config: Runtime configuration dict
			state_machine: Phase state machine definitions
			phase_guide: Phase guide strings
			phase_rules: Phase rules (soft constraints per phase)
			tool_groups: Tool group name → tool name list mapping
			mcp_servers: MCP server configurations
			system_prompt_template: System prompt template string
			mcp_registry: Optional MCP tool registry
		"""
		self.project = project
		self.client = client
		self.config = config
		self.state_machine = state_machine
		self.phase_guide = phase_guide
		self.phase_rules = phase_rules
		self.tool_groups = tool_groups
		self.mcp_servers = mcp_servers
		self.system_prompt_template = system_prompt_template
		self.mcp_registry = mcp_registry

	def should_continue(self, state: dict[str, Any]) -> bool:
		"""Check if session should continue based on current state."""
		phase = state.get("phase")
		if self.config.get("stop_only_on_complete"):
			return phase != "READY_FOR_HUMAN"
		return phase != "READY_FOR_HUMAN"

	def run_iteration(self) -> IterationResult:
		"""Execute a single iteration of the AI workflow.
		
		Returns:
			IterationResult with state, loop_row, status, and token counts
		"""
		state = self.project.state.to_dict()
		manuscript_info = get_manuscript_info_data(self.project.manuscript_path)
		show_status(self.project.name, state, manuscript_info, self.state_machine)
		
		# Show recent phase history for context
		phase_history = state.get("phase_history", [])
		if phase_history:
			show_phase_history(phase_history, max_entries=5)

		start_time = time.time()
		total_in_tokens = 0
		total_out_tokens = 0
		total_tool_calls = 0
		initial_phase = state.get("phase", "CHARACTER_CREATION")
		phase_transitions: list[str] = []
		phase_changed = False
		final_text = ""
		status = "Success"

		system_prompt = build_system_prompt(
			state,
			manuscript_info,
			self.state_machine,
			self.phase_guide,
			self.phase_rules,
			self.system_prompt_template,
			self.config,
		)
		project_dir = self.project.project_dir
		user_content_blocks = build_user_message(state, project_dir)

		# Resolve which tools are allowed for the current phase
		allowed_tools = resolve_phase_tools(
			initial_phase, self.state_machine, self.tool_groups
		)
		if allowed_tools is not None:
			console.print(
				f"[dim]Phase tools ({len(allowed_tools)}): "
				f"{', '.join(sorted(allowed_tools))}[/dim]"
			)

		# Merge MCP tool definitions if an MCP registry is available
		phase_tools_config = self.state_machine.get(initial_phase, {}).get("tools", {})
		mcp_server_names: list[str] = phase_tools_config.get("mcp", []) if phase_tools_config else []
		mcp_tool_defs: list[dict[str, Any]] = []
		if self.mcp_registry and mcp_server_names:
			mcp_tool_defs = self.mcp_registry.get_all_definitions(mcp_server_names)
			if mcp_tool_defs:
				console.print(
					f"[dim]MCP tools ({len(mcp_tool_defs)}) from: "
					f"{', '.join(mcp_server_names)}[/dim]"
				)

		# Show feedback if images are being provided
		image_paths = state.get("image_paths", [])
		if image_paths:
			console.print(f"[cyan]Providing {len(image_paths)} image(s) as context[/cyan]")

		messages: list[dict[str, Any]] = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_content_blocks},
		]

		while True:
			if self.config["max_context_tokens"] > 0:
				current_context_tokens = estimate_tokens_messages(self.config["model"], messages)
				if current_context_tokens >= self.config["max_context_tokens"]:
					status = (
						f"Stopped early: context limit reached "
						f"({current_context_tokens}/{self.config['max_context_tokens']} tokens)"
					)
					console.print(f"[yellow]{status}[/yellow]")
					break

			if self.config["stream_console_updates"]:
				console.print("[dim]Thinking...[/dim]")

			streamed_tool_calls: dict[int, dict[str, str]] = {}
			streamed_tool_names_announced: set[int] = set()

			try:
				# Build tool list: filtered built-in tools + MCP tools
				combined_tools = get_tool_definitions(allowed_tools) + mcp_tool_defs
				with self.client.chat.completions.stream(
					model=self.config["model"],
					messages=messages,
					tools=combined_tools,
					tool_choice="auto",
				) as stream:
					if self.config["stream_console_updates"]:
						console.print("[magenta]Assistant:[/magenta] ", end="")

					for event in stream:
						event_type = getattr(event, "type", "")

						if event_type == "content.delta":
							delta = getattr(event, "delta", "") or ""
							if delta:
								final_text += delta
								if self.config["stream_console_updates"]:
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
								self.config["stream_console_updates"]
								and entry["name"]
								and index not in streamed_tool_names_announced
							):
								console.print(f"\n[cyan]Tool planned:[/cyan] {entry['name']}")
								streamed_tool_names_announced.add(index)

					completion = stream.get_final_completion()
			except Exception as exc:
				status = f"Failed: {exc}"
				if self.config["stream_console_updates"]:
					console.print()
				break

			if self.config["stream_console_updates"]:
				console.print()

			# Note: streaming responses don't reliably include usage data from the API
			# Instead, we estimate tokens from the request and response content
			usage = completion.usage
			if usage:
				total_in_tokens += int(usage.prompt_tokens or 0)
				total_out_tokens += int(usage.completion_tokens or 0)
			else:
				# Fallback: estimate tokens when usage data is unavailable (common with streaming)
				# Estimate input tokens from messages excluding the current response
				estimated_in = estimate_tokens_messages(self.config["model"], messages)
				# Estimate output tokens from the assistant's response
				estimated_out = estimate_tokens_text(self.config["model"], final_text or "(no response)")
				total_in_tokens += estimated_in
				total_out_tokens += estimated_out
				if self.config["stream_console_updates"]:
					console.print(f"[dim]Token estimates (API usage unavailable): in={estimated_in}, out={estimated_out}[/dim]")

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
			for tool_call in normalized_tool_calls:
				if self.config["max_tool_calls_per_iteration"] > 0 and total_tool_calls >= self.config["max_tool_calls_per_iteration"]:
					status = (
						f"Stopped early: tool call limit reached "
						f"({total_tool_calls}/{self.config['max_tool_calls_per_iteration']})"
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
				if not phase_transitions or phase_transitions[-1] != phase_before_call:
					phase_transitions.append(phase_before_call)

				# Route MCP tool calls to the MCP registry
				if is_mcp_tool(fn_name) and self.mcp_registry:
					result = self.mcp_registry.route_tool_call(fn_name, args)
				else:
					result = execute_tool(fn_name, args, state, self.project.state_path, self.project.manuscript_path, self.state_machine, allowed_tools)
				phase_after_call = state.get("phase", "CHARACTER_CREATION")

				# Detect phase change from any tool call in the batch
				if phase_after_call != phase_before_call:
					phase_changed = True

				result_text = json.dumps(result, ensure_ascii=False)
				args_preview = compact_json(args if args else raw_args, self.config["tool_args_max_chars"])
				result_preview = compact_json(result, self.config["tool_result_max_chars"])

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

			# Stop iteration on tool limit or phase change
			if stop_due_to_tool_limit:
				break

			if phase_changed and self.config["stop_after_phase_change"]:
				console.print(
					f"[yellow]Phase changed → {state.get('phase')}. "
					f"Ending iteration to start fresh with new phase context.[/yellow]"
				)
				break

		# Reload state from disk so ProjectState reflects changes made by execute_tool
		# (execute_tool writes directly to state.json, bypassing ProjectState._data)
		self.project.state.load()
		self.project.state.user_feedback = ""

		final_phase = state.get("phase", "UNKNOWN")
		
		# Update phase history with loop count
		phase_history = state.get("phase_history", [])
		if not phase_history:
			# Initialize with current phase if empty
			phase_history = [{
				"phase": final_phase,
				"entered_at": now_iso(),
				"exited_at": None,
				"exit_reason": None,
				"loops": 1,
			}]
			state["phase_history"] = phase_history
			self.project.state.update({"phase_history": phase_history})
		elif phase_history[-1].get("phase") == final_phase:
			# Increment loop count for current phase
			phase_history[-1]["loops"] = phase_history[-1].get("loops", 0) + 1
			self.project.state.update({"phase_history": phase_history})
		
		loop_row = {
			"timestamp": now_iso(),
			"start_phase": initial_phase,
			"end_phase": final_phase,
			"phase_transitions": phase_transitions,
			"status": status,
			"in_tokens": total_in_tokens,
			"out_tokens": total_out_tokens,
			"duration_seconds": round(time.time() - start_time, 3),
			"tool_calls": total_tool_calls,
		}
		stats = self.project.append_stats(loop_row)
		show_stats_table(stats)

		if status.startswith("Failed"):
			time.sleep(3)

		return IterationResult(
			state=state,
			loop_row=loop_row,
			status=status,
			total_tokens=total_in_tokens + total_out_tokens,
			phase_changed=phase_changed,
		)

	def run_session(self) -> None:
		"""Run the main session loop until completion or user interrupt."""
		from rich.prompt import Prompt
		from rich.panel import Panel

		console.print(
			Panel(
				f"Model: {self.config['model']}\nBase URL: {self.config['base_url']}\n"
				f"Auto-pilot: {self.config['auto_pilot']}\n"
				f"Stop after phase change: {self.config['stop_after_phase_change']}",
				title="Session Config",
				border_style="green",
			)
		)

		while True:
			try:
				result = self.run_iteration()
			except KeyboardInterrupt:
				console.print("\n[yellow]Interrupted by user.[/yellow]")
				break
			except Exception as exc:
				console.print(f"[red]Fatal iteration error:[/red] {exc}")
				time.sleep(3)
				continue

			if not self.should_continue(result.state):
				console.print("[bold green]Reached READY_FOR_HUMAN. Session complete.[/bold green]")
				break

			# Check if phase changed and stop_after_phase_change is enabled
			if self.config["stop_after_phase_change"] and result.phase_changed:
				if not self.config["auto_pilot"]:
					feedback = Prompt.ask(
						"Optional feedback before continuing with new phase",
						default=""
					).strip()
					if feedback:
						self.project.state.user_feedback = feedback
				# Continue to next iteration with new phase context
				time.sleep(1)
				continue
			time.sleep(2)
			continue

		feedback = Prompt.ask("Optional feedback (blank to continue)", default="").strip()
		if feedback:
			self.project.state.user_feedback = feedback
