from __future__ import annotations

import argparse
import re
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
from ralph_writer.core import Project
from ralph_writer.images import extract_image_refs, validate_image_paths
from ralph_writer.manuscript import get_manuscript_info_data
from ralph_writer.orchestrator import SessionOrchestrator, show_stats_table, show_status


PROJECTS_DIR = Path("projects")
console = Console()


def load_phase_config(config_path: Path) -> LoadedPhaseConfig:
	"""Load phase configuration from YAML file."""
	if not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_path}")

	with config_path.open("r", encoding="utf-8") as handle:
		config = yaml.safe_load(handle) or {}

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


def get_default_state(default_phase: str, phase: str | None = None) -> dict[str, Any]:
	"""Create default state using configured default phase."""
	resolved_phase = phase or default_phase
	return {
		"phase": resolved_phase,
		"manuscript_file": "",
		"initial_seed": "",
		"user_feedback": "",
		"previous_summary": "",
		"ai_state": {},
		"image_paths": [],
	}


def safe_project_name(name: str) -> str:
	"""Sanitize project name."""
	cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
	cleaned = cleaned.replace(" ", "_")
	return cleaned or "untitled_project"


def choose_or_create_project(default_phase: str) -> Project:
	"""Interactively choose or create a project."""
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
	init_state = get_default_state(default_phase)
	init_state["manuscript_file"] = str(project_dir / "manuscript.md").replace("\\", "/")
	init_state["initial_seed"] = seed
	init_state["user_feedback"] = seed

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


def get_project(project_name: str) -> Project | None:
	"""Load a project by name."""
	name = safe_project_name(project_name)
	project_dir = PROJECTS_DIR / name
	if (project_dir / "state.json").exists():
		return Project(name, project_dir)
	return None


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(description="Ralph Writer")
	parser.add_argument(
		"--info",
		dest="info_project",
		help="Print status for an existing project and exit",
	)
	parser.add_argument(
		"--list-project",
		"--list-projects",
		action="store_true",
		dest="list_projects",
		help="List existing projects with compact status and exit",
	)
	parser.add_argument(
		"--config",
		dest="config_path",
		help="Path to config YAML file (default: auto-select from discovered config files)",
	)
	parser.add_argument(
		"--list-configs",
		action="store_true",
		help="List discovered config files and exit",
	)
	parser.add_argument(
		"--show-config",
		dest="show_config_target",
		metavar="CONFIG",
		help="Show one config by name or file path and exit",
	)
	return parser.parse_args()


def discover_config_files(root: Path) -> list[Path]:
	"""Find candidate config files in project root."""
	candidates: set[Path] = set()
	for pattern in ("config*.yaml", "config*.yml", "*config*.yaml", "*config*.yml"):
		for path in root.glob(pattern):
			if path.is_file():
				candidates.add(path)

	default_path = root / "config.yaml"
	if default_path.exists():
		candidates.add(default_path)

	return sorted(candidates, key=lambda p: p.name.lower())


def load_raw_config(config_path: Path) -> dict[str, Any]:
	"""Load raw YAML config for preview/selection."""
	with config_path.open("r", encoding="utf-8") as handle:
		loaded = yaml.safe_load(handle) or {}
	if not isinstance(loaded, dict):
		return {}
	return loaded


def show_config_table(config_paths: list[Path]) -> None:
	"""Display concise table of config file summaries."""
	table = Table(title="Available Config Files", show_lines=False)
	table.add_column("#", justify="right")
	table.add_column("File")
	table.add_column("Description")
	table.add_column("Model")
	table.add_column("Base URL")
	table.add_column("Default Phase")
	table.add_column("Phases", justify="right")

	for index, config_path in enumerate(config_paths, start=1):
		try:
			config = load_raw_config(config_path)
			settings = config.get("settings", {}) if isinstance(config.get("settings", {}), dict) else {}
			description = str(config.get("description", "")).strip() or "-"
			if len(description) > 60:
				description = description[:57].rstrip() + "..."
			model = str(settings.get("model", "")) or "-"
			base_url = str(settings.get("base_url", "")) or "-"
			if len(base_url) > 42:
				base_url = base_url[:39].rstrip() + "..."
			default_phase = str(config.get("default_phase", "")) or "-"
			phase_count = len(config.get("phases", {})) if isinstance(config.get("phases", {}), dict) else 0
		except Exception as exc:
			description = "invalid"
			model = "invalid"
			base_url = "invalid"
			default_phase = f"Error: {exc}"
			phase_count = 0

		table.add_row(
			str(index),
			config_path.name,
			description,
			model,
			base_url,
			default_phase,
			str(phase_count),
		)

	console.print(table)


def show_config_content(config_path: Path, config: dict[str, Any]) -> None:
	"""Display a readable config summary view."""
	settings = config.get("settings", {}) if isinstance(config.get("settings", {}), dict) else {}
	phases = config.get("phases", {}) if isinstance(config.get("phases", {}), dict) else {}
	description = str(config.get("description", "")).strip() or "-"
	version = str(config.get("version", "")).strip() or "-"
	default_phase = str(config.get("default_phase", "")).strip() or "-"

	header = (
		f"[bold]File:[/bold] {config_path}\n"
		f"[bold]Description:[/bold] {description}\n"
		f"[bold]Version:[/bold] {version}\n"
		f"[bold]Default Phase:[/bold] {default_phase}\n"
		f"[bold]Phase Count:[/bold] {len(phases)}"
	)
	console.print(Panel(header, title="Selected Configuration", border_style="blue"))

	settings_table = Table(title="Runtime Settings", show_lines=False)
	settings_table.add_column("Key")
	settings_table.add_column("Value")
	for key, value in settings.items():
		display_value = "***" if "key" in key.lower() else str(value)
		settings_table.add_row(str(key), display_value)
	if settings:
		console.print(settings_table)

	phases_table = Table(title="Phases", show_lines=False)
	phases_table.add_column("Phase")
	phases_table.add_column("Description")
	phases_table.add_column("Transitions", justify="right")
	for phase_name, phase_config in phases.items():
		desc = ""
		transitions_count = 0
		if isinstance(phase_config, dict):
			desc = str(phase_config.get("description", ""))
			transitions = phase_config.get("transitions", [])
			transitions_count = len(transitions) if isinstance(transitions, list) else 0
		if len(desc) > 70:
			desc = desc[:67].rstrip() + "..."
		phases_table.add_row(phase_name, desc or "-", str(transitions_count))
	if phases:
		console.print(phases_table)


def discover_projects(root: Path) -> list[Path]:
	"""Find project directories that contain a state file."""
	projects_dir = root / PROJECTS_DIR
	if not projects_dir.exists():
		return []
	return sorted(
		[
			path
			for path in projects_dir.iterdir()
			if path.is_dir() and (path / "state.json").exists()
		],
		key=lambda p: p.name.lower(),
	)


def format_relative_time(ts: float | None) -> str:
	"""Render file timestamp as concise date/time."""
	if not ts:
		return "-"
	return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def collect_project_summary_row(project: Project) -> list[str]:
	"""Build a compact project summary row."""
	state = project.state.to_dict()
	manuscript_info = get_manuscript_info_data(project.manuscript_path)
	stats = project.get_stats()

	latest_update = max(
		(
			project.state_path.stat().st_mtime if project.state_path.exists() else 0.0,
			project.stats_path.stat().st_mtime if project.stats_path.exists() else 0.0,
			project.manuscript_path.stat().st_mtime if project.manuscript_path.exists() else 0.0,
		),
		default=0.0,
	)

	phase = str(state.get("phase", "-"))
	words = str(manuscript_info.get("total_words", 0))
	sections = str(manuscript_info.get("section_count", 0))
	loops = str(len(stats.get("loops", [])))
	updated = format_relative_time(latest_update)
	return [project.name, phase, words, sections, loops, updated]


def show_project_summary_table(rows: list[list[str]], title: str) -> None:
	"""Render compact project summary table."""
	table = Table(title=title, show_lines=False)
	table.add_column("Project")
	table.add_column("Phase")
	table.add_column("Words", justify="right")
	table.add_column("Sections", justify="right")
	table.add_column("Loops", justify="right")
	table.add_column("Updated")
	for row in rows:
		table.add_row(*row)
	console.print(table)


def list_projects(root: Path) -> None:
	"""Display all existing projects with compact stats."""
	project_dirs = discover_projects(root)
	if not project_dirs:
		console.print("[yellow]No projects found.[/yellow]")
		return

	rows: list[list[str]] = []
	for project_dir in project_dirs:
		project = Project(project_dir.name, project_dir)
		rows.append(collect_project_summary_row(project))

	show_project_summary_table(rows, title="Projects")


def resolve_show_config_target(target: str, root: Path) -> Path:
	"""Resolve --show-config target from file path, file name, or stem."""
	provided = Path(target).expanduser()
	candidates: list[Path] = []

	if provided.is_absolute():
		candidates.append(provided)
	else:
		candidates.append((root / provided).resolve())
		if provided.suffix.lower() not in {".yaml", ".yml"}:
			candidates.append((root / f"{provided}.yaml").resolve())
			candidates.append((root / f"{provided}.yml").resolve())

	for candidate in candidates:
		if candidate.exists() and candidate.is_file():
			return candidate

	discovered = discover_config_files(root)
	normalized = target.strip().lower()
	for path in discovered:
		if path.name.lower() == normalized or path.stem.lower() == normalized:
			return path

	raise FileNotFoundError(f"Config file not found for --show-config: {target}")


def resolve_config_path(args: argparse.Namespace) -> Path | None:
	"""Resolve which config file to use based on args and interactive selection."""
	root = Path.cwd()

	if args.config_path:
		config_path = Path(args.config_path).expanduser()
		if not config_path.is_absolute():
			config_path = (root / config_path).resolve()
		if not config_path.exists():
			raise FileNotFoundError(f"Config file not found: {config_path}")
		return config_path

	config_paths = discover_config_files(root)

	if args.list_configs:
		if not config_paths:
			console.print("[red]No config files found.[/red]")
		else:
			show_config_table(config_paths)
		return None

	if not config_paths:
		raise FileNotFoundError("No config files found. Expected config.yaml or config*.yaml in project root.")

	if len(config_paths) == 1:
		return config_paths[0]

	show_config_table(config_paths)
	if len(config_paths) <= 3:
		console.print("[cyan]Config previews:[/cyan]")
		for path in config_paths:
			try:
				show_config_content(path, load_raw_config(path))
			except Exception as exc:
				console.print(f"[yellow]Could not preview {path.name}: {exc}[/yellow]")

	default_selection = 1
	for index, path in enumerate(config_paths, start=1):
		if path.name.lower() == "config.yaml":
			default_selection = index
			break

	selection_text = Prompt.ask(
		"Choose config file number",
		default=str(default_selection),
	)

	try:
		selection_idx = int(selection_text)
	except ValueError as exc:
		raise ValueError(f"Invalid config selection: {selection_text}") from exc

	if selection_idx < 1 or selection_idx > len(config_paths):
		raise ValueError(f"Config selection out of range: {selection_idx}")

	return config_paths[selection_idx - 1]


def main() -> None:
	"""CLI entry point for Ralph Writer."""
	args = parse_args()
	root = Path.cwd()

	if args.list_projects:
		list_projects(root)
		return

	if args.show_config_target:
		config_path = resolve_show_config_target(args.show_config_target, root)
		show_config_content(config_path, load_raw_config(config_path))
		return

	if args.info_project:
		project = get_project(args.info_project)
		if not project:
			console.print(f"[red]Project not found:[/red] {args.info_project}")
			PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
			existing = sorted([p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()])
			if existing:
				console.print(f"[cyan]Available projects:[/cyan] {', '.join(existing)}")
			return

		show_project_summary_table([collect_project_summary_row(project)], title="Project Summary")

		state = project.state.to_dict()
		manuscript_info = get_manuscript_info_data(project.manuscript_path)
		console.print(f"[bold]Phase:[/bold] {state.get('phase', '-')}")
		console.print(f"[bold]Words:[/bold] {manuscript_info.get('total_words', 0)}")
		console.print(f"[bold]Sections:[/bold] {manuscript_info.get('section_count', 0)}")
		
		# Show phase history if available
		phase_history = state.get("phase_history", [])
		if phase_history:
			from ralph_writer.orchestrator import show_phase_history
			show_phase_history(phase_history, max_entries=0)
		
		stats = project.get_stats()
		show_stats_table(stats, max_entries=0)
		return

	config_path = resolve_config_path(args)
	if config_path is None:
		return

	phase_config = load_phase_config(config_path)
	state_machine = phase_config.state_machine
	phase_guide = phase_config.phase_guide
	default_phase = phase_config.default_phase
	system_prompt_template = phase_config.system_prompt_template
	runtime_config = RuntimeSettings.from_sources(phase_config.settings).to_dict()

	client = OpenAI(base_url=runtime_config["base_url"], api_key=runtime_config["api_key"])
	project = choose_or_create_project(default_phase)

	orchestrator = SessionOrchestrator(
		project=project,
		client=client,
		config=runtime_config,
		state_machine=state_machine,
		phase_guide=phase_guide,
		system_prompt_template=system_prompt_template,
	)

	orchestrator.run_session()
