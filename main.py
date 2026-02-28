from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from rich.console import Console
from rich.prompt import Prompt

from ralph_writer.config.models import LoadedPhaseConfig, RuntimeSettings
from ralph_writer.core import Project
from ralph_writer.images import extract_image_refs, validate_image_paths
from ralph_writer.manuscript import get_manuscript_info_data
from ralph_writer.orchestrator import SessionOrchestrator, show_stats_table, show_status


PROJECTS_DIR = Path("projects")
CONFIG: dict[str, Any] = {}  # Will be populated from config.yaml


def load_phase_config(config_path: Path = Path("config.yaml")) -> LoadedPhaseConfig:
	"""Load phase configuration from YAML file."""
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

console = Console()


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


def safe_project_name(name: str) -> str:
	"""Sanitize project name."""
	cleaned = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip()
	cleaned = cleaned.replace(" ", "_")
	return cleaned or "untitled_project"


def choose_or_create_project() -> Project:
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



def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(description="Ralph Writer")
	parser.add_argument(
		"--info",
		dest="info_project",
		help="Print status for an existing project and exit",
	)
	return parser.parse_args()


def main() -> None:
	"""Main entry point."""
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
		show_status(project.name, state, manuscript_info, STATE_MACHINE)
		stats = project.get_stats()
		show_stats_table(stats, max_entries=0)
		return

	client = OpenAI(base_url=CONFIG["base_url"], api_key=CONFIG["api_key"])
	project = choose_or_create_project()

	orchestrator = SessionOrchestrator(
		project=project,
		client=client,
		config=CONFIG,
		state_machine=STATE_MACHINE,
		phase_guide=PHASE_GUIDE,
		system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
	)

	orchestrator.run_session()


if __name__ == "__main__":
	main()

