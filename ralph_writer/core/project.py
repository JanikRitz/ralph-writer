"""Project management for Ralph Writer."""

from pathlib import Path
from typing import Any

from ralph_writer.core.state import ProjectState
from ralph_writer.utils import read_json, write_json


class Project:
    """Represents a Ralph Writer project with state, manuscript, and stats."""

    def __init__(self, name: str, project_dir: Path):
        self.name = name
        self.project_dir = project_dir
        self.state_path = project_dir / "state.json"
        self.stats_path = project_dir / "stats.json"
        self.manuscript_path = project_dir / "manuscript.md"
        self.state = ProjectState(self.state_path)

    @classmethod
    def load_or_create(cls, name: str, project_dir: Path, **defaults: Any) -> "Project":
        """Load existing project or create new one with defaults."""
        project_dir.mkdir(parents=True, exist_ok=True)
        project = cls(name, project_dir)

        if not project.state_path.exists():
            project.state.update(defaults)
            write_json(
                project.stats_path,
                {
                    "loops": [],
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_time_seconds": 0.0,
                    "total_tool_calls": 0,
                },
            )

        return project

    def get_stats(self) -> dict[str, Any]:
        """Get current stats."""
        return read_json(
            self.stats_path,
            {
                "loops": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_time_seconds": 0.0,
                "total_tool_calls": 0,
            },
        )

    def append_stats(self, loop_row: dict[str, Any]) -> dict[str, Any]:
        """Append a loop row to stats and update totals."""
        stats = self.get_stats()
        stats["loops"].append(loop_row)
        stats["total_input_tokens"] += int(loop_row.get("in_tokens", 0))
        stats["total_output_tokens"] += int(loop_row.get("out_tokens", 0))
        stats["total_time_seconds"] += float(loop_row.get("duration_seconds", 0.0))
        stats["total_tool_calls"] += int(loop_row.get("tool_calls", 0))
        write_json(self.stats_path, stats)
        return stats
