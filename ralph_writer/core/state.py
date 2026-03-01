"""Core state management for Ralph Writer."""

from pathlib import Path
from typing import Any

from ralph_writer.utils import read_json, write_json


class ProjectState:
    """Manages project state with persistent storage."""

    def __init__(self, state_path: Path):
        self.path = state_path
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load state from disk."""
        default = {
            "phase": "CHARACTER_CREATION",
            "manuscript_file": "",
            "initial_seed": "",
            "user_feedback": "",
            "previous_summary": "",
            "ai_state": {},
            "image_paths": [],
            "phase_history": [],
        }
        self._data = read_json(self.path, default)

    def save(self) -> None:
        """Persist state to disk."""
        write_json(self.path, self._data)

    @property
    def phase(self) -> str:
        return self._data.get("phase", "CHARACTER_CREATION")

    @phase.setter
    def phase(self, value: str) -> None:
        self._data["phase"] = value
        self.save()

    @property
    def manuscript_file(self) -> str:
        return self._data.get("manuscript_file", "")

    @manuscript_file.setter
    def manuscript_file(self, value: str) -> None:
        self._data["manuscript_file"] = value
        self.save()

    @property
    def initial_seed(self) -> str:
        return self._data.get("initial_seed", "")

    @initial_seed.setter
    def initial_seed(self, value: str) -> None:
        self._data["initial_seed"] = value
        self.save()

    @property
    def user_feedback(self) -> str:
        return self._data.get("user_feedback", "")

    @user_feedback.setter
    def user_feedback(self, value: str) -> None:
        self._data["user_feedback"] = value
        self.save()

    @property
    def previous_summary(self) -> str:
        return self._data.get("previous_summary", "")

    @previous_summary.setter
    def previous_summary(self, value: str) -> None:
        self._data["previous_summary"] = value
        self.save()

    @property
    def ai_state(self) -> dict[str, Any]:
        if "ai_state" not in self._data:
            self._data["ai_state"] = {}
        return self._data["ai_state"]

    @property
    def image_paths(self) -> list[str]:
        return self._data.get("image_paths", [])

    @image_paths.setter
    def image_paths(self, value: list[str]) -> None:
        self._data["image_paths"] = value
        self.save()

    @property
    def phase_history(self) -> list[dict[str, Any]]:
        if "phase_history" not in self._data:
            self._data["phase_history"] = []
        return self._data["phase_history"]

    def to_dict(self) -> dict[str, Any]:
        """Return state as dictionary."""
        return dict(self._data)

    def update(self, data: dict[str, Any]) -> None:
        """Update multiple fields at once."""
        self._data.update(data)
        self.save()
