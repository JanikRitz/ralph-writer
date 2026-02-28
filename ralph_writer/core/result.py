"""Result types for orchestration and iteration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class IterationResult:
	"""Result from a single iteration of the orchestrator."""
	state: dict[str, Any]
	loop_row: dict[str, Any]
	status: str
	total_tokens: int
