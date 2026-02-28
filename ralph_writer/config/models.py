from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeSettings:
    base_url: str
    model: str
    api_key: str
    auto_pilot: bool
    stop_only_on_complete: bool
    stop_after_phase_change: bool
    max_context_tokens: int
    max_tool_calls_per_iteration: int
    tool_args_max_chars: int
    tool_result_max_chars: int
    stream_console_updates: bool
    summary_max_chars: int

    @classmethod
    def from_sources(cls, settings: dict[str, Any], environ: dict[str, str] | None = None) -> "RuntimeSettings":
        env = environ or os.environ
        return cls(
            base_url=settings.get("base_url", env.get("RALPH_BASE_URL", "http://localhost:1234/v1")),
            model=settings.get("model", env.get("RALPH_MODEL", "qwen/qwen3.5-35b-a3b")),
            api_key=settings.get("api_key", env.get("OPENAI_API_KEY", "lm-studio")),
            auto_pilot=settings.get("auto_pilot", env.get("RALPH_AUTO_PILOT", "true").lower() == "true"),
            stop_only_on_complete=settings.get(
                "stop_only_on_complete",
                env.get("RALPH_STOP_ONLY_ON_COMPLETE", "true").lower() == "true",
            ),
            stop_after_phase_change=settings.get(
                "stop_after_phase_change",
                env.get("RALPH_STOP_AFTER_PHASE_CHANGE", "true").lower() == "true",
            ),
            max_context_tokens=settings.get("max_context_tokens", int(env.get("RALPH_MAX_CONTEXT_TOKENS", "64000"))),
            max_tool_calls_per_iteration=settings.get(
                "max_tool_calls_per_iteration",
                int(env.get("RALPH_MAX_TOOL_CALLS_PER_ITERATION", "12")),
            ),
            tool_args_max_chars=settings.get("tool_args_max_chars", int(env.get("RALPH_TOOL_ARGS_MAX_CHARS", "240"))),
            tool_result_max_chars=settings.get(
                "tool_result_max_chars",
                int(env.get("RALPH_TOOL_RESULT_MAX_CHARS", "320")),
            ),
            stream_console_updates=settings.get(
                "stream_console_updates",
                env.get("RALPH_STREAM_CONSOLE_UPDATES", "true").lower() == "true",
            ),
            summary_max_chars=settings.get("summary_max_chars", int(env.get("RALPH_SUMMARY_MAX_CHARS", "800"))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "auto_pilot": self.auto_pilot,
            "stop_only_on_complete": self.stop_only_on_complete,
            "stop_after_phase_change": self.stop_after_phase_change,
            "max_context_tokens": self.max_context_tokens,
            "max_tool_calls_per_iteration": self.max_tool_calls_per_iteration,
            "tool_args_max_chars": self.tool_args_max_chars,
            "tool_result_max_chars": self.tool_result_max_chars,
            "stream_console_updates": self.stream_console_updates,
            "summary_max_chars": self.summary_max_chars,
        }


@dataclass(frozen=True)
class LoadedPhaseConfig:
    state_machine: dict[str, dict[str, Any]]
    phase_guide: dict[str, str]
    default_phase: str
    system_prompt_template: str
    settings: dict[str, Any]
