from __future__ import annotations

import json
from typing import Any

import tiktoken


def estimate_tokens_text(model: str, text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_tokens_messages(model: str, messages: list[dict[str, Any]]) -> int:
    flattened = "\n".join(json.dumps(m, ensure_ascii=False, default=str) for m in messages)
    return estimate_tokens_text(model, flattened)
