"""Helpers for loading optional local JSON context for LLM prompts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_context_json_block(config) -> str:
    """
    Load optional user-provided JSON context and return a pretty JSON string.

    Returns an empty string when context is not configured or cannot be loaded.
    """
    analysis_cfg = getattr(config, "analysis", None)
    path_value = getattr(analysis_cfg, "context_json_path", None) if analysis_cfg else None
    if not path_value:
        return ""

    context_path = Path(str(path_value)).expanduser()
    if not context_path.exists():
        logger.warning("Configured context_json_path does not exist: %s", context_path)
        return ""

    try:
        raw_text = context_path.read_text()
        parsed = json.loads(raw_text)
        serialized = json.dumps(parsed, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed to parse context JSON from %s: %s", context_path, e)
        return ""

    max_chars = 12000
    if analysis_cfg is not None:
        try:
            max_chars = int(getattr(analysis_cfg, "context_max_chars", 12000) or 12000)
        except Exception:
            max_chars = 12000
    if max_chars < 1:
        max_chars = 1

    if len(serialized) > max_chars:
        logger.info(
            "Context JSON truncated from %d to %d chars for prompt safety.",
            len(serialized),
            max_chars,
        )
        serialized = serialized[:max_chars]

    return serialized

