#!/usr/bin/env python3
"""
Curator boundary helpers.

Centralized, fail-closed parsing for SynapseEvent envelopes:
- Counter metrics per consumer
- Structured, sampled warning logs
- No exceptions leak across the boundary
"""

import logging
from collections import Counter
from typing import Any, Dict, Optional

from curator_adapters import SynapseEvent

logger = logging.getLogger(__name__)

REJECTION_METRICS = Counter()
REJECTION_SAMPLE_LIMIT = 10


def safe_parse_synapse(raw_event: Dict[str, Any], consumer_name: str) -> Optional[SynapseEvent]:
    """Parse a raw event into SynapseEvent. Returns None on invalid input."""
    try:
        return SynapseEvent.from_dict(raw_event)
    except Exception as exc:
        key = f"{consumer_name}.synapse_event_invalid"
        REJECTION_METRICS[key] += 1
        if REJECTION_METRICS[key] <= REJECTION_SAMPLE_LIMIT:
            logger.warning(
                "SynapseEvent rejected",
                extra={
                    "error": str(exc),
                    "consumer": consumer_name,
                    "event_id": raw_event.get("id") if isinstance(raw_event, dict) else None,
                    "event_type": raw_event.get("event_type") if isinstance(raw_event, dict) else None,
                    "reject_count": REJECTION_METRICS[key],
                },
            )
        return None
