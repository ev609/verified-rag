"""Core data types for the verified-RAG layer.

Domain-neutral: a ``Fact`` is any numeric assertion that came from a named
source for a given period. The whole library operates on lists of facts and
plain answer text — it knows nothing about finance, equities or any vertical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Fact:
    """One numeric fact from one source.

    Args:
        key: stable identifier of the metric, optionally period-suffixed
            (e.g. ``revenue_2024``, ``headcount_ltm``, ``current_price``).
            The period suffix is parsed automatically for period-aware matching.
        value: numeric value.
        source: name of the source that produced it (used in citations).
        confidence: trust score 0..1 (per-source baseline; boosted/penalised
            during reconciliation).
        period: optional explicit period ("FY2024", "ltm", "next_12m"); if
            omitted it is inferred from ``key``.
        unit: optional unit ("%", "$", "USD", "x", ...). Informational.
        source_url: optional provenance URL, surfaced in the audit trail.
        as_of: optional timestamp/string describing when the value was true.
        meta: free-form extra fields kept on the reconciled output.
    """

    key: str
    value: float
    source: str
    confidence: float = 0.5
    period: Optional[str] = None
    unit: Optional[str] = None
    source_url: Optional[str] = None
    as_of: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    """A numeric claim extracted from an answer text."""

    value: float
    unit: str = ""
    context: str = ""
    context_before: str = ""
    position: int = 0
    period: Optional[str] = None
    claim_text: str = ""
    source: str = "regex"  # "regex" | "llm" | custom extractor name
