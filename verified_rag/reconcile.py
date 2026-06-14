"""Cross-source reconciliation.

Given the same metric reported by several sources, pick a primary value,
collect agreeing sources (which *boosts* confidence) and disagreeing ones
(which *penalises* it and is surfaced as a conflict). This is what turns "the
model said X" into "X, confirmed by 3 of 4 sources".

Ported and decoupled from a production pipeline where this lifted answer
faithfulness from 32% to 60%. Pure function — no I/O, no DB.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

from .types import Fact

FactLike = Union[Fact, dict]

# Default agreement / disagreement thresholds (percent difference).
AGREE_PCT = 1.0          # within this → "supporting" (same value)
MAX_BOOST = 0.10         # cap on confidence boost from agreement
BOOST_PER_SOURCE = 0.05  # +5% per agreeing source
CONFLICT_PENALTY = 0.15  # subtracted once if any source disagrees
MIN_CONFIDENCE = 0.30    # floor after penalties


def _as_fact(f: FactLike) -> Fact:
    if isinstance(f, Fact):
        return f
    return Fact(
        key=f["key"],
        value=float(f["value"]),
        source=f.get("source", "unknown"),
        confidence=float(f.get("confidence", 0.5)),
        period=f.get("period"),
        unit=f.get("unit"),
        source_url=f.get("source_url"),
        as_of=f.get("as_of"),
        meta=f.get("meta", {}) or {},
    )


def reconcile_facts(
    facts: Iterable[FactLike],
    *,
    sanity_ranges: Optional[dict[str, tuple[float, float]]] = None,
    agree_pct: float = AGREE_PCT,
) -> dict:
    """Reconcile candidate facts into one consensus value per key.

    Args:
        facts: iterable of :class:`Fact` (or equivalent dicts) — possibly many
            per key, from different sources.
        sanity_ranges: optional ``{key: (min, max)}``. A primary value outside
            its range is dropped into ``insane_facts`` instead of reconciled.
            Keys ending in ``_`` match by prefix (``dividend_`` covers
            ``dividend_2024``). Default: no range checks.
        agree_pct: percentage tolerance for two values to count as agreeing.

    Returns:
        ``{reconciled_facts, insane_facts, sources_used, fact_count,
        conflict_count}``. ``reconciled_facts`` maps key -> dict with
        ``value, confidence, primary_source, supporting_sources, conflicts,
        reconciled`` etc.
    """
    ranges = sanity_ranges or {}

    by_key: dict[str, list[Fact]] = {}
    for raw in facts:
        f = _as_fact(raw)
        by_key.setdefault(f.key, []).append(f)

    reconciled: dict[str, dict] = {}
    insane: list[dict] = []
    sources_used: set[str] = set()

    for key, candidates in by_key.items():
        # Highest confidence wins as primary.
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        primary = candidates[0]
        pv = primary.value

        if not _is_sane(key, pv, ranges):
            insane.append({"key": key, "value": pv, "source": primary.source,
                           "reason": "out_of_sanity_range"})
            continue

        supporting: list[dict] = []
        conflicts: list[dict] = []
        for c in candidates[1:]:
            diff_pct = abs(c.value - pv) / abs(pv) * 100 if pv else 100.0
            if diff_pct < agree_pct:
                supporting.append({"source": c.source, "confidence": c.confidence})
            else:
                conflicts.append({"source": c.source, "value": c.value,
                                  "diff_pct": round(diff_pct, 1),
                                  "confidence": c.confidence})

        conf = min(primary.confidence + min(BOOST_PER_SOURCE * len(supporting),
                                            MAX_BOOST), 1.0)
        if conflicts:
            conf = max(conf - CONFLICT_PENALTY, MIN_CONFIDENCE)

        reconciled[key] = {
            "value": pv,
            "unit": primary.unit,
            "period": primary.period,
            "as_of": primary.as_of,
            "primary_source": primary.source,
            "source_url": primary.source_url,
            "confidence": round(conf, 2),
            "reconciled": bool(supporting) and not conflicts,
            "supporting_sources": supporting,
            "conflicts": conflicts,
            **primary.meta,
        }
        sources_used.add(primary.source)
        sources_used.update(s["source"] for s in supporting)

    return {
        "reconciled_facts": reconciled,
        "insane_facts": insane,
        "sources_used": sorted(sources_used),
        "fact_count": len(reconciled),
        "conflict_count": sum(1 for r in reconciled.values() if r["conflicts"]),
    }


def _is_sane(key: str, value: float, ranges: dict[str, tuple[float, float]]) -> bool:
    rng = ranges.get(key)
    if rng:
        return rng[0] <= value <= rng[1]
    for prefix, r in ranges.items():
        if prefix.endswith("_") and key.startswith(prefix):
            return r[0] <= value <= r[1]
    return True
