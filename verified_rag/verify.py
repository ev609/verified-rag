"""Period-aware verification of claims against reconciled facts.

Two-stage matching:

1. **Period-aware** — if a claim's text implies a period (``FY2024``, ``LTM``),
   only match facts of that period. This is what stops a 2024 claim from being
   "verified" by a 2023 number — the single biggest source of false citations.
2. **Value-only fallback** — for period-agnostic claims (a current price), match
   the closest value within tolerance. Wrong-period near-matches are *not*
   accepted; they are flagged as ``possible_match_wrong_period``.
"""
from __future__ import annotations

import re
from typing import Optional

from .claims import extract_period_hint, fact_key_period
from .types import Claim


def verify_claims_against_facts(
    claims: list[Claim],
    reconciled_facts: dict,
    tolerance_pct: float = 5.0,
) -> dict:
    """Cross-check claims against facts. Returns verified/unverified split + rate.

    Returns:
        ``{verified, unverified, total_claims, verified_count,
        unverified_count, verification_rate}``. Each verified claim carries a
        ``match`` dict (``fact_key, fact_value, source, confidence,
        period_matched, match_quality``). ``match_quality`` is ``period_value``
        (strong) or ``value_only`` (weaker).
    """
    verified: list[dict] = []
    unverified: list[dict] = []

    indexed: list[tuple[float, str, Optional[str], dict]] = []
    for key, info in reconciled_facts.items():
        val = info.get("value")
        if not isinstance(val, (int, float)):
            continue
        period = fact_key_period(key)
        if not period and info.get("period"):
            m = re.search(r"(\d{4})", str(info["period"]))
            if m:
                period = m.group(1)
            elif re.search(r"ttm|ltm", str(info["period"]), re.IGNORECASE):
                period = "ltm"
        indexed.append((float(val), key, period, info))

    for claim in claims:
        cv = claim.value
        claim_period = claim.period or extract_period_hint(claim.context, claim.context_before)
        matched = None
        wrong_period = None

        # Stage 1: period-aware
        if claim_period:
            best = []
            for fv, fk, fp, fi in indexed:
                if abs(fv) < 1e-9 or fp != claim_period:
                    continue
                d = abs(fv - cv) / abs(fv) * 100
                if d <= tolerance_pct:
                    best.append((d, fv, fk, fi))
            if best:
                best.sort(key=lambda x: x[0])
                d, fv, fk, fi = best[0]
                matched = _match(fk, fv, d, fi, claim_period, "period_value")

        # Stage 2: value-only fallback
        if not matched:
            best = []
            for fv, fk, fp, fi in indexed:
                if abs(fv) < 1e-9:
                    continue
                d = abs(fv - cv) / abs(fv) * 100
                if d > tolerance_pct:
                    continue
                if claim_period and fp and fp != claim_period:
                    if not wrong_period or d < wrong_period["diff_pct"]:
                        wrong_period = {"fact_key": fk, "fact_value": fv,
                                        "diff_pct": round(d, 1),
                                        "source": fi["primary_source"],
                                        "period_mismatch": f"claim={claim_period}, fact={fp}"}
                    continue
                best.append((d, fv, fk, fi))
            if best:
                best.sort(key=lambda x: x[0])
                d, fv, fk, fi = best[0]
                matched = _match(fk, fv, d, fi, None, "value_only")

        entry = _claim_dict(claim)
        if matched:
            verified.append({**entry, "match": matched})
        else:
            if wrong_period:
                entry["possible_match_wrong_period"] = wrong_period
            unverified.append(entry)

    total = len(claims)
    return {
        "verified": verified,
        "unverified": unverified,
        "total_claims": total,
        "verified_count": len(verified),
        "unverified_count": len(unverified),
        "verification_rate": round(len(verified) / total, 3) if total else 1.0,
    }


def _match(fk, fv, d, fi, period, quality) -> dict:
    return {"fact_key": fk, "fact_value": fv, "diff_pct": round(d, 1),
            "source": fi["primary_source"], "confidence": fi["confidence"],
            "period_matched": period, "match_quality": quality}


def _claim_dict(c: Claim) -> dict:
    return {"value": c.value, "unit": c.unit, "context": c.context,
            "context_before": c.context_before, "position": c.position,
            "period": c.period, "claim_text": c.claim_text, "source": c.source}
