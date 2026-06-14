"""High-level API: turn an answer + your facts into a verified, cited answer.

This is the one function most users call. Everything else is composable parts.
"""
from __future__ import annotations

from typing import Iterable, Optional

from .citations import inject_citations
from .claims import extract_numeric_claims
from .llm import LLMExtractor, merge_claims
from .reconcile import FactLike, reconcile_facts
from .verify import verify_claims_against_facts


def verify_answer(
    answer: str,
    facts: Iterable[FactLike],
    *,
    tolerance_pct: float = 5.0,
    llm_extractor: Optional[LLMExtractor] = None,
    sanity_ranges: Optional[dict[str, tuple[float, float]]] = None,
) -> dict:
    """Verify every number in ``answer`` against ``facts`` and cite the sources.

    Args:
        answer: the model-generated text to audit.
        facts: your retrieved facts (possibly several per metric / source).
        tolerance_pct: how close a number must be to count as supported.
        llm_extractor: optional, to also catch natural-language claims.
        sanity_ranges: optional per-key ``{key: (min, max)}`` guards.

    Returns a report dict:
        ``faithfulness`` (0..1 verified-claim rate), ``cited_answer`` (answer
        with ``[src: ...]`` tags injected), ``verified`` / ``unverified`` claim
        lists, ``conflicts`` (facts where sources disagree), ``reconciliation``
        (full consensus output) and ``citation_stats``.
    """
    reconciled = reconcile_facts(facts, sanity_ranges=sanity_ranges)
    rf = reconciled["reconciled_facts"]

    claims = extract_numeric_claims(answer)
    if llm_extractor is not None:
        try:
            claims = merge_claims(claims, llm_extractor.extract(answer))
        except Exception:  # noqa: BLE001
            pass

    verification = verify_claims_against_facts(claims, rf, tolerance_pct)
    cited_answer, cite_stats = inject_citations(
        answer, rf, tolerance_pct, llm_extractor=llm_extractor)

    conflicts = {k: v["conflicts"] for k, v in rf.items() if v["conflicts"]}
    return {
        "faithfulness": verification["verification_rate"],
        "answer": answer,
        "cited_answer": cited_answer,
        "verified": verification["verified"],
        "unverified": verification["unverified"],
        "verified_count": verification["verified_count"],
        "total_claims": verification["total_claims"],
        "conflicts": conflicts,
        "insane_facts": reconciled["insane_facts"],
        "reconciliation": reconciled,
        "citation_stats": cite_stats,
    }


def compare_modes(answer: str, facts: Iterable[FactLike],
                  tolerance_pct: float = 5.0) -> dict:
    """Show what period-aware matching buys you over naive value-only matching.

    Naive verifiers match a number to *any* fact within tolerance — including
    facts from the wrong period, which become confident-looking false
    citations. Period-aware matching rejects those. This quantifies the gap.

    Returns ``{total_claims, period_aware_verified, period_aware_rate,
    naive_verified, naive_rate, false_citations_avoided}``.
    """
    rf = reconcile_facts(facts)["reconciled_facts"]
    claims = extract_numeric_claims(answer)
    strict = verify_claims_against_facts(claims, rf, tolerance_pct)

    wrong_period = [u for u in strict["unverified"] if "possible_match_wrong_period" in u]
    naive_verified = strict["verified_count"] + len(wrong_period)
    total = strict["total_claims"]
    return {
        "total_claims": total,
        "period_aware_verified": strict["verified_count"],
        "period_aware_rate": round(strict["verified_count"] / total, 3) if total else 1.0,
        "naive_verified": naive_verified,
        "naive_rate": round(naive_verified / total, 3) if total else 1.0,
        "false_citations_avoided": len(wrong_period),
    }
