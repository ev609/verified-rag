"""Auto-citation injection.

Walk the answer, and for every numeric claim that a fact supports, splice an
inline ``[src: source, period, trust 0.92]`` tag right after it — unless the
author already cited it. Injects from the end backwards so earlier positions
don't drift.
"""
from __future__ import annotations

from typing import Optional

from .claims import _NUMBER_CLAIM_RE, extract_numeric_claims
from .llm import LLMExtractor, merge_claims
from .verify import verify_claims_against_facts


def inject_citations(
    text: str,
    reconciled_facts: dict,
    tolerance_pct: float = 5.0,
    *,
    llm_extractor: Optional[LLMExtractor] = None,
) -> tuple[str, dict]:
    """Return ``(text_with_citations, stats)``.

    Args:
        text: the answer to annotate.
        reconciled_facts: output of :func:`verified_rag.reconcile.reconcile_facts`
            (the ``reconciled_facts`` sub-dict).
        tolerance_pct: value tolerance for a claim to count as supported.
        llm_extractor: optional :class:`~verified_rag.llm.LLMExtractor` to also
            catch natural-language claims ("doubled", "grew threefold"). When
            omitted, only regex claims are cited.

    stats: ``{injected_count, already_cited, total_verified, total_claims,
    verification_rate}``.
    """
    claims = extract_numeric_claims(text)
    if llm_extractor is not None:
        try:
            claims = merge_claims(claims, llm_extractor.extract(text))
        except Exception:  # noqa: BLE001 — extractor failures must never break citing
            pass

    verification = verify_claims_against_facts(claims, reconciled_facts, tolerance_pct)
    by_pos = sorted(verification["verified"], key=lambda c: -c.get("position", 0))

    out = text
    injected = already = 0
    for v in by_pos:
        # Only regex claims carry a reliable position; LLM (natural-language)
        # claims are counted toward faithfulness but not auto-cited.
        if v.get("source") not in (None, "regex"):
            continue
        pos = v.get("position", 0)
        match = v["match"]

        # Insertion point = real end of the number+unit in the text (re-match at
        # pos). Reconstructing length from the parsed float breaks on "5.0" etc.
        # Processing is position-DESC, so earlier positions never drift.
        m = _NUMBER_CLAIM_RE.match(out, pos)
        if not m:
            continue
        # The regex's \s* can swallow a trailing space into the match; back up
        # over it so the citation sits flush against the number, not the word.
        insertion = m.end()
        while insertion > pos and out[insertion - 1].isspace():
            insertion -= 1

        # Already cited? Only inspect a *tiny* window at the insertion point, so
        # a neighbouring claim's freshly-injected citation can't trip this.
        window = out[insertion:insertion + 14].lower()
        if "[src:" in window or "[source:" in window:
            already += 1
            continue

        cite = f" [src: {match['source']}"
        if match.get("period_matched"):
            cite += f", {match['period_matched']}"
        cite += f", trust {match['confidence']:.2f}]"
        out = out[:insertion] + cite + out[insertion:]
        injected += 1

    return out, {
        "injected_count": injected,
        "already_cited": already,
        "total_verified": len(verification["verified"]),
        "total_claims": verification["total_claims"],
        "verification_rate": verification["verification_rate"],
    }
