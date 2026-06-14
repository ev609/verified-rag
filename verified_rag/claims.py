"""Numeric claim extraction from answer text + period detection.

Regex-based, language-neutral (handles ``$ € £ ₽``, ``B/M/K``, ``bn/mn``,
``млрд/млн/тыс`` and bare percentages). Catches the explicit ``"revenue $4.2B"``
style claims. Natural-language claims (``"doubled"``, ``"grew threefold"``)
need an LLM extractor — see :mod:`verified_rag.llm`.
"""
from __future__ import annotations

import re
from typing import Optional

from .types import Claim

# value + optional unit/magnitude. Order matters: longer tokens first.
_NUMBER_CLAIM_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я_])"
    r"(?P<value>-?\d{1,12}(?:[\.,]\d+)?)"
    r"\s*"
    r"(?P<unit>%|‰|\$|€|£|₽|bn|mn|млрд|млн|тыс|billion|million|thousand|[BMK]\b|x|х)?",
    re.IGNORECASE,
)

_DATE_LIKE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}|\d{2}/\d{2}/\d{4}|FY\d{4}|\bQ[1-4]\s*\d{4}",
    re.IGNORECASE,
)
_CITATION_BLOCK_RE = re.compile(
    r"\[src:[^\]]+\]|\[source:[^\]]+\]|\[источник:[^\]]+\]", re.IGNORECASE)


def extract_numeric_claims(text: str, min_value: float = 0.5) -> list[Claim]:
    """Pull numeric claims out of an answer.

    Skips: trivial numbers (``|x| < min_value``), bare years 2000-2030, numbers
    inside dates, and numbers already inside a ``[src: ...]`` citation block.

    Each claim keeps a 200-char ``context`` and a ``context_before`` window so
    the period can be inferred from text to its left.
    """
    skip: list[tuple[int, int]] = []
    for rx in (_DATE_LIKE_RE, _CITATION_BLOCK_RE):
        skip.extend((m.start(), m.end()) for m in rx.finditer(text))

    def _skipped(a: int, b: int) -> bool:
        return any(a >= s and b <= e for s, e in skip)

    claims: list[Claim] = []
    seen: set[int] = set()
    for m in _NUMBER_CLAIM_RE.finditer(text):
        try:
            value = float(m.group("value").replace(",", "."))
        except (ValueError, TypeError):
            continue
        if abs(value) < min_value:
            continue
        unit = (m.group("unit") or "").strip()
        if 2000 <= value <= 2030 and not unit:
            continue
        if _skipped(m.start(), m.end()) or m.start() in seen:
            continue
        seen.add(m.start())
        cs = max(0, m.start() - 200)
        ce = min(len(text), m.end() + 50)
        claims.append(Claim(
            value=value,
            unit=unit,
            context=text[cs:ce].replace("\n", " "),
            context_before=text[cs:m.start()].replace("\n", " "),
            position=m.start(),
        ))
    return claims


# --- period detection -------------------------------------------------------

def extract_period_hint(context: str, context_before: Optional[str] = None) -> Optional[str]:
    """Infer the period a claim refers to, picking the marker closest to it.

    Recognised (EN + RU): ``LTM/TTM`` -> ``"ltm"``; ``next 12m`` -> ``"next_12m"``;
    ``FY2024`` / ``for 2024`` / ``за 2024`` -> ``"2024"``; bare year as fallback.
    """
    primary = context_before if context_before is not None else context
    if not primary:
        return None
    primary = re.sub(r"\d{4}-\d{2}-\d{2}", "          ", primary)

    cand: list[tuple[int, str]] = []
    for m in re.finditer(r"\b(LTM|TTM)\b", primary, re.IGNORECASE):
        cand.append((m.end(), "ltm"))
    for m in re.finditer(r"\bnext\s+12\s*m\b|\bпрогноз[^.]{0,30}12\s*мес", primary, re.IGNORECASE):
        cand.append((m.end(), "next_12m"))
    for m in re.finditer(r"\bFY\s*(\d{4})\b", primary, re.IGNORECASE):
        cand.append((m.end(), m.group(1)))
    for m in re.finditer(r"\b(?:for|in|за)\s+(\d{4})\b", primary, re.IGNORECASE):
        cand.append((m.end(), m.group(1)))
    if not cand:
        for m in re.finditer(r"\b(20[12][0-9])\b", primary):
            cand.append((m.end(), m.group(1)))
    if not cand:
        return None
    cand.sort(key=lambda c: -c[0])
    return cand[0][1]


def fact_key_period(key: str) -> Optional[str]:
    """Extract the period encoded in a fact key (``revenue_2024`` -> ``"2024"``)."""
    m = re.search(r"_(20[12]\d)(?:_|$)", key)
    if m:
        return m.group(1)
    low = key.lower()
    if low.endswith("_ltm") or "ttm" in low:
        return "ltm"
    if "next_12m" in low:
        return "next_12m"
    return None
