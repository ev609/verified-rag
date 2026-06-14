"""Pluggable LLM claim extractor (optional).

Regex catches ``"revenue $4.2B"``. It misses natural-language claims like
``"revenue doubled"`` or ``"grew threefold"`` — and those are where models
quietly hallucinate. An LLM extractor turns them into structured claims so the
verifier can check them too. In our production deployment, adding this lifted
the verified-claim rate from ~32% (regex only) to ~60%.

Bring your own model: implement :class:`LLMExtractor`, or use the bundled
:class:`AnthropicExtractor` (needs ``pip install anthropic`` and
``ANTHROPIC_API_KEY``). The core library has *zero* hard LLM dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional, Protocol

from .types import Claim

PROMPT = """Extract EVERY numeric claim from the text below.

INCLUDE:
- Explicit number+unit ("dividend 75 RUB", "P/E 4.5", "EBITDA $12.3B", "+8.7%")
- Natural-language multipliers ("doubled" -> value=2.0 unit="x", "halved" ->
  value=0.5 unit="x", "tripled" -> 3.0, "grew N-fold" -> N, "выросла в 2 раза" -> 2.0)
- Ratios ("ROE 12%", "Debt/EBITDA 1.8x")
- Period markers where explicit (FY2024, LTM, TTM, "next 12m")

IGNORE:
- Trivial numbers |x| < 0.5
- Standalone years (2000-2030 without a unit)
- Numbers already inside [src: ...] citations
- ID-like strings

Output STRICT JSON array, no markdown fences:
[{"value": 75.0, "unit": "RUB", "period": "FY2024", "claim_text": "dividend FY2024 = 75 RUB"}]

If there are no claims, output []. JSON only, no explanations.

TEXT:
{text}
"""


class LLMExtractor(Protocol):
    """Anything that turns answer text into a list of :class:`Claim`."""

    def extract(self, text: str) -> list[Claim]:  # pragma: no cover - protocol
        ...


class NullExtractor:
    """No-op extractor (the default when no LLM is configured)."""

    def extract(self, text: str) -> list[Claim]:
        return []


class AnthropicExtractor:
    """Extract natural-language claims with Claude. Optional, disk-cached.

    Args:
        model: Claude model id (default a cheap, fast model).
        api_key: overrides ``ANTHROPIC_API_KEY``.
        cache_dir: where to memoise results (SHA256 of text+model).
        max_chars: truncation guard for very long answers.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 max_chars: int = 60_000) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_chars = max_chars

    def extract(self, text: str) -> list[Claim]:
        if not text or not text.strip() or not self.api_key:
            return []
        text = text[:self.max_chars]
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        raw = self._call(text)
        claims = _parse(raw)
        self._cache_put(text, claims)
        return claims

    def _call(self, text: str) -> str:
        try:
            import anthropic  # optional dependency
        except ImportError:  # pragma: no cover
            return ""
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model, max_tokens=4000,
            messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content
                       if getattr(b, "type", "") == "text")

    def _key(self, text: str) -> str:
        h = hashlib.sha256()
        h.update(self.model.encode()); h.update(b"\x00")
        h.update(text.encode("utf-8", "ignore"))
        return h.hexdigest()

    def _cache_get(self, text: str) -> Optional[list[Claim]]:
        if not self.cache_dir:
            return None
        fp = self.cache_dir / f"{self._key(text)}.json"
        if not fp.exists():
            return None
        try:
            return [Claim(**c) for c in json.loads(fp.read_text())]
        except Exception:  # noqa: BLE001
            return None

    def _cache_put(self, text: str, claims: list[Claim]) -> None:
        if not self.cache_dir:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{self._key(text)}.json").write_text(
                json.dumps([c.__dict__ for c in claims], ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            pass


def _parse(raw: str) -> list[Claim]:
    """Pull a JSON array of claims out of an LLM response (fence-tolerant)."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if m:
            raw = m.group(1).strip()
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        arr = json.loads(raw[a:b + 1])
    except json.JSONDecodeError:
        return []
    out: list[Claim] = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        try:
            value = float(item["value"])
        except (TypeError, ValueError):
            continue
        out.append(Claim(value=value, unit=str(item.get("unit") or "")[:10],
                         period=item.get("period"),
                         claim_text=str(item.get("claim_text") or "")[:200],
                         source="llm"))
    return out


def merge_claims(regex_claims: list[Claim], llm_claims: list[Claim]) -> list[Claim]:
    """Merge regex + LLM claims, de-duping on value(±0.5%)+unit. Regex wins ties."""
    out = list(regex_claims)
    keys = {(round(c.value, 4), (c.unit or "").lower()) for c in regex_claims}
    for lc in llm_claims:
        k = (round(lc.value, 4), (lc.unit or "").lower())
        if k in keys:
            continue
        dup = any((lc.unit or "").lower() == u and rv and abs(lc.value - rv) / abs(rv) < 0.005
                  for rv, u in keys)
        if dup:
            continue
        keys.add(k)
        out.append(lc)
    return out
