"""Turn whatever your retriever returns into :class:`Fact` objects.

The one bit of friction in adopting this library is mapping *your* retrieved
records (LangChain ``Document``s, LlamaIndex nodes, vector-store rows, API JSON)
onto the ``Fact`` shape. ``facts_from_records`` does that with a tiny mapping
spec — each field is either a record attribute/key name, a callable, or a
constant.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from .types import Fact

Spec = Union[str, Callable[[Any], Any], float, int, None]


def _read(record: Any, spec: Spec) -> Any:
    """Resolve one field spec against a record.

    - callable -> ``spec(record)``
    - str that names a dict key / attribute -> that value
    - anything else (incl. a str with no matching field) -> used as a constant
    """
    if callable(spec):
        return spec(record)
    if isinstance(spec, str):
        if isinstance(record, dict) and spec in record:
            return record[spec]
        if hasattr(record, spec):
            return getattr(record, spec)
    return spec


def facts_from_records(
    records: Iterable[Any],
    *,
    key: Spec,
    value: Spec,
    source: Spec,
    confidence: Spec = 0.5,
    period: Spec = None,
    unit: Spec = None,
    source_url: Spec = None,
) -> list[Fact]:
    """Build a list of :class:`Fact` from retrieved records.

    Each keyword is a *spec*: a field name (dict key or attribute), a callable
    ``record -> value``, or a constant applied to every record.

    Example (framework-agnostic)::

        rows = [{"metric": "revenue_2024", "val": 4.2, "src": "10-K"}, ...]
        facts = facts_from_records(
            rows, key="metric", value="val", source="src", confidence=1.0)

    Records whose ``value`` can't be coerced to float, or that have no ``key``,
    are skipped (your retriever returns prose chunks too — those aren't facts).
    """
    out: list[Fact] = []
    for r in records:
        k = _read(r, key)
        v = _read(r, value)
        if k is None or v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        out.append(Fact(
            key=str(k),
            value=fv,
            source=str(_read(r, source) or "unknown"),
            confidence=_coerce_conf(_read(r, confidence)),
            period=_opt_str(_read(r, period)),
            unit=_opt_str(_read(r, unit)),
            source_url=_opt_str(_read(r, source_url)),
        ))
    return out


def _coerce_conf(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.5


def _opt_str(v: Any) -> Optional[str]:
    return None if v is None else str(v)
