"""verified-rag — make your RAG bot cite every number (and stop it lying).

Drop-in verification layer for any RAG / LLM answer:

    >>> from verified_rag import verify_answer, Fact
    >>> facts = [
    ...     Fact("revenue_2024", 4.2, source="10-K", confidence=1.0, unit="$B"),
    ...     Fact("revenue_2023", 3.5, source="10-K", confidence=1.0, unit="$B"),
    ... ]
    >>> report = verify_answer("Revenue for 2024 was $4.2B, up from 2023.", facts)
    >>> report["faithfulness"], report["cited_answer"]

What it does, with no vertical assumptions:
  * cross-source reconciliation (consensus + conflict detection)
  * period-aware claim matching (a 2024 claim never "verifies" against 2023)
  * a faithfulness score (verified-claim rate)
  * inline citation injection ``[src: source, period, trust 0.92]``
  * optional LLM extraction of natural-language claims ("doubled", "grew 3x")
"""
from .adapters import facts_from_records
from .citations import inject_citations
from .claims import extract_numeric_claims, extract_period_hint, fact_key_period
from .faithfulness import compare_modes, verify_answer
from .llm import AnthropicExtractor, LLMExtractor, NullExtractor, merge_claims
from .reconcile import reconcile_facts
from .types import Claim, Fact
from .verify import verify_claims_against_facts

__version__ = "0.1.0"

__all__ = [
    "Fact",
    "Claim",
    "facts_from_records",
    "verify_answer",
    "compare_modes",
    "reconcile_facts",
    "extract_numeric_claims",
    "extract_period_hint",
    "fact_key_period",
    "verify_claims_against_facts",
    "inject_citations",
    "LLMExtractor",
    "NullExtractor",
    "AnthropicExtractor",
    "merge_claims",
]
