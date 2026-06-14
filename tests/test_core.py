"""Core tests for the verified-rag layer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verified_rag import (  # noqa: E402
    Fact,
    compare_modes,
    extract_numeric_claims,
    reconcile_facts,
    verify_answer,
)


def test_reconcile_boost_and_conflict():
    facts = [
        Fact("rev_2024", 100.0, source="a", confidence=0.8),
        Fact("rev_2024", 100.5, source="b", confidence=0.7),   # agrees (<1%) -> boost
        Fact("rev_2024", 130.0, source="c", confidence=0.6),   # conflicts (>1%)
    ]
    out = reconcile_facts(facts)["reconciled_facts"]["rev_2024"]
    assert out["value"] == 100.0           # highest-confidence primary
    assert out["supporting_sources"]       # b supports
    assert out["conflicts"]                # c conflicts
    # conflict penalty applied -> below the boosted ceiling
    assert out["confidence"] <= 0.80


def test_period_aware_rejects_wrong_period():
    facts = [
        Fact("arr_2024", 5.0, source="deck", confidence=0.9, unit="$B"),
        Fact("arr_2023", 4.0, source="deck", confidence=0.9, unit="$B"),
    ]
    # claim says 2023 = 5.0 (false; 5.0 was 2024). Must NOT verify.
    report = verify_answer("For 2023, ARR was $5.0B.", facts)
    assert report["verified_count"] == 0
    assert report["unverified"][0].get("possible_match_wrong_period")


def test_correct_period_verifies_and_cites():
    facts = [Fact("revenue_2024", 4.2, source="10-K", confidence=1.0, unit="$B")]
    report = verify_answer("In 2024, revenue was $4.2B.", facts)
    assert report["verified_count"] == 1
    assert "[src: 10-K" in report["cited_answer"]
    assert report["faithfulness"] == 1.0


def test_compare_modes_quantifies_false_citations():
    facts = [
        Fact("arr_2024", 5.0, source="deck", confidence=0.9),
        Fact("arr_2023", 4.0, source="deck", confidence=0.9),
        Fact("revenue_2024", 4.2, source="10-K", confidence=1.0),
    ]
    answer = "In 2024, revenue was $4.2B. For 2023, ARR was $5.0B."
    cmp = compare_modes(answer, facts)
    assert cmp["false_citations_avoided"] == 1
    assert cmp["naive_verified"] > cmp["period_aware_verified"]


def test_all_verified_claims_get_cited_even_when_adjacent():
    # Regression: close-together claims must each get a citation; a neighbour's
    # freshly-injected [src:] must not make this one look "already cited".
    facts = [
        Fact("revenue_2024", 4.2, source="10-K", confidence=1.0, unit="$B"),
        Fact("revenue_2023", 3.5, source="10-K", confidence=1.0, unit="$B"),
        Fact("gross_margin_2024", 78.0, source="10-K", confidence=1.0, unit="%"),
    ]
    answer = ("In 2024, revenue was $4.2B. In 2023, revenue was $3.5B. "
              "Gross margin in 2024 was 78%.")
    report = verify_answer(answer, facts)
    assert report["verified_count"] == 3
    assert report["cited_answer"].count("[src:") == 3


def test_citation_not_injected_inside_a_decimal():
    # Regression: "$5.0B" must become "$5.0B [src..]", never "$5 [src..].0B".
    facts = [Fact("arr_2024", 5.0, source="deck", confidence=0.8, unit="$B")]
    report = verify_answer("In 2024, ARR reached $5.0B.", facts)
    assert "$5.0B [src:" in report["cited_answer"]
    assert ".0B" not in report["cited_answer"].split("[src:")[0][-3:]


def test_extract_skips_years_and_citations():
    claims = extract_numeric_claims("In 2024 revenue hit $4.2B [src: x, trust 0.9].")
    values = [c.value for c in claims]
    assert 4.2 in values
    assert 2024 not in values     # bare year skipped
    # the 0.9 inside the citation block must be skipped
    assert 0.9 not in values


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("ALL PASSED")
