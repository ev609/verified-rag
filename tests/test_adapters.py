"""Tests for facts_from_records (retriever integration)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verified_rag import Fact, facts_from_records, verify_answer  # noqa: E402


def test_from_dicts_with_field_names():
    rows = [
        {"metric": "revenue_2024", "val": 4.2, "src": "10-K", "trust": 1.0},
        {"metric": "revenue_2023", "val": 3.5, "src": "10-K", "trust": 1.0},
    ]
    facts = facts_from_records(rows, key="metric", value="val",
                              source="src", confidence="trust")
    assert len(facts) == 2
    assert all(isinstance(f, Fact) for f in facts)
    assert facts[0].confidence == 1.0


def test_skips_non_numeric_and_missing_key():
    rows = [
        {"metric": "rev_2024", "val": 4.2, "src": "x"},
        {"text": "prose chunk, no number", "src": "x"},      # no key/value
        {"metric": "bad", "val": "not-a-number", "src": "x"},  # non-numeric
    ]
    facts = facts_from_records(rows, key="metric", value="val", source="src")
    assert len(facts) == 1


def test_callable_specs_and_constant_source():
    rows = [{"k": "headcount_2024", "v": "1200"}]
    facts = facts_from_records(
        rows, key=lambda r: r["k"], value=lambda r: r["v"],
        source="LinkedIn",  # constant — no such field, used as-is
        confidence=0.6,
    )
    assert facts[0].key == "headcount_2024"
    assert facts[0].value == 1200.0
    assert facts[0].source == "LinkedIn"


def test_end_to_end_from_records():
    rows = [{"metric": "revenue_2024", "val": 4.2, "src": "10-K", "trust": 1.0}]
    facts = facts_from_records(rows, key="metric", value="val",
                              source="src", confidence="trust")
    report = verify_answer("In 2024, revenue was $4.2B.", facts)
    assert report["faithfulness"] == 1.0
    assert "[src: 10-K" in report["cited_answer"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("ALL PASSED")
