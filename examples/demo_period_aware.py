"""Demo: period-aware verification catches a confident-looking lie.

No API key needed — pure, deterministic. Run:

    python examples/demo_period_aware.py

The planted answer contains one false claim ("ARR for 2023 was $5.0B" — that
was actually 2024; 2023 was $4.0B). A naive value-only verifier rubber-stamps
it (the number 5.0 exists *somewhere* in the facts). Period-aware matching
rejects it and flags the wrong period.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verified_rag import Fact, compare_modes, verify_answer  # noqa: E402

# Retrieved facts — several per metric / source, mixed periods, one conflict,
# one near-duplicate that should boost confidence.
FACTS = [
    Fact("revenue_2024", 4.20, source="10-K filing", confidence=1.0, unit="$B"),
    Fact("revenue_2024", 4.18, source="press release", confidence=0.9, unit="$B"),  # agrees -> boost
    Fact("revenue_2023", 3.50, source="10-K filing", confidence=1.0, unit="$B"),
    Fact("arr_2024", 5.00, source="investor deck", confidence=0.8, unit="$B"),
    Fact("arr_2023", 4.00, source="investor deck", confidence=0.8, unit="$B"),
    Fact("gross_margin_2024", 78.0, source="10-K filing", confidence=1.0, unit="%"),
    Fact("nps_2024", 72.0, source="customer survey", confidence=0.7),
    Fact("nps_2024", 50.0, source="marketing blog", confidence=0.4),  # conflicts
]

# Model answer. Note the planted error on the ARR-2023 line.
ANSWER = (
    "In 2024, revenue was $4.2B. In 2023, revenue was $3.5B. "
    "For 2023, ARR was $5.0B. Gross margin in 2024 was 78%. NPS in 2024 was 72."
)


def main() -> None:
    print("=" * 72)
    print("ANSWER UNDER AUDIT:\n  " + ANSWER)
    print("=" * 72)

    cmp = compare_modes(ANSWER, FACTS)
    print("\nNaive value-only verifier:  "
          f"{cmp['naive_verified']}/{cmp['total_claims']} 'verified'  "
          f"({cmp['naive_rate']:.0%})  <- looks perfect, but trusts a lie")
    print("Period-aware verifier:      "
          f"{cmp['period_aware_verified']}/{cmp['total_claims']} verified  "
          f"({cmp['period_aware_rate']:.0%})  <- the honest number")
    print(f"False citations avoided:    {cmp['false_citations_avoided']}")

    report = verify_answer(ANSWER, FACTS)
    print("\n--- CITED ANSWER ----------------------------------------------------")
    print(report["cited_answer"])
    print("\n--- UNVERIFIED (flagged, NOT cited) ---------------------------------")
    for u in report["unverified"]:
        line = f"  value={u['value']} {u['unit']}".rstrip()
        wp = u.get("possible_match_wrong_period")
        if wp:
            line += (f"  ->  closest fact {wp['fact_key']}={wp['fact_value']} "
                     f"but {wp['period_mismatch']}")
        print(line)
    print("\n--- SOURCE CONFLICTS ------------------------------------------------")
    for key, conflicts in report["conflicts"].items():
        for c in conflicts:
            print(f"  {key}: primary vs {c['source']}={c['value']} "
                  f"(off {c['diff_pct']}%)")
    print("\nfaithfulness score:", report["faithfulness"])


if __name__ == "__main__":
    main()
