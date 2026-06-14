"""Minimal quickstart — verify and cite an answer in ~10 lines.

    python examples/quickstart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verified_rag import Fact, verify_answer  # noqa: E402

facts = [
    Fact("revenue_2024", 4.2, source="annual report", confidence=1.0, unit="$B"),
    Fact("employees_2024", 1200, source="LinkedIn", confidence=0.6),
]

answer = "In 2024 the company made $4.2B in revenue with about 1200 employees."

report = verify_answer(answer, facts)
print("faithfulness:", report["faithfulness"])
print(report["cited_answer"])
