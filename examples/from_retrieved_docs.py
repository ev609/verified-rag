"""Integrate with your retriever (LangChain / LlamaIndex / vector store / API).

Runnable as-is with a stand-in retriever. The only integration work is mapping
your retrieved records onto facts with ``facts_from_records`` — shown for the
common frameworks in the comments.

    python examples/from_retrieved_docs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verified_rag import facts_from_records, verify_answer  # noqa: E402


# --- 1. Whatever your retriever returns -------------------------------------
# Here: plain dicts, as you'd pull from a vector store / SQL / API. Each row is
# a numeric fact with provenance. (Prose chunks without a numeric value are
# simply skipped by facts_from_records.)
retrieved = [
    {"metric": "revenue_2024", "val": 4.20, "src": "10-K filing", "trust": 1.0},
    {"metric": "revenue_2023", "val": 3.50, "src": "10-K filing", "trust": 1.0},
    {"metric": "arr_2024",     "val": 5.00, "src": "investor deck", "trust": 0.8},
    {"metric": "headcount_2024", "val": 1200, "src": "LinkedIn", "trust": 0.6},
    {"text": "The company is headquartered in Austin.", "src": "wiki"},  # not a fact
]

facts = facts_from_records(
    retrieved,
    key="metric",
    value="val",
    source="src",
    confidence="trust",
)

# --- 2. Your LLM's answer ---------------------------------------------------
answer = ("In 2024, revenue was $4.2B, up from $3.5B in 2023. "
          "ARR in 2024 reached $5.0B with around 1200 employees.")

# --- 3. Verify + cite -------------------------------------------------------
report = verify_answer(answer, facts)
print("faithfulness:", report["faithfulness"])
print(report["cited_answer"])


# ---------------------------------------------------------------------------
# LangChain:
#   docs = retriever.invoke(question)            # list[Document]
#   facts = facts_from_records(
#       docs,
#       key=lambda d: d.metadata["metric"],
#       value=lambda d: d.metadata["value"],
#       source=lambda d: d.metadata.get("source", "retrieved"),
#       confidence=lambda d: d.metadata.get("score", 0.5),
#   )
#
# LlamaIndex:
#   nodes = retriever.retrieve(question)         # list[NodeWithScore]
#   facts = facts_from_records(
#       nodes,
#       key=lambda n: n.metadata["metric"],
#       value=lambda n: n.metadata["value"],
#       source=lambda n: n.metadata.get("source", "retrieved"),
#       confidence=lambda n: n.score or 0.5,
#   )
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pass
