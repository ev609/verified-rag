# verified-rag — stop your RAG bot from lying

[![CI](https://github.com/ev609/verified-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ev609/verified-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![deps](https://img.shields.io/badge/core%20deps-0-brightgreen.svg)

**A drop-in layer that checks every number your LLM produces against your
sources, cites the ones it can prove, and flags the ones it can't.**

Your retrieval works. Your model still occasionally states a number that isn't
in the context — or cites the *right value for the wrong period*. Generic
"is it grounded?" scorers miss that second one entirely. This library is built
around it.

> Ported from a production pipeline where adding this layer lifted answer
> faithfulness from **32% → 60%** verified claims, with every verified number
> carrying an inline source citation.

---

## What it does

- **Cross-source reconciliation** — the same metric from several sources becomes
  one consensus value; agreement *boosts* confidence, disagreement is surfaced
  as a conflict (not silently averaged away).
- **Period-aware claim matching** — a `2024` claim never "verifies" against a
  `2023` fact. This is the #1 source of confident-looking false citations, and
  the thing most verifiers get wrong.
- **Faithfulness score** — a single 0–1 number: what fraction of the answer's
  claims are backed by your facts.
- **Inline citation injection** — splices `[src: 10-K filing, 2024, trust 1.00]`
  after every verified number; leaves unverifiable ones visibly uncited.
- **Optional LLM extraction** — also catch natural-language claims like
  "revenue *doubled*" or "grew *threefold*" (bring your own model).

Core is **pure Python, zero dependencies**. The LLM extractor is optional.

---

## See it catch a lie (no API key)

```bash
python examples/demo_period_aware.py
```

```
ANSWER UNDER AUDIT:
  In 2024, revenue was $4.2B. In 2023, revenue was $3.5B. For 2023, ARR was
  $5.0B. Gross margin in 2024 was 78%. NPS in 2024 was 72.

Naive value-only verifier:  5/5 'verified'  (100%)  <- looks perfect, but trusts a lie
Period-aware verifier:      4/5 verified  (80%)  <- the honest number
False citations avoided:    1

--- CITED ANSWER ---
In 2024, revenue was $4.2B [src: 10-K filing, 2024, trust 1.00]. In 2023,
revenue was $3.5B [src: 10-K filing, 2023, trust 1.00]. For 2023, ARR was $5.0B.
Gross margin in 2024 was 78% [src: 10-K filing, 2024, trust 1.00]. NPS in 2024
was 72 [src: customer survey, 2024, trust 0.55].

--- FLAGGED (not cited) ---
  value=5.0  ->  closest fact arr_2024=5.0 but claim=2023, fact=2024
```

The answer *said* "For 2023, ARR was $5.0B." That's false — $5.0B was 2024;
2023 was $4.0B. A naive verifier rubber-stamps it because the number 5.0 exists
*somewhere*. This one rejects it and tells you why.

---

## Quickstart

```python
from verified_rag import verify_answer, Fact

facts = [
    Fact("revenue_2024", 4.2, source="annual report", confidence=1.0, unit="$B"),
    Fact("employees_2024", 1200, source="LinkedIn",     confidence=0.6),
]

report = verify_answer(
    "In 2024 the company made $4.2B in revenue with about 1200 employees.",
    facts,
)

report["faithfulness"]    # -> 1.0
report["cited_answer"]    # answer with [src: ...] tags injected
report["unverified"]      # claims you should not trust
report["conflicts"]       # facts where your sources disagree
```

A `Fact` is anything: a row from your vector store, a field from an API, a
number scraped from a PDF. The library is domain-neutral — finance, legal,
medical, internal docs, whatever.

### Add natural-language claim extraction (optional)

```python
from verified_rag import verify_answer, AnthropicExtractor

extractor = AnthropicExtractor()            # uses ANTHROPIC_API_KEY
report = verify_answer(answer, facts, llm_extractor=extractor)
```

Or implement the 1-method `LLMExtractor` protocol for any model you like.

---

## Works with your stack (LangChain, LlamaIndex, anything)

The only glue is turning what your retriever returns into `Fact`s —
`facts_from_records` does it with a tiny mapping (a field name, a callable, or a
constant per field):

```python
from verified_rag import facts_from_records, verify_answer

# LangChain
docs = retriever.invoke(question)              # list[Document]
facts = facts_from_records(
    docs,
    key=lambda d: d.metadata["metric"],
    value=lambda d: d.metadata["value"],
    source=lambda d: d.metadata.get("source", "retrieved"),
    confidence=lambda d: d.metadata.get("score", 0.5),
)
report = verify_answer(llm_answer, facts)
```

LlamaIndex nodes, vector-store rows and API JSON map the same way. Runnable
end-to-end example: [`examples/from_retrieved_docs.py`](examples/from_retrieved_docs.py).

---

## Install

```bash
pip install verified-rag                # core, zero deps
pip install "verified-rag[llm]"         # + anthropic for NL claim extraction
```

(Or drop the `verified_rag/` folder into your project — it's self-contained.)

---

## API surface

| Function | Purpose |
|---|---|
| `verify_answer(answer, facts, ...)` | one call: reconcile → verify → cite |
| `facts_from_records(records, ...)` | map retrieved docs/rows → `Fact`s |
| `reconcile_facts(facts)` | cross-source consensus + conflicts |
| `extract_numeric_claims(text)` | pull numbers + period context out of text |
| `verify_claims_against_facts(claims, reconciled)` | period-aware matching |
| `inject_citations(text, reconciled)` | add inline `[src: ...]` tags |
| `compare_modes(answer, facts)` | quantify false citations avoided |

---

## How it compares

- **vs. LangChain / RAGFlow templates** — those retrieve; they don't verify the
  *output's* numbers against the retrieved facts, and they're period-blind.
- **vs. Ragas / faithfulness scorers** — those give you a score; this gives you
  the score **plus** the corrected, cited answer and the specific false claims.

---

## Free core, optional paid extras (open-core)

The whole library above is **MIT-licensed and free** — see [`LICENSE`](LICENSE).

Paid, optional, *not required* to use it:
- **Pro add-ons** — ready-made source adapters, tuned LLM claim extraction,
  faithfulness-report tooling + a CI gate, priority support.
- **Done-for-you** — I wire this into your stack with a before/after report on
  your real data. Contact me to discuss scope.

Details: [`LICENSE.md`](LICENSE.md).

---

⭐ If this saved you from a hallucinated number, star the repo — it helps others
find it.
