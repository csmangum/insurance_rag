# Evaluation questions schema and usage

The retrieval and RAG evaluation pipeline is driven by a JSON file of questions with expected keywords and sources. This document describes the schema, how to add questions, and how to run and interpret the evaluation.

## Eval file location

- **Canonical file:** `scripts/eval_questions.json`
- **Custom file:** Pass the path to `validate_and_eval.py` or `run_rag_eval.py` (see below).

## Question schema

Each question is an object with the following fields.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | Yes | string | Unique identifier (no duplicates in the file). |
| `query` | Yes | string | The natural-language question used for retrieval. |
| `expected_keywords` | Yes | array of strings | Keywords that should appear in at least one relevant retrieved document. Used to compute relevance (hit, precision, MRR). |
| `expected_sources` | Yes | array of strings | Source types that are considered relevant: `"iom"`, `"mcd"`, `"codes"`. A document is fully relevant only if it comes from one of these and matches keyword expectations. |
| `category` | Yes | string | Category for breakdown (e.g. `policy_coverage`, `code_lookup`). Must be present; the test suite requires at least 5 categories across the set. |
| `difficulty` | Yes | string | Difficulty level for breakdown: `"easy"`, `"medium"`, or `"hard"`. The suite requires at least 2 difficulty levels. |
| `description` | No | string | Human-readable description of what the question is testing. |
| `consistency_group` | No | string | If set, questions sharing the same value are used for consistency evaluation (rephrased pairs); each group must have at least 2 questions. |

### Categories in use

The current `eval_questions.json` uses categories such as:

- `policy_coverage`, `claims_billing`, `coding_modifiers`, `code_lookup`, `lcd_policy`
- `appeals_denials`, `cross_source`, `semantic_retrieval`, `abbreviation`, `edge_case`
- `consistency`, `payment`, `compliance`

You can add new category names; the evaluation will report metrics by category.

### Expected sources

- `iom` — Internet-Only Manuals (100-02, 100-03, 100-04)
- `mcd` — Medicare Coverage Database (LCDs, NCDs, Articles)
- `codes` — HCPCS and/or ICD-10-CM code documents

A question can list multiple sources (e.g. `["iom", "codes"]`) for cross-source questions.

## Example question

```json
{
  "id": "part_b_coverage",
  "query": "What does Medicare Part B cover?",
  "category": "policy_coverage",
  "difficulty": "easy",
  "expected_keywords": ["Part B", "outpatient", "medical", "coverage"],
  "expected_sources": ["iom"],
  "description": "Basic Part B coverage — should match IOM 100-02 content"
}
```

## Running evaluation

### Index validation and retrieval eval

```bash
# Validate index and run retrieval eval (default: scripts/eval_questions.json, k=5)
python scripts/validate_and_eval.py

# Validation only
python scripts/validate_and_eval.py --validate-only

# Retrieval eval only, custom k
python scripts/validate_and_eval.py --eval-only -k 10

# Metrics as JSON (e.g. for CI)
python scripts/validate_and_eval.py --eval-only --json

# Write markdown report
python scripts/validate_and_eval.py --eval-only --report data/eval_report.md
```

### Full RAG eval (answer quality)

Runs the RAG chain (retriever + LLM) on each question and writes a report for manual review:

```bash
python scripts/run_rag_eval.py --eval-file scripts/eval_questions.json --out data/rag_eval_report.md -k 8
```

## Metrics explained

| Metric | Meaning |
|--------|--------|
| **Hit rate** | Fraction of questions where at least one fully relevant document appears in the top-k results. |
| **MRR** (mean reciprocal rank) | Average of 1/rank of the first relevant document (1.0 = first result is relevant). |
| **Precision@k** | Fraction of the top-k results that are fully relevant. |
| **Recall@k** | Fraction of all relevant documents for that question that appear in the top-k (when relevance set is defined). |
| **NDCG@k** | Normalized discounted cumulative gain; rewards relevant docs ranked higher. |

Relevance is determined by: document source in `expected_sources` and document content containing enough of the `expected_keywords` (see the evaluation implementation in the codebase for exact matching rules).

## Adding or editing questions

1. Open `scripts/eval_questions.json`.
2. Add a new object with unique `id`, `query`, `expected_keywords`, `expected_sources`, `category`, and `difficulty`.
3. Ensure the file is valid JSON (no trailing commas).
4. Run the test suite: `pytest tests/test_search_validation.py -v` — it checks for duplicate IDs, required fields, and minimum variety (categories, difficulties, consistency groups).
