# Full Evaluation Suite — Results and Analysis

**Date:** 2026-02-23  
**Environment:** Python 3.14, CUDA (sentence-transformers + Chroma on GPU), local LLM (TinyLlama) for RAG eval.

This document summarizes the results of the Medicare RAG evaluation suite: unit tests, index validation, retrieval evaluation, and full RAG chain evaluation.

---

## 1. Unit tests (pytest)

| Metric | Value |
|--------|--------|
| **Command** | `pytest tests/ -v --tb=short` |
| **Result** | **393 passed** |
| **Time** | ~23.2 s |
| **Warnings** | 2 (Pydantic V1 / Python 3.14; Chroma asyncio deprecation) |

All tests in `tests/` passed. The two warnings are non-blocking: LangChain’s Pydantic V1 usage is incompatible with Python 3.14+, and Chroma uses a deprecated asyncio API.

---

## 2. Index validation and retrieval evaluation

**Command:** `python scripts/validate_and_eval.py --json`

### 2.1 Index validation

Validation runs checks on the Chroma index (structure, metadata, sources, search, filters).

| Check | Passed | Detail |
|-------|--------|--------|
| chroma_dir_exists | ✓ | `data/chroma` |
| collection_accessible | ✓ | `medicare` (default domain) |
| collection_non_empty | ✓ | 28,470 docs |
| bulk_metadata_fetch | ✓ | 28,470 docs |
| metadata_key_doc_id | ✓ | 100% |
| metadata_key_content_hash | ✓ | 100% |
| metadata_key_source | ✓ | present |
| source_iom_present | ✓ | 16,976 |
| source_mcd_present | ✓ | 11,481 |
| **source_codes_present** | **✗** | **0** (no HCPCS/ICD codes ingested) |
| no_empty_documents | ✓ | 0 empty |
| no_duplicate_ids | ✓ | all unique |
| no_duplicate_content_hashes | ✓ | all unique |
| similarity_search_* | ✓ | generic, claims, codes queries return results |
| embedding_dimension | ✓ | 384 |
| filter_retrieval/correctness (iom, mcd) | ✓ | 2 results each, all match filter |

**Overall validation:** **Failed** due to a single check: **source_codes_present**. The index currently has no `source=codes` documents (HCPCS/ICD code files were not ingested in this run). All other 20 checks passed.

**Index stats:**

- **Total documents:** 28,470  
- **Source distribution:** IOM 16,976, MCD 11,481, codes 0  
- **Content length:** min 8, median 884, mean ~667, p95 1,142 chars  
- **Embedding dimension:** 384  

### 2.2 Retrieval evaluation (top-k = 5)

Eval set: **79 questions** from `scripts/eval_questions.json` (categories: policy_coverage, claims_billing, code_lookup, lcd_policy, appeals, consistency, etc.).

#### Aggregate metrics

| Metric | Value |
|--------|--------|
| **Hit rate** | **82.3%** (65 / 79 questions with ≥1 relevant doc in top-5) |
| **MRR** | **0.576** |
| **Avg precision@5** | 0.539 |
| **Avg recall@5** | 0.707 |
| **Avg NDCG@5** | 0.908 |
| **Latency (median)** | 410 ms |
| **Latency (p95)** | 702 ms |
| **Latency (p99)** | 900 ms |

#### By category (selected)

| Category | n | Hit rate | MRR | Avg P@5 | Avg R@5 | Avg NDCG@5 |
|----------|---|----------|-----|---------|---------|------------|
| claims_billing | 6 | 100% | 1.00 | 0.97 | 1.00 | 0.97 |
| coding_modifiers | 5 | 100% | 1.00 | 0.96 | 0.50 | 1.00 |
| compliance | 3 | 100% | 0.57 | 0.40 | 1.00 | 0.88 |
| consistency | 6 | 100% | 0.67 | 0.70 | 0.83 | 0.91 |
| cross_source | 4 | 100% | 0.69 | 0.75 | 0.67 | 0.92 |
| policy_coverage | 11 | 90.9% | 0.62 | 0.58 | 0.86 | 0.92 |
| summary_retrieval | 5 | 100% | 0.55 | 0.68 | 0.83 | 0.87 |
| lcd_policy | 8 | 75.0% | 0.36 | 0.40 | 0.69 | 0.93 |
| semantic_retrieval | 5 | 80.0% | 0.50 | 0.28 | 0.80 | 0.93 |
| appeals_denials | 5 | 60.0% | 0.35 | 0.28 | 0.60 | 0.98 |
| **code_lookup** | **7** | **0%** | **0.00** | **0.00** | **0.00** | **0.64** |
| edge_case | 4 | 75.0% | 0.63 | 0.50 | 0.50 | 0.93 |

**Observation:** `code_lookup` has 0% hit rate and 0 MRR because the index has no `codes` source; those questions expect HCPCS/ICD content.

#### By difficulty

| Difficulty | n | Hit rate | MRR | Avg P@5 | Avg R@5 | Avg NDCG@5 |
|------------|---|----------|-----|---------|---------|------------|
| easy | 9 | 66.7% | 0.61 | 0.44 | 0.57 | 0.77 |
| medium | 51 | 86.3% | 0.57 | 0.57 | 0.74 | 0.92 |
| hard | 19 | 78.9% | 0.56 | 0.49 | 0.68 | 0.94 |

Medium-difficulty questions perform best on hit rate and precision; easy questions have lower recall in top-5, possibly due to broader or more generic queries.

#### By expected source

| Expected source | n | Hit rate | MRR | Avg P@5 | Avg R@5 |
|-----------------|---|----------|-----|---------|---------|
| iom | 66 | 92.4% | 0.67 | 0.62 | 0.79 |
| mcd | 27 | 88.9% | 0.54 | 0.57 | 0.70 |
| codes | 19 | 63.2% | 0.50 | 0.48 | 0.33 |

IOM-heavy questions perform best (92.4% hit rate, 0.67 MRR). Codes-heavy questions are limited by the absence of codes in the index; the 63.2% hit rate reflects relevance from IOM/MCD text that mentions codes.

#### Query consistency (rephrased pairs)

Consistency is measured as Jaccard overlap of retrieved doc IDs for paraphrased questions in the same group.

| Group | Consistency score (Jaccard) |
|-------|-----------------------------|
| cardiac_rehab | 0.43 |
| wheelchair | 0.67 |
| hospice | 0.50 |
| **Overall avg** | **0.53** |

Rephrased questions often retrieve overlapping but not identical sets; there is room to improve consistency (e.g., query expansion or hybrid search tuning).

---

## 3. Analysis and recommendations

### Strengths

- **High overall hit rate (82.3%)** and **strong NDCG@5 (0.91)** indicate that for most questions at least one relevant document appears in the top 5 and ranking is good.
- **IOM and MCD coverage:** Hit rates for IOM (92.4%) and MCD (88.9%) expected-source questions are strong.
- **Claims/billing and modifiers:** 100% hit rate and MRR for claims_billing and coding_modifiers.
- **Summary retrieval:** 100% hit rate for summary_retrieval; topic/document summaries are being retrieved as intended.
- **Latency:** Median retrieval ~410 ms, p95 ~700 ms — acceptable for interactive use.

### Weak spots

1. **No codes source:** Validation fails `source_codes_present`, and **code_lookup** has 0% hit rate. Ingesting HCPCS/ICD code files (e.g. `scripts/download_all.py --source codes` then re-running ingest) would fix this and improve code-heavy questions.
2. **Code-heavy questions:** 19 questions expect `codes`; until codes are indexed, their metrics will stay low or rely on IOM/MCD text.
3. **Appeals/denials:** 60% hit rate, 0.35 MRR — worth checking if eval labels and/or content coverage for appeals are sufficient.
4. **LCD policy:** 75% hit rate, 0.36 MRR — some LCD-specific queries may need better targeting (e.g., LCD-aware expansion already in place; could tune k or filters).
5. **Consistency:** Average 0.53 Jaccard — rephrased queries don’t always return the same set; consider hybrid search, expansion, or RRF tuning to improve stability.

### Recommended next steps

1. **Ingest codes:** Run download + ingest for `codes` so that code_lookup and codes-expected metrics are meaningful.
2. **Re-run validation:** After codes ingest, re-run `validate_and_eval.py`; `source_codes_present` should pass.
3. **Regression baseline:** Use `--save-baseline scripts/eval_baseline.json` after a “good” run, then `--baseline scripts/eval_baseline.json` in CI to guard against retrieval regressions.
4. **RAG report:** Review `data/rag_eval_report.md` for answer quality and citation accuracy (manual assessment); consider prompt improvements or a larger model to increase citations and reduce garbled output.

---

## 4. Full RAG chain evaluation

**Command:** `python scripts/run_rag_eval.py --out data/rag_eval_report.md`  
**Output:** `data/rag_eval_report.md` ✓ (completed)

The script ran all 79 questions through the full RAG pipeline (retriever k=8 + local LLM TinyLlama). The report includes an **Answer Quality Summary** (automated heuristics) and per-question answers, quality metrics, and cited sources.

### 4.1 RAG answer quality summary (automated heuristics)

| Metric | Value |
|--------|--------|
| Questions | 79 |
| **Avg keyword coverage** | **39.0%** |
| **Avg citation count** | **0.1** |
| **% answers with citations** | **5%** |
| Avg repetition ratio | 0.93 (1.0 = no repetition) |
| Avg answer length | 913 chars |

### 4.2 RAG analysis

- **Keyword coverage (39%):** The model often includes expected terms when it produces coherent answers, but many answers are short, generic, or garbled, which lowers average coverage.
- **Citations (5% with any, 0.1 avg count):** The local LLM (TinyLlama) almost never emits `[1]`-style citations despite being given source chunks. This is a **generation/prompting** issue, not retrieval: the retriever supplies 8 relevant docs per question (retrieval hit rate 82.3%), but the model is not instructed or capable of consistently citing them.
- **Repetition (0.93):** Slight repetition in some answers; overall acceptable.
- **Answer length (913 chars):** Variable; some answers are one paragraph, others are truncated or corrupted by model artifacts (e.g. `<|...|>` tokens, stray “.net”, “Can you.”). Manual review in `data/rag_eval_report.md` shows a mix of good answers (e.g. Part B/Part A coverage, modifiers) and poor ones (garbled text, no citations).

**Conclusion:** Retrieval is strong (see §2); the main gap is **answer quality and citation behavior** of the small local LLM. To improve RAG output: (1) add explicit citation instructions and few-shot examples in the prompt, (2) consider a larger or instruction-tuned model, or (3) use a hosted API model for production. The detailed report in `data/rag_eval_report.md` is the right place for manual assessment of relevance, hallucinations, and citation accuracy per question.

---

## 5. How to reproduce

```bash
# From project root, with venv activated
pip install -e ".[dev]"

# Unit tests
pytest tests/ -v --tb=short

# Index validation + retrieval eval (human-readable)
python scripts/validate_and_eval.py

# Machine-readable metrics
python scripts/validate_and_eval.py --json

# Full RAG chain report (writes data/rag_eval_report.md)
python scripts/run_rag_eval.py --out data/rag_eval_report.md
```

Optional: `--eval-only`, `--validate-only`, `-k 10`, `--filter-source iom`, `--k-values 1,3,5,10`, and baseline options — see `scripts/validate_and_eval.py` docstring.
