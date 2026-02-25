# Topic definitions

Topic definitions drive **topic clustering** during ingest and **topic-summary boosting** at retrieval time. Chunks are assigned to topics via keyword patterns; topic-cluster summaries and document-level summaries are then generated and indexed as stable "anchor" chunks to improve retrieval consistency when users rephrase questions.

## Schema

Each topic is an object in a JSON array with these fields:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Internal identifier (e.g. `cardiac_rehab`). Used in metadata as `topic_clusters`. |
| `label` | No | string | Human-readable label (e.g. "Cardiac Rehabilitation"). Defaults to `name` if omitted. |
| `patterns` | Yes | array of strings | List of regex patterns (case-insensitive). A chunk is assigned to this topic if it matches at least `min_pattern_matches` of these. |
| `summary_prefix` | No | string | Prefix used when generating topic summary content (e.g. "Cardiac Rehabilitation: "). Defaults to `""`. |
| `min_pattern_matches` | No | integer | Minimum number of patterns that must match for the chunk to be assigned to this topic. Defaults to `1`. |

Patterns are compiled with `re.IGNORECASE`. A chunk may belong to multiple topics (e.g. "cardiac rehab billing codes" can match both `cardiac_rehab` and a billing-related topic if defined).

## Where the file lives

1. **Override (recommended for custom topics):**  
   `{DATA_DIR}/topic_definitions.json`  
   If this path exists, it is used. `DATA_DIR` is set in `.env` or defaults to the repo `data/` directory.

2. **Package default:**  
   `src/insurance_rag/data/topic_definitions.json`  
   Used when the file above does not exist. Shipped with the package.

To add or change topics without editing the package, create or edit `data/topic_definitions.json` (after at least one run that creates `data/`).

## Example definition

```json
{
  "name": "cardiac_rehab",
  "label": "Cardiac Rehabilitation",
  "patterns": [
    "cardiac\\s*rehab",
    "heart\\s*rehab",
    "cardiovascular\\s*rehab",
    "\\bICR\\b.*program"
  ],
  "summary_prefix": "Cardiac Rehabilitation: ",
  "min_pattern_matches": 1
}
```

- Use `\\b` for word boundaries (e.g. `\\bDME\\b` to avoid matching inside longer words).
- Use `\\s*` for optional whitespace between words (e.g. `wound\\s*care`).
- `min_pattern_matches: 2` is useful for broad topics (e.g. DME) so that only chunks with at least two distinct pattern matches are tagged, reducing false positives.

## How topics affect retrieval

1. **Ingest:**  
   `cluster.py` assigns `topic_clusters` metadata to each chunk based on pattern matches. The summarization step then builds **document-level summaries** and **topic-cluster summaries** (one per topic that has enough chunks). These summary documents are indexed alongside normal chunks.

2. **Retrieval:**  
   The retriever (LCD-aware or hybrid) can **detect query topics** using the same pattern set, then **inject and boost** topic-summary and document-summary chunks in the result list so that stable anchor content appears even when the user rephrases the question.

Adding or refining topics improves consistency for those clinical/policy areas; avoid overly broad patterns that tag too many unrelated chunks.

## Adding a new topic

1. Choose a `name` (snake_case) and optional `label`.
2. Define regex `patterns` that appear in chunks you want to group (e.g. procedure names, acronyms, phrases).
3. Set `summary_prefix` if you want topic summaries to start with a standard heading.
4. Set `min_pattern_matches` (default 1) — use 2 or more for broad topics to reduce noise.
5. Add the object to `data/topic_definitions.json` (or to the package default if you are editing the package).
6. Re-run ingest so clustering and summarization run again:  
   `python scripts/ingest_all.py --source all`  
   (Summaries are controlled by `ENABLE_TOPIC_SUMMARIES`; use `--no-summaries` to disable.)

No code changes are required; the pipeline loads the JSON at runtime.
