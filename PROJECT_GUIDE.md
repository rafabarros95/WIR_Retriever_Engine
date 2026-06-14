# WIR Retriever Engine — Project Guide

**Web Information Retrieval (WIR), Summer 2026 · Prof. Dr. Philipp Schaer · TH Köln**
**Assignment 3 + 4** — build and evaluate an IR system for scientific search on **LongEval‑Sci**.

This guide documents the code in `src/` and the reasoning behind each suggestion.



## 1. The one‑sentence goal

> Build and **evaluate** a retrieval system for scientific web search on **LongEval‑Sci** (CORE papers), beat the **TF‑IDF / BM25 baselines**, and write up what we found.

This is a real entry into the **CLEF 2026 LongEval‑Sci: Ad‑Hoc Scientific Retrieval** shared task.


## 2. Setup

PyTerrier runs on the JVM, so we need **JDK 11+** ( README confirms Temurin 17 is installed). Dependencies via `uv`, but you guys can use another one like `pip`: 

```bash
uv add python-terrier ir-datasets ir-datasets-longeval pandas scipy numpy
# optional, only if we would like to try the neural re-ranking fallback, let's keep that in mind:
# uv add pyterrier-t5
```

Run anything through the env with `uv run python src/<script>.py`.

All knobs live in **`src/config.py`** — dataset id, index path, stemmer/stopwords, which metadata fields to store, eval metrics. Change them there, not in five files.

- I was thinking of applying SnowballStemmer and Porterstemmer while text pre-processing to check if performance significantly vary.


## 3. Step‑by‑step

### Step 1 — Inspect the data *(`src/inspect_data.py`)*

```bash
uv run python src/inspect_data.py
```

Prints the document fields, a sample doc, sample queries and qrels, and a verdict.
**This is a gate:** the power‑law boost only works if the documents actually expose `author` / `subject`. CORE records usually do (the LongEval‑Sci docs come from CORE, whose UI shows author names), but confirm it here and update the exact field names in `config.META_FIELDS` to match what prints.

- Facets present → continue with the boost (our distinctive angle).
- Only `text` present → skip the boost, use neural re‑ranking instead.

### Step 2 — Build the index *(`src/build_index.py`)*

```bash
uv run python src/build_index.py
```

Builds a Terrier index that **also stores** the isness fields (author/subject/pubyear) so they're available at re‑rank time, and writes a **facet document‑frequency map** (`data/facet_df.json`) — the IDF ingredient for the boost. Verifies the build with `getCollectionStatistics()`; don't trust a silent indexer.

> **Blessed shortcut:** if indexing ~870k docs is too heavy, load the prebuilt index the organisers publish: `pt.Artifact.from_hf("jueri/longeval-2026-snapshot-1-index")`. However, let's build our own for boosting.

### Step 3 — Evaluate *(`src/evaluate.py`)*

```bash
uv run python src/evaluate.py
```

Builds every system, runs one `pt.Experiment` so the numbers are comparable, and prints:
1. a **metrics table** led by **nDCG@10** (+ MAP, MRR, Recall@1000), with Bonferroni‑corrected deltas vs BM25 — *this table is your report's core*;
2. **paired significance tests** (Student's t + Wilcoxon) per system vs BM25.

### Step 4 — Submit *(`src/make_submission.py`)*

```bash
uv run python src/make_submission.py
```

Runs your chosen system on each test snapshot, writes `runs/snapshot-{1,2,3}/run.txt.gz`, and an `ir-metadata.yml` skeleton. Fill in every `ENTER_VALUE_HERE`, then upload to TIRA.

> **Confirm the format first.** The `longeval-code/clef26/scientific-retrieval` repo documents per‑snapshot `run.txt.gz` + `ir-metadata.yml`; the task page also mentions a TREC‑RAG `jsonl` variant. Check which the 2026 task expects on TIRA before uploading.

---

## 4. The systems implemented (the ideas, in code)

### Baseline ladder *(`src/run_baselines.py`)* — what we must beat

| System | What it adds | Why |
|---|---|---|
| `TF` | raw term frequency | weakest — no normalisation |
| `TF-IDF` | + IDF | rewards rare query terms |
| `BM25` | + doc‑length normalisation + tf saturation | the real baseline |
| `BM25-tuned` | grid‑search `b`, `k1` | **improvement #1**, lowest effort |
| `BM25+RM3` | pseudo‑relevance query expansion | **improvement #2**, it was used by LongEval 2025 teams |

BM25 should beat TF‑IDF should beat TF — we gotta explain that ordering (document‑length normalisation) in our report.

### Distinctive angle *(`src/powerlaw_boost.py`)* — the informetric boost

A PyTerrier re‑ranker that runs after BM25 and implements the idea as an **EF × IDF** boost over a metadata field:

- **EF** (element frequency): how often a facet value (an author, a subject term) appears among the documents retrieved for *this* query — a signal of what the query is about in author/subject space.
- **IDF** (across the whole collection): a *rare* author/subject is more informative than a common one (We learnt that a rare shared citation matters more than a common one).
- `final_score = normalise(bm25) + λ · normalise(EF×IDF boost)`, then re‑rank.

Instantiated in `evaluate.py`:

- `BM25+PL[authors]` — expected to **help**.
- `BM25+PL[pubyear]*` — **negative control**, expected to **hurt**.

> **Data reality (verified against the installed loader, 2026‑06‑13):** the
> LongEval‑Sci 2026 `LongEvalSciDoc` exposes `doc_id, title, abstract, authors,
> createdDate, doi, arxivId, pubmedId, magId, oaiIds, links, publishedDate,
> updatedDate`. `authors` is a list of `{"name": "Last,First"}` dicts. There is
> **no subject/keyword field**, so the slide's `subject` facet is not available
> here — we use `authors` as the positive facet and derive `pubyear` from
> `publishedDate` for the negative control. Run `inspect_data.py` to confirm.

Tune `λ` (0.1–0.5) on training data.

---

## 5. Stage 4 — hypotheses & error analysis

Phrase hypotheses so they're falsifiable, and back them with the paired tests from `evaluate.py` (aim for ≥50 queries; Bonferroni if many comparisons):

1. *"Re‑ranking BM25 with an author power‑law boost significantly improves nDCG@10 on LongEval‑Sci (paired t‑test, p < 0.05)."*
2. *"Boosting by publication year does **not** improve nDCG@10"* — the built‑in negative control;

For error analysis, sort per‑query nDCG@10 differences (our system − BM25) and inspect the worst topics: vocabulary mismatch? short query? sparse author metadata?

---

## 6. Sources (verified 2026‑06‑13)

- Course slides WIR‑02 … WIR‑10 (project files) — Cranfield, BM25, power laws, the lab stages.
- LongEval‑Sci loader & snapshots — `github.com/clef-longeval/ir-datasets-longeval`.
- Official PyTerrier baseline + prebuilt HF index — `github.com/clef-longeval/longeval-code/tree/main/clef26/scientific-retrieval`.
- Task description (CORE data, TIRA) — `clef-longeval.github.io/tasks/`.
- PyTerrier API (Retriever, IterDictIndexer, Experiment, GridSearch, `pt.apply`) — `pyterrier.readthedocs.io`.
- Teaching tutorials & component dashboard — `github.com/tira-io/teaching-ir-with-shared-tasks/tree/main/tutorials` and `tira-io.github.io/teaching-ir-with-shared-tasks`.
- LongEval prior years — CEUR Vol‑4038 (2025), Vol‑3740 (2024).


## Flow:

Run order: `inspect_data.py` → `build_index.py` → `evaluate.py` → `make_submission.py`.
