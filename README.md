# WIR_Retriever_Engine

Final project for **Web Information Retrieval** (WIR, Summer 2026 · Prof. Dr. Philipp Schaer, TH Köln).

Build and **evaluate** an IR system for **scientific web search** on the **LongEval-Sci**
collection, compare it against the **TF-IDF and BM25 baselines**, and submit it to **TIRA**.
See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the full plan and the reasoning behind each script.

## Project layout

```
WIR_Retriever_Engine/
├── pyproject.toml              # uv-managed dependencies
├── data/                       # ai.json, downloaded snapshots (gitignored)
├── index/                      # built Terrier indices + facet_df.json (gitignored — large!)
├── notebooks/
│   ├── 01_intro_smalldata.ipynb     # Phase A — TF/TF-IDF on small ai.json
│   ├── 02_index_longeval.ipynb      # Phase B — index LongEval-Sci
│   └── 03_retrieve_evaluate.ipynb   # Phase C — BM25 vs TF-IDF, evaluated
├── src/
│   ├── config.py               # single source of truth (paths, dataset id, knobs)
│   ├── inspect_data.py         # STEP 0 — inspect document fields / facets
│   ├── build_index.py          # STEP 1 — build the LongEval-Sci index + facet-DF map
│   ├── run_baselines.py        # baseline ladder: TF / TF-IDF / BM25 / tuned / RM3
│   ├── powerlaw_boost.py       # informetric EF×IDF re-ranker (authors, pubyear control)
│   ├── evaluate.py             # STEP 2 — pt.Experiment metrics + significance tests
│   └── make_submission.py      # STEP 3 — TREC run files + ir-metadata.yml for TIRA
└── runs/                       # TREC run files + ir-metadata.yml for TIRA
    ├── snapshot-1/  snapshot-2/  snapshot-3/
    └── ir-metadata.yml
```

## Setup

PyTerrier runs on the JVM, so a **JDK 11+** must be on your machine.

> ✅ **Environment is already provisioned** — `uv sync` created `.venv/` (CPython 3.12),
> installed the full stack, and the `wir-2026` Jupyter kernel is registered. Verified:
> PyTerrier 1.0.1 on Terrier 5.11, JDK 17 (Temurin), JVM starts cleanly.

To reproduce on another machine:

```bash
java -version          # if missing: install Temurin/OpenJDK 11 or 17

uv sync                # creates .venv/ and installs the pinned stack (uv.lock)
uv run python -m ipykernel install --user --name wir-2026   # register Jupyter kernel
```

Run any script through the env with `uv run`, and select the **wir-2026** kernel in Jupyter.

## Workflow

Run the scripts in order (all paths/knobs live in `src/config.py`):

```bash
uv run python src/inspect_data.py      # STEP 0 — confirm document fields / facets
uv run python src/build_index.py       # STEP 1 — build the index + facet_df.json (~10 min, 870k docs)
uv run python src/evaluate.py          # STEP 2 — nDCG@10-led table + paired significance tests
uv run python src/make_submission.py   # STEP 3 — write runs/snapshot-*/run.txt.gz + ir-metadata.yml
```

`run_baselines.py` and `powerlaw_boost.py` are libraries imported by `evaluate.py`
and `make_submission.py` — you don't run them directly. After STEP 3, fill in every
`ENTER_VALUE_HERE` in `runs/ir-metadata.yml` and upload to TIRA.

> **Note:** `make_submission.py` reuses the training index for snapshot-1 but builds a
> fresh index for snapshot-2 and snapshot-3 (they are separate document collections),
> so the first full run indexes those two snapshots (~10 min each).

Or work through the three notebooks (`notebooks/01–03`) for the guided, Phase-A→C version.
