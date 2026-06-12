# WIR_Retriever_Engine

Final project for **Web Information Retrieval** (WIR, Summer 2026 · Prof. Philipp Schaer, TH Köln).

Build and **evaluate** an IR system for **scientific web search** on the **LongEval-Sci**
collection, compare it against the **TF-IDF and BM25 baselines**, and submit it to **TIRA**.
See [WIR_SearchEngine_Roadmap.md](WIR_SearchEngine_Roadmap.md) for the full plan.

## Project layout

```
WIR_Retriever_Engine/
├── pyproject.toml              # uv-managed dependencies
├── data/                       # ai.json, downloaded snapshots (gitignored)
├── index/                      # built Terrier indices (gitignored — large!)
├── notebooks/
│   ├── 01_intro_smalldata.ipynb     # Phase A — TF/TF-IDF on small ai.json
│   ├── 02_index_longeval.ipynb      # Phase B — index LongEval-Sci
│   └── 03_retrieve_evaluate.ipynb   # Phase C — BM25 vs TF-IDF, evaluated
├── src/
│   ├── build_index.py          # build the LongEval-Sci index
│   ├── run_baselines.py        # TF/TF-IDF/BM25 → TREC run files
│   └── evaluate.py             # pt.Experiment metrics + significance tests
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

1. **Phase A** — `notebooks/01_intro_smalldata.ipynb` on `data/ai.json`.
2. **Phase B** — build the index: `uv run python src/build_index.py` (or run notebook 02).
3. **Phase C** — baselines + evaluation:
   ```bash
   uv run python src/run_baselines.py   # writes TREC runs into runs/snapshot-1/
   uv run python src/evaluate.py        # prints the nDCG@10-led comparison table
   ```
4. **Submit** — fill `runs/ir-metadata.yml`, upload to TIRA.

### Milestones
- **26 June** — valid TIRA run beating the TF baseline.
- **10 July** — final system submitted; ≤2-page report (lead with nDCG@10).
