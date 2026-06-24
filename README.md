# WIR_Retriever_Engine — project report

Lab project for **Web Information Retrieval** (WIR, Summer 2026 · Prof. Dr. Philipp
Schaer · TH Köln). We build and **evaluate** an IR system for **scientific web search**
on the **LongEval-Sci** collection (CLEF 2026, Task 1 — Ad-Hoc Scientific Retrieval),
compare it against TF / TF-IDF / BM25, and test research hypotheses with neural
re-ranking.

This README is a **full report of the project** — every step, the tools used, the
approaches taken, and the results — so it doubles as the lab logbook. The whole project
is **notebook-only** (no `src/` package): each numbered notebook in
[`notebooks/`](notebooks/) is one self-contained stage and runs top-to-bottom on the
`wir-2026` Jupyter kernel. The notebooks extend the four official course tutorials at
[irgroup-classrooms/wir-2026](https://github.com/irgroup-classrooms/wir-2026)
(`pyterrier-intro`, `pyterrier-irdatasets-indexing`, `pyterrier-retrieval`,
`pyterrier-ltr`).

> 📄 The professor-facing **2-page preliminary report** is at
> [`paper/preliminary-report.md`](paper/preliminary-report.md). Course slides live in
> [`theory/`](theory/).

---

## 0. TL;DR — headline results

All numbers on `longeval-sci-2026/snapshot-1/train/dctr` (100 judged queries, 8,772
qrels, 869,902 docs); official metric **nDCG@10**.

| Finding | Result |
|---|---|
| **Best system** | **BM25 ≫ multilingual cross-encoder (interp.)** — nDCG@10 **0.324**, **+0.032 vs BM25, significant** (t *p*=0.032, Wilcoxon *p*=0.014) |
| Close 2nd (cheaper) | BM25 ≫ multilingual-dense bi-encoder (interp.) — 0.317, +0.025 (*p*=0.039) |
| Strong lexical baseline | BM25-tuned (`b=0.9, k1=1.2`) — 0.305 |
| Baseline order | `BM25 ≳ TF-IDF ≫ TF` (0.292 / 0.292 / 0.047) |
| Power-law author boost (our angle) | nDCG-neutral — **H1 rejected** (Δ+0.0008, *p*≈0.88) |
| Pub-year boost (negative control) | significant **drop** −0.091 (*p*<0.0001) — **H2 supported** |
| RM3 query expansion | hurt (−0.014) on this collection |
| Cross-lingual *lexical* (EN→ES translate+fuse) | **does not help** — best Spanish weight = 0 (relevant docs ~96 % English) |
| Cross-lingual *semantic* (dense bi-encoder) | **helps** — +0.025, significant; gain is general semantic matching, not specifically cross-lingual |
| Cross-lingual *semantic* (cross-encoder) | **best system** — +0.032; but only a tiny, *non-significant* step over the bi-encoder |

---

## 1. The lab, the assignments, and what this repo delivers

The lab has four stages (WIR-04 / WIR-10 exercise slides). This repo covers **Stage 3 +
Stage 4**:

| Assignment | Stage | Goal | Notebooks |
|---|---|---|---|
| **III** | Stage 3 | Build **and** evaluate an IR system | 02, 03 |
| **IV** | Stage 4 | Derive & test hypotheses; statistical + error analysis | 04 |
| (extension) | — | **Research question**: cross-lingual retrieval (lexical, dense & cross-encoder) + LTR | 05, 06, 07 |

> Stage-3 deliverable (WIR-10, due **26 June 2026**): a 2-page report **+ a system-demo
> Jupyter notebook**. Notebooks 03/06/07 are the system demo; 04 backs the Stage-4 report;
> 05–07 are the research idea for the paper ([`paper/preliminary-report.md`](paper/preliminary-report.md)).
> *(The team opted not to submit to TIRA; the former `05_tira_submission` notebook was removed.)*

---

## 2. Tools & environment (what we used and why)

| Tool | Version | Used for |
|---|---|---|
| **PyTerrier** | 1.0.1 (Terrier 5.11) | indexing, retrieval, experiments, LTR, fusion |
| **JDK** (Temurin) | 17 | Terrier runs on the JVM |
| Python | 3.12 (uv-managed `.venv`) | everything |
| **ir-datasets-longeval** | — | load LongEval-Sci docs / queries / qrels |
| pandas, numpy | — | data wrangling |
| **scipy** | — | paired significance tests (t-test, Wilcoxon) |
| matplotlib | — | per-query / sensitivity plots |
| nltk | — | Porter stemmer (IDF-by-hand demo) |
| **langdetect** | 1.0.9 | language audit of corpus + relevant docs (NB05) |
| **transformers + torch** | 5.12.1 / 2.12.1+cpu | MarianMT EN→ES query translation (NB05); neural re-rankers (NB06/07) |
| sentencepiece, sacremoses | — | MarianMT tokenisation (NB05) |
| **xgboost / lightgbm / scikit-learn** | 3.3.0 / — / — | LambdaMART learned re-ranking (NB05) |
| **sentence-transformers** | 5.6.0 | multilingual dense bi-encoder (NB06) + cross-encoder (NB07) |
| uv | — | dependency + venv management |
| nbformat / nbclient | — | notebooks authored + executed programmatically |

Models: **`Helsinki-NLP/opus-mt-en-es`** (neural EN→ES, NB05),
**`intfloat/multilingual-e5-small`** (multilingual dense bi-encoder, NB06), and
**`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`** (multilingual cross-encoder, NB07).

**Setup:**
```bash
java -version                 # need JDK 11+ (Temurin 17 used here)
uv sync                       # creates .venv/ and installs the stack
uv run python -m ipykernel install --user --name wir-2026
```
Open the notebooks and select the **`wir-2026`** kernel. On Windows + OneDrive, if uv
hits file-lock errors add `--link-mode=copy` (see §8).

---

## 3. Step-by-step walkthrough (the report)

### Notebook 01 — [`01_intro_smalldata.ipynb`](notebooks/01_intro_smalldata.ipynb) · Phase A
**Goal:** learn the PyTerrier machinery on the tiny `data/ai.json` (2,000 BibSonomy AI
records) before the real collection.
**Steps:** load + clean (drop >50 %-empty columns) → set `docno`/`text` → index with
`IterDictIndexer` → read the **lexicon** (term `Nt`/`TF`) → run `Tf` then `TF_IDF`
retrievers → recompute **IDF by hand** from the lexicon to demystify it.
**Result/insight:** index of 2,000 docs / 2,915 terms; `idf(robotics)=1.90 >
idf(intelligence)=0.30`, and `idf(the)=NaN` (Terrier dropped it as a stopword) — IDF
rewards rare terms; document length is still unhandled (BM25's job).

### Notebook 02 — [`02_inspect_and_index.ipynb`](notebooks/02_inspect_and_index.ipynb) · Phase B
**Goal:** inspect LongEval-Sci and build the real index.
**Steps:** load `snapshot-1/train/dctr` → inspect document fields, queries, qrels →
decide which informetric facets exist → build a Terrier index (Porter stemmer + Terrier
stopwords) that **also stores `authors`/`pubyear`/`title`** → in the same pass build a
**facet document-frequency map** `facet_df.json` (the IDF ingredient for the boost). The
build cell **reuses an existing index** unless `FORCE_REBUILD=True` (rebuild ≈ 10 min).
**Result:** **869,902 docs**, 1,529,555 terms. Facets: `authors` exists (a list of
`{"name":...}` dicts), **no** subject/keyword field. The author distribution is a
textbook **power law** — **2,696,925** distinct authors, **83 % appear in exactly one
document**. We use `authors` as the positive facet and `pubyear` as a negative control.

### Notebook 03 — [`03_systems_and_evaluation.ipynb`](notebooks/03_systems_and_evaluation.ipynb) · Assignment III
**Goal:** build the systems and produce the core metrics table.
**Approaches:** baseline ladder (TF, TF-IDF, BM25); **improvement #1** BM25-tuned via
`pt.GridSearch` over `b`,`k1`; **improvement #2** BM25+RM3 pseudo-relevance feedback;
**our distinctive angle** an informetric **EF×IDF power-law re-ranker** on `authors`
(`final = norm(bm25) + λ·norm(EF×IDF)`) plus the same on `pubyear` as a control. One
`pt.Experiment` ranks them by nDCG@10.

| System | nDCG@10 | MAP | Recall@1000 |
|---|---|---|---|
| **BM25-tuned** | **0.3050** | 0.2641 | 0.8598 |
| BM25 | 0.2922 | 0.2573 | 0.8581 |
| TF-IDF | 0.2921 | 0.2592 | 0.8575 |
| BM25+PL[authors] | 0.2885 | 0.2529 | 0.8581 |
| BM25+RM3 | 0.2781 | 0.2402 | 0.8701 |
| BM25+PL[pubyear]* | 0.2015 | 0.1734 | 0.8581 |
| TF | 0.0470 | 0.0379 | 0.6113 |

**Insight:** length normalisation matters most at the bottom (`TF` collapses); BM25 and
TF-IDF are nearly tied here; the biggest honest win is simply **tuning BM25**.

### Notebook 04 — [`04_hypotheses_and_error_analysis.ipynb`](notebooks/04_hypotheses_and_error_analysis.ipynb) · Assignment IV
**Goal:** test two falsifiable hypotheses with statistics, not single means.
**Approaches:** per-query nDCG@10 + paired **t-test** and **Wilcoxon** vs BM25; a
**λ-sensitivity sweep** for the author boost; per-query **error analysis** (best/worst
topics) with a plot.

- **H1** — *author power-law boost improves nDCG@10*: **rejected.** Δ=+0.0008, t-test
  *p*≈0.88, Wilcoxon *p*≈0.41; 23 wins / 21 losses / 56 unchanged; λ peaks at 0.2
  (0.2930) — never meaningfully above BM25. Mechanism: 83 % of authors are singletons,
  so the EF signal inside a result set is too sparse.
- **H2** — *pub-year boost does not help (control)*: **supported.** Δ=−0.0907,
  *p*<0.0001 — a large, significant drop, validating the experimental setup.

### Notebook 05 — [`05_crosslingual_ltr.ipynb`](notebooks/05_crosslingual_ltr.ipynb) · research question (lexical)
See §4 — the *lexical* cross-lingual study (translate EN→ES + weighted RRF fusion) +
**LambdaMART** learning-to-rank, extending the prof's `pyterrier-ltr` tutorial.

### Notebook 06 — [`06_dense_reranking.ipynb`](notebooks/06_dense_reranking.ipynb) · research question (semantic, bi-encoder)
See §4 — the *semantic* cross-lingual study: BM25 ≫ multilingual **dense bi-encoder**
(`multilingual-e5-small`). The first significant win (0.317).

### Notebook 07 — [`07_cross_encoder_reranking.ipynb`](notebooks/07_cross_encoder_reranking.ipynb) · research question (semantic, cross-encoder)
See §4 — BM25 ≫ multilingual **cross-encoder** (`mmarco-mMiniLMv2-L12-H384-v1`). **Our
best system (0.324) and the positive headline.**

---

## 4. Research question — cross-lingual retrieval (notebooks 05, 06 & 07)

> **On LongEval-Sci, does fusing monolingual (English) BM25 with a cross-lingual branch
> — queries machine-translated into Spanish, the collection's largest non-English
> language — improve nDCG@10 over monolingual BM25, and does any gain concentrate on the
> topics whose relevant set contains non-English papers, without harming the English-only
> majority?**

**Language audit (the motivation, measured with langdetect).**

| Where | English | Top non-English |
|---|---|---|
| Whole corpus (4k sample) | ~80 % | **es 6 % · pt 3 % · de 3 % · fr 2 %** |
| **Judged-relevant docs** (n=1,085) | **~96 %** | **es 2.9 %**, rest <1 % |
| Topics with ≥1 non-English relevant doc | **10 / 100** | — |

So the corpus is ~20 % non-English but the *relevant* set is ~96 % English — little
headroom for a cross-lingual branch on these judgments.

**Method:** translate the 100 queries EN→ES with **MarianMT**, **accent-fold** them to
match the index (the index folds `educación`→`educacion`), retrieve a **Spanish BM25**
branch on the same index, then combine via **weighted Reciprocal Rank Fusion** and via a
**LambdaMART** ranker that takes the ES-BM25 score as an extra feature.

**Results (answer: the cross-lingual branch does *not* help here).**

| System | nDCG@10 |
|---|---|
| BM25-tuned | 0.3050 |
| RRF best (ES weight = 0) ≈ BM25 | 0.2923 |
| BM25 | 0.2922 |
| RRF(EN+ES) naive (equal weight) | 0.1577 |
| BM25 [ES-only] | 0.0237 |

- The **ES-weight sweep is monotonically decreasing** — the best Spanish weight is **0**
  (i.e. ignore it). Naive equal-weight fusion roughly **halves** nDCG@10.
- **LambdaMART** (test split): BM25 0.3076 / LambdaMART 0.3093 / **LambdaMART+CL 0.2765**
  — the cross-lingual feature *hurts* the learned ranker too.
- **Subset analysis (decisive):** the Spanish branch significantly hurts both the 10
  topics *with* non-English relevant docs (Δ=−0.11, *p*=0.017) and the 90 English-only
  ones (Δ=−0.137, *p*<0.0001). The few Spanish relevant papers it surfaces do not
  outweigh the noise.

**Contribution for the paper:** the value is the **diagnosis** — a language audit that
quantifies the corpus-vs-relevant-set language mismatch and explains why an appealing
cross-lingual idea fails on these judgments, plus the conditions under which it would pay
off (a Spanish-analyzed index; or the group's own qrels, which may contain more
non-English relevant papers).

### The principled version — multilingual dense re-ranking (notebook 06)

Lexical translation forces *same-token* matching. A **multilingual dense** model instead
embeds query and document into a **shared semantic space**, so cross-lingual (and synonym)
matches work without translation. We re-rank BM25's top-50 with
`intfloat/multilingual-e5-small`.

| System | nDCG@10 | vs BM25 |
|---|---|---|
| **BM25 ≫ dense (interp. 0.5/0.5)** | **0.3167** | **+0.0245, significant** (t-test *p*=0.039) |
| BM25-tuned | 0.3050 | +0.013 |
| BM25 | 0.2922 | — |
| BM25 ≫ dense (pure) | 0.2602 | −0.032 (n.s.) |

- **H (dense improves nDCG@10): supported** — the interpolated re-ranker beats BM25
  significantly (38 wins / 25 losses).
- **Blend, don't replace:** *pure* dense re-ranking **hurts** (e5-small alone is weaker
  than BM25 lexically); only `0.5·BM25 + 0.5·dense` wins — lexical + semantic are
  complementary.
- **Honest twist:** the gain is **not** concentrated on the cross-lingual topics — it is
  larger and significant on the English-only topics (Δ=+0.0267, *p*=0.040) and negligible
  on the 10 cross-lingual ones (Δ=+0.0049, *p*=0.82). So the win is **general semantic
  matching**, not provably cross-lingual recovery on these qrels (the subset is small and
  already English-covered).

### The strongest version — multilingual cross-encoder re-ranking (notebook 07)

A **cross-encoder** scores each (query, document) pair *jointly* (full cross-attention) —
the strongest re-ranking family. We re-rank the same BM25 top-50 with the multilingual
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (trained on mMARCO, 15 languages).

| System | nDCG@10 | vs BM25 |
|---|---|---|
| **BM25 ≫ CE (interp. 0.5/0.5)** | **0.3242** | **+0.0320, significant** (t *p*=0.032, Wilcoxon *p*=0.014) |
| BM25 ≫ dense (interp.) | 0.3167 | +0.0245, significant |
| BM25 | 0.2922 | — |
| BM25 ≫ CE (pure) | 0.2654 | −0.027 (n.s.) |

- **Best system overall** (0.3242), and a more robust win over BM25 than the bi-encoder
  (41 wins / 27 losses, Wilcoxon *p*=0.014).
- **But cross-encoder ≈ bi-encoder:** the CE beats e5 by only +0.0074, **not significant**
  (*p*=0.41) — the much heavier model barely improves on the tiny e5-small.
- **Same honest twist:** the gain is on English-only topics (Δ=+0.0350, *p*=0.033) and
  negligible on the 10 cross-lingual ones (Δ=+0.0050, *p*=0.84). Scaling the model up does
  **not** move the cross-lingual subset.

**Paper arc:** language audit → *lexical* cross-lingual fusion fails & why (NB05) →
*dense bi-encoder* gives a significant gain (NB06) → *cross-encoder* is the project best
(NB07). Problem → naive fix fails → principled fix works (and a bigger model only nudges it).

---

## 5. Project layout

```
WIR_Retriever_Engine/
├── pyproject.toml            # uv-managed deps (notebooks-only, not a package)
├── README.md                 # this report
├── notebooks/                # the entire project — run 01 → 07 in order
│   ├── 01_intro_smalldata.ipynb
│   ├── 02_inspect_and_index.ipynb
│   ├── 03_systems_and_evaluation.ipynb
│   ├── 04_hypotheses_and_error_analysis.ipynb
│   ├── 05_crosslingual_ltr.ipynb
│   ├── 06_dense_reranking.ipynb
│   └── 07_cross_encoder_reranking.ipynb
├── paper/preliminary-report.md   # the 2-page report for the professor
├── theory/                   # course slides (WIR-01 … WIR-11 PDFs)
├── data/                     # ai.json (+ downloaded snapshots)        [gitignored]
├── index/longeval-sci/       # Terrier index + facet_df.json           [gitignored, ~1.5 GB]
├── ai_index/                 # tiny teaching index from notebook 01    [gitignored]
└── runs/                     # legacy TREC run files (team is not submitting to TIRA)
    ├── snapshot-1/  snapshot-2/  snapshot-3/
    └── ir-metadata.yml
```
Indexes and downloaded data are gitignored. Each notebook starts with a **Configuration**
cell (paths, dataset id, knobs) and auto-detects the repo root, so they run whether the
kernel's cwd is the repo or `notebooks/`.

---

## 6. Approaches & techniques used (index)

Cranfield evaluation paradigm · inverted index / lexicon · TF, TF-IDF, **BM25** weighting
· BM25 `b`/`k1` **grid tuning** · **RM3** pseudo-relevance feedback · informetric
**EF×IDF power-law re-ranking** (author / pub-year) · paired **t-test & Wilcoxon**
significance (Bonferroni-aware) · per-query **error analysis** · **language detection**
audit · neural **MT query translation** (MarianMT) · **accent-folding** for CLIR matching
· weighted **Reciprocal Rank Fusion** · **LambdaMART** learning-to-rank with custom
features · **multilingual dense re-ranking** (e5 bi-encoder) · **multilingual cross-encoder
re-ranking** (mMARCO mMiniLM) — both with lexical+semantic interpolation.

---

## 7. Reproducing

1. `uv sync` and select the `wir-2026` kernel. (On Windows + OneDrive, if uv hits
   file-lock errors use `uv sync --link-mode=copy`.)
2. Run notebooks **01 → 07** in order. NB02 builds/reuses the ~870k-doc index (~10 min on
   first build); NB05 downloads the MarianMT model (~300 MB); NB06 downloads the e5 model
   (~470 MB) and NB07 the cross-encoder (~470 MB); each neural re-rank encodes/scores ~5k
   pairs on CPU (~3 min).

> *The team is **not** submitting to TIRA.* The `runs/` directory holds legacy TREC run
> files from an earlier experiment and is not part of the current deliverable.

---

## 8. Engineering notes & gotchas (decisions documented)

- **Notebooks-only**: the previous `src/*.py` modules and `PROJECT_GUIDE.md` were ported
  into the notebooks and removed; `pyproject.toml` uses `[tool.uv] package = false`.
- **Index reuse**: NB02–07 standardise on `index/longeval-sci/` (stores
  `authors`/`pubyear`). A leftover text-only index directly under `index/` from an early
  run is orphaned and safe to delete (~1.4 GB).
- **Neural re-ranking on CPU**: cache the candidate text once and re-rank only BM25's
  top-50 — keeps each of NB06 (bi-encoder) and NB07 (cross-encoder) to ~3 min on CPU. The
  full text/abstract is fetched from the `ir_datasets` docstore (the index only stores
  title, not abstract).
- **Windows + OneDrive**: avoid `shutil.rmtree` on index dirs (file-lock `WinError 5`) —
  rely on Terrier `overwrite=True`; and run uv with **`--link-mode=copy`** (OneDrive
  blocks uv's hardlinks → `os error 396` / "Access is denied").
- **Accent-folding (CLIR)**: the index tokenizer folds accents (`relación`→`relacion`),
  so Spanish queries are NFKD-folded to ASCII before retrieval; Terrier then applies the
  same English Porter stemmer to both branches.
- **Topic split**: use `df.iloc[...]` (not `np.split`, which returns ndarrays on a
  mixed-dtype DataFrame) for the LTR train/val/test split.
- **Fusion validation**: custom `pt.apply.generic` fusion transformers are run with
  `validate="ignore"` in `pt.Experiment`.

---

## 9. Credit & sources

Per the lab rules, document **who contributed to what** in your report/submission.

- Course slides WIR-02 … WIR-11 (in this repo) — Cranfield, BM25, power laws, informetrics, LTR.
- Course tutorials — <https://github.com/irgroup-classrooms/wir-2026> and <https://github.com/tira-io/teaching-ir-with-shared-tasks>
- LongEval-Sci loader & snapshots — <https://github.com/clef-longeval/ir-datasets-longeval>
- Official PyTerrier baseline + prebuilt HF index — <https://github.com/clef-longeval/longeval-code/tree/main/clef26/scientific-retrieval>
- Task description (CORE data, TIRA) — <https://clef-longeval.github.io/>
- MarianMT model — <https://huggingface.co/Helsinki-NLP/opus-mt-en-es>
- LongEval prior years — CEUR Vol-4038 (2025), Vol-3740 (2024).
