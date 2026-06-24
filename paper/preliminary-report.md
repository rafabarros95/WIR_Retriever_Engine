# Lexical, Informetric, and Multilingual-Dense Retrieval for LongEval-Sci

**WIR 2026 · TH Köln — Stage 3/4 preliminary report (max. 2 pages)**


## 1. Goal & research question

We build and evaluate an ad-hoc retrieval system for **LongEval-Sci** (CLEF 2026, Task 1;
CORE scientific papers) and try to beat the TF-IDF / BM25 baselines. Beyond tuning the
lexical baseline, our research idea targets the collection's **multilinguality**: ~20 % of
the corpus is non-English. This motivates our research question:

> Can we exploit the multilinguality of LongEval-Sci to retrieve relevant papers
> that a monolingual English BM25 misses, and thereby improve nDCG@10 — and how should
> the cross-lingual signal be incorporated (lexical query translation vs. a shared
> multilingual embedding space)?

We test it through contrasting lexical and neural approaches and five falsifiable hypotheses.

## 2. Data, tools & evaluation

**Data.** `longeval-sci-2026/snapshot-1/train/dctr`: **869,902** documents
(title + abstract), **100** judged queries, **8,772** qrels (click-derived graded
relevance). Documents expose `authors` and dates but **no subject/keyword field**.

**Language audit (our motivation, measured with `langdetect`).** Corpus ≈ **80 % English**
(es 6 %, pt 3 %, de 3 %, fr 2 %); but the **judged-relevant** documents are ≈ **96 %
English / 3 % Spanish**, and only **10 / 100 topics** have any non-English relevant paper.
This mismatch is central to our findings.

**Tools.** PyTerrier 1.0.1 (Terrier 5.11, JDK 17); `ir-datasets-longeval`; `scipy` for
significance tests; `langdetect`; **MarianMT** (`Helsinki-NLP/opus-mt-en-es`) via
`transformers`; `xgboost`/`lightgbm` (LambdaMART); **`sentence-transformers`** with
`intfloat/multilingual-e5-small` (dense bi-encoder) and
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual cross-encoder). The project is
delivered as **seven Jupyter notebooks**.

**Evaluation.** Official metric **nDCG@10**; we also report MAP and Recall@1000. Systems
are compared in a single `pt.Experiment`; differences vs. BM25 are tested with paired
**Student's t** and **Wilcoxon** tests (p < 0.05).

## 3. Method & steps

1. **Index** the 870k docs with Terrier (Porter stemmer + Terrier stopwords), also storing
   `authors`/`pubyear` and a collection facet document-frequency map.
2. **Baselines & tuning:** TF, TF-IDF, BM25; grid-search BM25 `b`,`k1`; BM25+RM3.
3. **Informetric re-ranker (Stage-4 hypotheses):** an EF×IDF author power-law boost with publication year as a negative control.
4. **Cross-lingual, lexical (approach A):** translate queries EN→ES with MarianMT,
   accent-fold to match the index, retrieve a Spanish BM25 branch, fuse with weighted RRF,
   and add the ES score as a LambdaMART feature.
5. **Cross-lingual, semantic — bi-encoder (approach B):** re-rank BM25's top-50 with a
   **multilingual dense** model (e5) that embeds query and document into a shared space —
   no translation.
6. **Cross-lingual, semantic — cross-encoder (approach B+):** re-rank the same top-50 with a
   **multilingual cross-encoder** (mMARCO mMiniLM) that scores each query–document pair
   *jointly*; interpolate `0.5·BM25 + 0.5·CE`.

## 4. Results

**Baselines & re-rankers (nDCG@10, 100 topics).**

| System | nDCG@10 | MAP |
|---|---|---|
| **BM25 ≫ CE (interp.)** | **0.3242** | 0.261 |
| BM25 ≫ dense (interp.) | 0.3167 | 0.258 |
| BM25-tuned (`b=0.9,k1=1.2`) | 0.3050 | 0.264 |
| BM25 | 0.2922 | 0.257 |
| TF-IDF | 0.2921 | 0.259 |
| BM25+PL[authors] | 0.2885 | 0.253 |
| BM25+RM3 | 0.2781 | 0.240 |
| BM25 ≫ CE (pure) | 0.2654 | 0.219 |
| BM25 ≫ dense (pure) | 0.2602 | 0.212 |
| BM25+PL[pubyear]* (control) | 0.2015 | 0.173 |
| TF | 0.0470 | 0.038 |

**Hypotheses.**

- **H1 — author power-law boost improves nDCG@10:** **rejected.** Δ = +0.0008, t-test
  *p* ≈ 0.88 (n.s.); no λ helps. Mechanism: 83 % of authors occur in a single document, so
  the element-frequency signal is too sparse.
- **H2 — pub-year boost (negative control) does not help:** **supported.** Δ = −0.091,
  *p* < 0.0001 — a large significant drop, validating the setup.
- **H3 — lexical cross-lingual fusion improves nDCG@10:** **rejected.** ES-only BM25 scores
  0.024; equal-weight RRF halves nDCG@10; the optimal Spanish weight is **0**. With ~96 %
  English relevant docs there is almost no headroom, and fusing a weak branch only
  displaces relevant English docs.
- **H4 — multilingual neural re-ranking improves nDCG@10:** **supported.** Both blended
  neural re-rankers beat BM25. The **cross-encoder** `BM25 ≫ CE (interp.)` = **0.3242**,
  **+0.0320** over BM25, **significant** (t-test *p* = 0.032, Wilcoxon *p* = 0.014;
  41 wins / 27 losses) — our **best system**; the dense bi-encoder `BM25 ≫ dense (interp.)`
  = 0.3167 (+0.0245, *p* = 0.039) is a close second.
- **H5 — the (heavier) cross-encoder significantly beats the dense bi-encoder:**
  **rejected.** Δ = +0.0074, *p* = 0.41 (n.s.; 36 wins / 32 losses). The cross-encoder edges
  ahead on nDCG@10 but is statistically indistinguishable from the tiny `e5-small` — a cheap
  bi-encoder already captures most of the gain.

**Error & subset analysis.** *Blend, don't replace:* pure neural re-ranking **hurts** for
both models (CE 0.2654, dense 0.2602) — alone they are weaker than BM25's lexical signal —
but `0.5·BM25 + 0.5·neural` wins, showing lexical and semantic evidence are complementary.
Splitting by topic language, the gain is **larger and significant on English-only topics**
(cross-encoder Δ = +0.0350, *p* = 0.033; dense Δ = +0.0267, *p* = 0.040) and **negligible**
on the 10 cross-lingual topics (cross-encoder Δ = +0.0050, *p* = 0.84; dense Δ = +0.0049,
*p* = 0.82). **Interpretation:** the win comes from general **semantic matching**
(synonymy/paraphrase), *not* from cross-lingual recovery — and **scaling the model up
(bi-encoder → cross-encoder) does not move the cross-lingual subset**, because it is too
small and already English-covered on these qrels.

## 5. Conclusion, limitations & next steps

Our strongest, **statistically significant** system is **BM25 re-ranked by a multilingual
cross-encoder** (nDCG@10 0.324 vs. BM25 0.292); the dense bi-encoder (0.317) is a close,
much cheaper second. The project's narrative is honest end-to-end: a *language audit* sized
the cross-lingual opportunity, a *naive lexical* cross-lingual approach failed for
well-understood reasons, and *principled semantic* re-rankers delivered a real gain — though,
on the current judgments, that gain is semantic rather than provably cross-lingual, and
scaling from a bi-encoder to a cross-encoder raises nDCG@10 without recovering the
cross-lingual subset.

**Limitations / future work.** (1) We tested the smallest e5 bi-encoder and an mMiniLM-L12
cross-encoder; larger backbones (`e5-large`, a monoT5 / larger multilingual cross-encoder)
could raise nDCG@10 further. (2) The index uses one English analyzer
for all languages; a Spanish-analyzed (or per-language) index is a cleaner CLIR setup.
(3) Re-ranking only top-50 caps recall — deepen the pool for the submission. (4) Crucially,
we will **re-run the cross-lingual subset analysis on our group's own topics/qrels**, which
may contain more non-English relevant papers and is where H4's cross-lingual effect could
materialise. (5) Evaluate across snapshots for the temporal robustness that defines LongEval.

## 6. Reproducibility & contribution statement

All results are reproducible from the seven notebooks (`notebooks/01…07`) on the `wir-2026`
kernel; see the repository README for per-step details and exact commands.

*Contribution statement (fill in):* `ENTER_WHO_DID_WHAT` — e.g., indexing & baselines (X),
informetric boost & hypotheses (Y), cross-lingual + dense re-ranking (Z), report (all).

**References.** WIR-02/03/09/10/11 course slides; PyTerrier (pyterrier.readthedocs.io);
LongEval-Sci (clef-longeval.github.io); e5 (Wang et al., *Multilingual E5*); BM25/RM3
(Robertson & Zaragoza, 2009); LambdaMART (Burges, 2010).
