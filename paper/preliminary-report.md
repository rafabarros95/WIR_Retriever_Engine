# Lexical, Informetric, and Multilingual-Dense Retrieval for LongEval-Sci

**WIR 2026 · TH Köln — Stage 3/4 preliminary report (max. 2 pages)**
*Team: `ENTER_TEAM_NAME` — `ENTER_MEMBER_NAMES`. Contribution statement at the end.*

---

## 1. Goal & research question

We build and evaluate an ad-hoc retrieval system for **LongEval-Sci** (CLEF 2026, Task 1;
CORE scientific papers) and try to beat the TF-IDF / BM25 baselines. Beyond tuning the
lexical baseline, our research idea targets the collection's **multilinguality**: ~20 % of
the corpus is non-English. This motivates our research question:

> **RQ.** Can we exploit the multilinguality of LongEval-Sci to retrieve relevant papers
> that a monolingual English BM25 misses, and thereby improve nDCG@10 — and *how* should
> the cross-lingual signal be incorporated (lexical query translation vs. a shared
> multilingual embedding space)?

We test it through two contrasting approaches and four falsifiable hypotheses.

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
`intfloat/multilingual-e5-small` (dense re-ranker). The project is delivered as **seven
Jupyter notebooks** (the Stage-3 system demo).

**Evaluation.** Official metric **nDCG@10**; we also report MAP and Recall@1000. Systems
are compared in a single `pt.Experiment`; differences vs. BM25 are tested with paired
**Student's t** and **Wilcoxon** tests (p < 0.05).

## 3. Method & steps

1. **Index** the 870k docs with Terrier (Porter stemmer + Terrier stopwords), also storing
   `authors`/`pubyear` and a collection facet document-frequency map.
2. **Baselines & tuning:** TF, TF-IDF, BM25; grid-search BM25 `b`,`k1`; BM25+RM3.
3. **Informetric re-ranker (Stage-4 hypotheses):** an EF×IDF author power-law boost
   (WIR-09 s.40) with publication year as a negative control.
4. **Cross-lingual, lexical (approach A):** translate queries EN→ES with MarianMT,
   accent-fold to match the index, retrieve a Spanish BM25 branch, fuse with weighted RRF,
   and add the ES score as a LambdaMART feature.
5. **Cross-lingual, semantic (approach B):** re-rank BM25's top-50 with a **multilingual
   dense** model (e5) that embeds query and document into a shared space — no translation.

## 4. Results

**Baselines & re-rankers (nDCG@10, 100 topics).**

| System | nDCG@10 | MAP |
|---|---|---|
| **BM25 ≫ dense (interp.)** | **0.3167** | 0.258 |
| BM25-tuned (`b=0.9,k1=1.2`) | 0.3050 | 0.264 |
| BM25 | 0.2922 | 0.257 |
| TF-IDF | 0.2921 | 0.259 |
| BM25+PL[authors] | 0.2885 | 0.253 |
| BM25+RM3 | 0.2781 | 0.240 |
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
- **H4 — multilingual dense re-ranking improves nDCG@10:** **supported.**
  `BM25 ≫ dense (interp.)` = **0.3167**, **+0.0245** over BM25, **significant** (t-test
  *p* = 0.039, Wilcoxon *p* = 0.038; 38 wins / 25 losses). It is our best system.

**Error & subset analysis.** *Blend, don't replace:* pure dense re-ranking **hurts**
(0.2602) — the small e5 model is weaker than BM25 lexically — but `0.5·BM25 + 0.5·dense`
wins, showing lexical and semantic evidence are complementary. Splitting by topic language,
the dense gain is **larger and significant on English-only topics** (Δ = +0.0267,
*p* = 0.040) and **negligible** on the 10 cross-lingual topics (Δ = +0.0049, *p* = 0.82).
**Interpretation:** the win comes from general **semantic matching** (synonymy/paraphrase),
*not* specifically from cross-lingual recovery — the cross-lingual subset is too small and
already English-covered on these qrels to demonstrate the effect.

## 5. Conclusion, limitations & next steps

Our strongest, **statistically significant** system is **BM25 re-ranked by a multilingual
dense model** (nDCG@10 0.317 vs. BM25 0.292). The project's narrative is honest end-to-end:
a *language audit* sized the cross-lingual opportunity, a *naive lexical* cross-lingual
approach failed for well-understood reasons, and a *principled semantic* approach delivered
a real gain — though, on the current judgments, that gain is semantic rather than provably
cross-lingual. We submit the dense re-ranker (fallback: BM25-tuned) to TIRA per snapshot.

**Limitations / future work.** (1) `e5-small` is the smallest model; `e5-base`/`-large` or
a multilingual cross-encoder should score higher. (2) The index uses one English analyzer
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
