# Lexical, Informetric, and Multilingual-Neural Retrieval for LongEval-Sci

WIR 2026 · TH Köln — Stage 3/4 preliminary report

---

### 1. Goal

**We build and evaluate an ad-hoc retrieval system for LongEval-Sci (CLEF 2026, Task 1;
CORE scientific papers) and try to beat the TF-IDF / BM25 baselines. Beyond tuning the
lexical baseline, our research idea targets a structural property of the collection that
a standard English search engine ignores: LongEval-Sci is multilingual.**

### Why multilinguality is the interesting problem here

LongEval-Sci is built from CORE, a global aggregation of open-access repositories, so its
documents are written in many languages. Our language audit (§2) measures this directly:
~20 % of the corpus is non-English — Spanish ( roughly 6 %), Portuguese, German and French
being the largest minorities. A monolingual English engine treats those papers as
near-noise: BM25 can only match an English query against a Spanish abstract by accidental
lexical overlap (shared proper nouns, formulae, accent-folded cognates). If a relevant
paper is written in Spanish, an English BM25 will usually rank it far down. This is the
gap our research targets.

--- 
This motivates our `research question`:

**Can we exploit the multilinguality of LongEval-Sci to retrieve relevant papers that a
monolingual English BM25 misses, and thereby improve nDCG@10 — and how should the
cross-lingual signal be incorporated: by lexical query translation, or by a shared
multilingual embedding space (bi-encoder vs. cross-encoder)?**

---

The question has two halves: 
- The first is whether there is cross-lingual headroom on
these judgments at all. 
- The second — the more useful engineering question — is which
mechanism captures it: a cheap lexical route (translate the query, retrieve again, fuse)
or a neural route that places queries and documents of any language into one semantic
space. 
- We answer both with five falsifiable hypotheses (§4), each tested with a paired
significance test against BM25, and each split by topic language so we can see where an
effect comes from rather than trusting a single aggregate mean.

Alongside the multilingual question we also test a second, independent idea drawn from the
informetrics part of your course: re-ranking by an author power law. This
is the Stage-4 hypothesis pair (H1/H2) and serves as a controlled, in-domain experiment
with a built-in negative control.

---

## 2. Data, tools & evaluation

Data: `longeval-sci-2026/snapshot-1/train/dctr`: about 869,902 documents
(title + abstract), 100 judged queries, 8,772 qrels (click-derived graded
relevance). Documents expose `authors` and dates but no subject/keyword field.

Language audit: (the motivation, measured with `langdetect`). We detect the language
of the whole corpus (sampled) and, crucially, of every judged-relevant document:

| Where | English | Largest non-English |
|---|---|---|
| Whole corpus (sample) | ~80 % | es ~6 %, pt ~3 %, de ~3 %, fr ~2 % |
| Judged-relevant docs | ~96 % | es ~3 %, rest <1 % |
| Topics with ≥1 non-English relevant doc | 10 / 100 | — |

This single table is the backbone of the whole paper: the corpus is ~20 % non-English,
but the relevant set is ~96 % English, and only 10 of 100 topics have any non-English
relevant paper. So the aggregate headroom for a cross-lingual method is small by
construction — the interesting effect, if any, must be a targeted one on those 10
topics, which is exactly the subset analysis we run for every cross-lingual system. 

**Open question to you Prof. Dr. Schaer: would it make sense intead of using the sample of the whole collection, or to use the whole collection?**

Tools — what, where, and why: The project is delivered as seven Jupyter notebooks
that mirror the four course tutorials (`pyterrier-intro / -irdatasets-indexing /
-retrieval / -ltr`) and extend them. Each tool below was chosen for one specific job:

| Tool                                                                   | What it does | Where (NB) | Why we used it there                                                                                                                                                                                    |
|------------------------------------------------------------------------|---|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PyTerrier 1.0.1 / Terrier 5.11 (JDK 17)                                | indexing, retrieval, `pt.Experiment`, pipelines | all | The course's reference engine; gives us BM25/TF-IDF, a one-call evaluation harness, and a `>>` pipeline operator so every re-ranker plugs onto the same BM25 first stage. Terrier runs on the JVM, hence the JDK. |
| ir-datasets-longeval                                                   | loads LongEval-Sci docs / queries / qrels | 02–07 | The official loader for the collection; guarantees we use the same documents, topics and judged qrels as the shared task (no hand-parsing of TREC files).                                               |
| pandas / numpy                                                         | dataframes + numeric ops | all | PyTerrier speaks dataframes (topics, qrels, runs are all `DataFrame`s); numpy backs the score normalisation and the by-hand TF-IDF.                                                                     |
| nltk** (PorterStemmer)                                                 | word stemming | 01 | To recompute IDF/TF-IDF by hand we must stem query terms the same way the index does (`learning`->`learn`); the warm-up demystifies what BM25 does internally. Just for learning purposes               |
| matplotlib                                                             | plots | 03–07 | Per-system bar charts, the λ-sensitivity curve, the ES-weight sweep, and the per-query win/loss charts — visual error analysis for the Stage-4 story.                                                   |
| scipy** (`ttest_rel`, `wilcoxon`)                                      | paired significance tests | 04–07 | A single mean is not evidence. Paired Student's t (parametric) + Wilcoxon (non-parametric) tell us whether a re-ranker's per-query gain over BM25 is real (p<0.05) or luck.                             |
| langdetect                                                             | detects the language of a text | 05–07 | The motivation of the whole research question: it audits the language of the corpus and of every judged-relevant doc, and flags the 10 cross-lingual topics used in the subset analysis.                |
| transformers + torch running MarianMT (`Helsinki-NLP/opus-mt-en-es`)   | offline EN→ES neural translation | 05 | Approach A needs the queries in Spanish to run a Spanish BM25 branch. MarianMT is a compact, high-quality MT model; only ~100 short queries are translated, so it is cheap on CPU.                      |
| xgboost** (`XGBRanker`, LambdaMART, `rank:ndcg`)                       | learning-to-rank | 05 | The `pyterrier-ltr` technique: instead of a fixed formula, learn from qrels how to combine TF-IDF + PL2 + the cross-lingual score. We use xgboost's LambdaMART because it optimises nDCG directly.      |
| sentence-transformers -> `intfloat/multilingual-e5-small` (bi-encoder) | multilingual text embeddings | 06–07 | Approach B: embed query and document independently into one shared multilingual space so semantic / cross-lingual matches work without translation. "small" keeps CPU encoding to a few minutes.        |
| sentence-transformers -> `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (cross-encoder) | joint (query, doc) scoring | 07 | Approach B+: the strongest re-ranking family — reads query and document together (cross-attention). Trained on mMARCO (15 languages), so it scores English queries against non-English text in one pass. |

Key PyTerrier components used inside these notebooks: `pt.GridSearch` (tune BM25 `b`,`k1` — NB03), `pt.rewrite.RM3` (pseudo-relevance feedback — NB03), `pt.terrier.FeaturesRetriever` + `pt.ltr.apply_learned_model` (the LTR pipeline — NB05), and `pt.apply.by_query` / `pt.apply.generic` (our custom power-law, fusion and neural re-rankers — NB03–07).

Evaluation: Official metric nDCG@10; we also report MAP and Recall@1000. Every
system is compared in a single `pt.Experiment` on the same 100 topics, so all numbers are
directly comparable; differences vs. BM25 are tested per query with paired
Student's t and Wilcoxon tests (p < 0.05), and split into the 10 cross-lingual vs.
90 English-only topics.

---

## 3. Method & pipeline — how we attack the multilingual problem

The system is a two-stage design throughout: a strong lexical first stage (BM25)
produces a candidate pool, and every "idea" is a re-ranking or fusion on top of
it. This keeps the experiments comparable (same candidates) and cheap (we only score the
top-k). The pipeline, end to end:

```
        index 870k docs (Terrier, Porter + stopwords, store authors/pubyear)   [NB02]
                                  │
              ┌───────────────────┴───────────────────┐
        BM25 / TF-IDF / TF        BM25-tuned (grid b,k1)   BM25+RM3            [NB03]
              │                         │
              │                         └── informetric re-rank: EF×IDF author
              │                             power-law boost  +  pubyear control [NB03/04]
              │
        BM25 candidate pool ──► RESEARCH QUESTION: the multilingual branch
              │
   ┌──────────┼───────────────────────────┬──────────────────────────────┐
   A. lexical                       B. semantic (bi-encoder)      B+. semantic (cross-encoder)
   translate EN->ES (MarianMT),      embed query+doc with e5       score (query,doc) jointly with
   accent-fold, BM25[ES] branch,    in a shared space, re-rank    mMARCO mMiniLM, re-rank top-50  [NB07]
   weighted RRF fusion, +LTR        top-50 by cosine        [NB06]
   feature                  [NB05]
              │                            │                              │
              └──────────── evaluate nDCG@10 + paired significance + language-subset split
```

Stage 0 — Index (NB02): Build the Terrier index over the 870k papers (Porter stemmer
+ Terrier stopwords). We additionally *store* the `authors`/`pubyear` metadata and a
collection facet document-frequency map — the only ingredient the informetric hypothesis
needs. Spanish accents are folded to ASCII by the analyzer (`educación`→`educacion`); this
matters for the cross-lingual branch.

Stage 1 — Lexical baselines & tuning (NB03): TF -> TF-IDF -> BM25 (the ordering itself
is a result: length normalisation + tf-saturation make BM25 the baseline to beat), then
two standard improvements — BM25 grid-tuning of `b`,`k1` and BM25+RM3
pseudo-relevance feedback.

Stage 2 — Informetric re-rank & Stage-4 hypotheses (NB03/04): An EF × IDF author
power-law re-ranker: a document is boosted when its authors are both frequent inside
the query's result set (element frequency) and rare across the collection (IDF from the
facet map). We pair it with the same boost on publication year as a negative
control. These are H1/H2.

Stage 3 — The multilingual research question (NB05–07): Three escalating approaches on
the same BM25 candidate pool:

- Approach A — lexical cross-lingual fusion (NB05): Translate each query EN to ES with
  MarianMT, accent-fold the translation so its terms match the index, retrieve a
  Spanish BM25 branch on the same index, and combine EN+ES with weighted Reciprocal
  Rank Fusion(RRF). We sweep the Spanish-branch weight to find the best fusion, and also feed
  the ES-BM25 score as a custom feature into a LambdaMART ranker (the `pyterrier-ltr`
  technique). This is the obvious, cheap way — and the way we expect to fail, given the
  ~96 %-English relevant set.

- Approach B — multilingual dense bi-encoder (NB06): Skip translation entirely. Embed
  the query and each candidate document with `multilingual-e5-small` into one shared
  semantic space and re-rank BM25's top-50 by cosine similarity. An English query and a
  Spanish paper about the same topic land close together without any translation step.

- Approach B+ — multilingual cross-encoder (NB07): The strongest re-ranking family:
  instead of embedding query and document independently, feed the (query, document) pair
  jointly through `mmarco-mMiniLMv2` (trained on machine-translated MS MARCO, 15
  languages) so the model sees full cross-attention between query and document terms.

For every cross-lingual system we report the aggregate nDCG@10, the paired significance
test vs. BM25, and the split into the 10 cross-lingual vs. 90 English-only topics —
the analysis that tells us whether any gain is genuinely cross-lingual or just better
English semantics.

---

### Notebooks Overview:

| NB | Focus | Pipeline                                                                                   |
|----|---|--------------------------------------------------------------------------------------------|
| 01 | Warm-up | TF -> TF-IDF (built by hand to learn IDF mechanics)                                        |
| 02 | Indexing | Build Lucene index with author/year metadata (foundation for later boosting)               |
| 03 | Sparse baselines | TF < TF-IDF < BM25 < BM25-tuned -> RM3 query expansion -> power-law author/year boost      |
| 04 | Error analysis | BM25 -> paired significance tests (author boost vs. year control); per-query λ sensitivity |
| 05 | Cross-lingual lexical | BM25(EN) + BM25(ES translated) -> RRF fusion -> optional LambdaMART LTR                    |
| 06 | Dense re-ranking | BM25 top-50 -> e5-small bi-encoder embeddings -> cosine re-rank -> 0.5·BM25 + 0.5·dense    |
| 07 | Cross-encoder re-ranking | BM25 top-50 -> mMarco cross-encoder joint scoring -> 0.5·BM25 + 0.5·CE -> nDCG@10 = 0.324  |

Overall flow: sparse lexical baselines (01–03) -> hypothesis testing (04) -> cross-lingual fusion (05) -> semantic bi-encoder (06) -> semantic cross-encoder (07, best result).

---

## 4. Results

Baselines & re-rankers (nDCG@10, 100 topics, single `pt.Experiment`):

| System | nDCG@10 | MAP |
|---|---|---|
| BM25 ≫ CE (interp.) — multilingual cross-encoder | 0.3242 | 0.261 |
| BM25 ≫ dense (interp.) — multilingual bi-encoder | 0.3167 | 0.258 |
| BM25-tuned (`b=0.9,k1=1.2`) | 0.3050 | 0.264 |
| BM25 | 0.2922 | 0.257 |
| TF-IDF | 0.2921 | 0.259 |
| BM25+PL[authors] | 0.2885 | 0.253 |
| BM25+RM3 | 0.2781 | 0.240 |
| BM25 ≫ CE (pure) | 0.2654 | 0.219 |
| BM25 ≫ dense (pure) | 0.2602 | 0.212 |
| BM25+PL[pubyear] (control) | 0.2015 | 0.173 |
| TF | 0.0470 | 0.038 |

Hypotheses:

- H1 — author power-law boost improves nDCG@10: rejected. Δ = +0.0008, t-test
  p ≈ 0.88 (n.s.); no λ in a sweep helps. Mechanism: 83 % of authors occur in a single
  document, so the element-frequency signal inside a result set is too sparse.
- H2 — pub-year boost (negative control) does not help: supported. Δ = −0.091,
  p < 0.0001 — a large significant drop, which validates the experimental setup (our
  evaluation correctly detects a bad idea).
- H3 — lexical cross-lingual fusion improves nDCG@10: rejected. ES-only BM25 scores
  0.024; equal-weight RRF roughly halves nDCG@10 (0.158); the optimal Spanish weight is
  0 (the sweep is monotonically decreasing). With ~96 % English relevant docs there is
  almost no headroom, and fusing a weak branch only displaces relevant English docs.
  LambdaMART with the ES feature also drops (0.277 vs 0.309 without it).
- H4 — multilingual neural re-ranking improves nDCG@10: supported. Both blended
  neural re-rankers beat BM25. The cross-encoder `BM25 ≫ CE (interp.)` = 0.3242,
  +0.0320 over BM25, significant (t-test p = 0.032, Wilcoxon p = 0.014;
  41 wins / 27 losses) — our best system; the dense bi-encoder `BM25 ≫ dense (interp.)`
  = 0.3167 (+0.0245, p = 0.039) is a close, much cheaper second.
- H5 — the (heavier) cross-encoder significantly beats the dense bi-encoder:
  rejected. Δ = +0.0074, p = 0.41 (n.s.; 36 wins / 32 losses). The cross-encoder edges
  ahead on nDCG@10 but is statistically indistinguishable from the tiny `e5-small`.

Error & language-subset analysis (the heart of the research question). Two findings
recur across both neural approaches:

1. Blend, don't replace: Pure neural re-ranking hurts for both models
   (CE 0.2654, dense 0.2602) — alone they are weaker than BM25's lexical signal on short
   scientific records — but `0.5·BM25 + 0.5·neural` wins. Lexical and semantic evidence are
   complementary, not substitutes.
2. The gain is semantic, not (provably) cross-lingual: Splitting by topic language, the
   improvement is larger and significant on the English-only topics (cross-encoder
   Δ = +0.0350, p = 0.033; dense Δ = +0.0267, p = 0.040) and negligible on the 10
   cross-lingual topics (cross-encoder Δ = +0.0050, p = 0.84; dense Δ = +0.0049,
   p = 0.82). So the win comes from general semantic matching (synonymy / paraphrase),
   and scaling the model up (bi-encoder → cross-encoder) does not move the cross-lingual
   subset — because that subset is small (n = 10) and already English-covered on these
   qrels.

---

## 5. Conclusion, limitations & next steps

Our strongest, statistically significant system is **BM25 re-ranked by a multilingual
cross-encoder** (nDCG@10 0.324 vs. BM25 0.292); the dense bi-encoder (0.317) is a close,
much cheaper second. The answer to the research question is nuanced and honest:

- Yes, neural re-ranking helps — both multilingual models beat BM25 significantly when
  blended with it.
- But the cross-lingual mechanism is the wrong story on these judgments. The lexical
  translate-and-fuse route fails outright, and even the strongest multilingual model gains
  nothing measurable on the non-English-relevant topics. The improvement is general
  semantic matching, which a multilingual model happens to provide.
- The decisive cause is the data, and we measured it: the relevant set is ~96 %
  English, so there is almost no cross-lingual relevance to recover — the language audit
  predicted the result before we ran a single re-ranker.

This is the contribution we would lead the paper with: a diagnosis that sizes the
gap between corpus language (~20 % non-English) and relevant-set language (~4 %
non-English), shows why an intuitively appealing cross-lingual idea fails here, and
states the conditions under which it would pay off.

Limitations / future work: 
1) We tested the smallest e5 bi-encoder and an mMiniLM-L12 cross-encoder; larger multilingual backbones could raise nDCG@10 further. 
2) The index uses one English analyzer for all languages; a Spanish-analyzed (or per-language) index is a cleaner CLIR setup and the most promising lever for the lexical branch. 
3) Re-ranking only the top-50 caps recall — deepen the pool for the submission and keep the BM25 tail.
4) Crucially, we will re-run the cross-lingual subset analysis on our group's own topics/qrels, which may contain more non-English relevant papers and is where the cross-lingual effect (H4's targeted version, H3) could finally materialise. 
5) Evaluate across snapshots for the temporal robustness that defines LongEval.

---

## 6. Reproducibility & contribution statement

All results are reproducible from the seven notebooks (`notebooks/01…07`) on the `wir-2026`
kernel; each notebook runs top-to-bottom and states its goal + the tutorial it mirrors at
the top. See the repository README for per-step details and exact commands.


References: WIR-02/03/09/10/11 course slides; PyTerrier (pyterrier.readthedocs.io);
LongEval-Sci (clef-longeval.github.io); e5 (Wang et al., Multilingual E5); mMARCO
(Bonifacio et al., 2021); BM25/RM3 (Robertson & Zaragoza, 2009); LambdaMART (Burges, 2010). Also HuggingFace's transformers library, MarianMT, and sentence-transformers.
