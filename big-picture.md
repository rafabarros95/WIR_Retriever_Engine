### The big picture
- We're building a search engine for scientific papers and proving how good it is. Like building Google, but for a fixed pile of ~870,000 research papers, and then grading ourselves against an answer key.

- Everything else is either a piece of that machine or a way to measure it.

### Grouping Tools and methods in Storytelling modus

- The library and the answer key (the data):

   - LongEval-Sci = the pile of 870,000 papers we search through. (Built from "CORE", a real database of open-access papers.)

   - Topics / queries = what a user types into the search box (e.g. "social media in academic environments").

   - qrels / "relevance judgments" = the answer key. Humans looked at papers and marked which ones are actually relevant to each query. Without this, we couldn't grade anything. That's what we did in Assessment 1 & 2.
   
   - nDCG@10 = the standard evaluation/grade. It asks: "Of the 10 results you showed first, how many were the right ones (per the answer key), and were the best ones near the top?" Higher = better.

### The search engine itself:

- Index = like the index at the back of a textbook, but for 870k papers: a giant lookup table of "which word appears in which papers." Building it is called indexing. we do it once; then search is fast. It was notebook 2.

- BM25 = the classic, still-excellent formula for scoring how well a paper matches a query. It's smart about two things: rare words count more (matching "informetrics" matters more than matching "the"), and long papers don't get an unfair advantage. TF-IDF and TF are older, simpler versions of the same idea. BM25 is the baseline — the thing we "try" to beat.

- Tuning BM25 (b, k1) = BM25 has two adjustment dials; so far i tried few settings and kept the best. ("BM25-tuned".)

- scipy / "significance tests" (t-test, Wilcoxon) = statistics that answer "is this improvement real, or just random luck?" If an improvement passes the test (p < 0.05), we trust it.

### The research-idea tools:

- langdetect = guesses what language a piece of text is in. We used it to audit the papers' languages.

- MarianMT (run by the libraries transformers + torch) = an offline translation model — like Google Translate on your laptop. We used it to translate English queries into Spanish.

- RRF (Reciprocal Rank Fusion) = a simple recipe for merging two ranked lists into one combined list.

- LambdaMART (run by xgboost / lightgbm) = "learning to rank." Instead of one fixed formula, this is a machine-learning model that learns from the answer key how to best combine several signals into a ranking.

- sentence-transformers / our "e5" model = an AI model that turns any text — in any language — into a list of numbers (an embedding) that captures its meaning. Two texts about the same topic end up with similar numbers, even if they use different words or languages. "Dense re-ranking" = take BM25's top results and reorder them by meaning-similarity to the query. We also tried a stronger cousin, a *cross-encoder* (the `mmarco` model) that reads the query and a paper *together* and scores how well they match — slower, but the most powerful re-ranker.

### How they connect (the pipeline)

-> Load the papers + answer key → build the index → search with BM25 → grade with nDCG@10 → try to improve the ranking → re-rank the top results with the AI meaning-model → check with statistics whether the improvement is real.

### The actual story:

Build and tune the basic engine. BM25 works; tuning its dials makes it a bit better. Solid baseline.

We tested two "extra idea" boosts (the hypotheses):

- Boost papers whose authors show up a lot → didn't help.
- Boost papers by publication year → hurt (PL) (we expected this; it's a sanity check that our grading actually detects bad ideas).

*The research question: the papers aren't all English (~20% are other languages, Spanish most common). A plain English search might miss relevant Spanish papers. Can we catch them and score higher?*

- Cross-lingual attempt #1 — the obvious way (notebook 05): translate the query to Spanish, search again, and merge the two result lists (RRF). → It failed. Why? The answer key is 96% English papers, so there were almost no Spanish "right answers" to find. Adding a Spanish result list just shoved good English results down. We proved (with a sweep) that the best amount of Spanish to add was zero.

- Cross-lingual attempt #2 — the smart way (notebook 06): use the AI meaning-model (a *bi-encoder*) that already understands many languages at once, and re-rank BM25's top results by meaning. → It worked — a real, statistically-confirmed improvement (nDCG@10 0.317 vs BM25's 0.292).

- Cross-lingual attempt #3 — the strongest model (notebook 07): swap the bi-encoder for a multilingual *cross-encoder* that reads each query+paper pair together. → It became our **best system** (nDCG@10 **0.324**), and the win over BM25 is even more robust — but it's only a *tiny, non-significant* step above the much cheaper bi-encoder.

### Two honest findings: 

- the AI model on its own is worse than BM25 — you have to blend the two;

- the improvement actually came from understanding meaning/synonyms in English, not really from the cross-language magic (again, because there were so few non-English right answers to recover);

- bringing in the heavier, fancier model (the cross-encoder) barely helped over the small one — most of the gain was already captured by the cheap bi-encoder.

### The final takeaway:

Best system = BM25 + a multilingual cross-encoder re-ranker (nDCG@10 0.324). And the "research story" is honest and complete: we measured the language problem, showed the naive cross-language fix fails and explained why, then showed the principled fix (meaning-based AI) actually helps. Problem → naive fix fails → smart fix works.

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

### The 7 notebooks in other words (the story):

Each one `hands something to the next`: notebook 2 hands the index to everyone; notebook 3 hands the tuned BM25 settings and the BM25 candidate list to the rest; notebooks 5–7 all re-use that same candidate list so the comparison stays fair.

- Notebook 1 — Warm-up on a tiny pile of papers. Before touching 870k papers, we practice on a small file (`ai.json`, a few thousand AI papers). We build a mini-index, look inside it, and even recompute the matching score *by hand* to prove there's no magic — just counting rare words. **Nothing here is graded; it's training wheels.** → teaches us the machinery used everywhere else.

- Notebook 2 — Build the real library index. We load the real 870k-paper collection and build the big index (the word→papers lookup table). We also store each paper's `authors` and `year` on the side, because notebook 3/4 will need them. This is "build the engine" part 1. -> hands the index to notebooks 3–7.

- Notebook 3 — Build the search systems and grade them (Assignment III). Here we actually search. We line up TF, TF-IDF, BM25, then improve BM25 two standard ways (tune its dials; expand the query with "RM3"), plus our own author-boost idea — and grade them all with nDCG@10 in one comparison. This is the system demo deliverable. → hands the best BM25 settings + the candidate list forward, and the author-boost idea to notebook 4.

- Notebook 4 — Test a hypothesis properly (Assignment IV). Notebook 3 showed averages; an average can lie. Here we ask "is the author-boost improvement real or luck?" using statistics (t-test, Wilcoxon), query by query. We also test a deliberately-bad idea (boost by year) as a sanity check. Result: author-boost = no real effect; year-boost = clearly hurts (as expected). -> confirms our grading actually detects good vs. bad ideas, which we lean on for notebooks 5–7.

- Notebook 5 — Cross-language attempt 1: translation (the obvious fix). The research question starts here: the papers aren't all English. So we translate each query into Spanish, run a second Spanish search, and merge the two lists. We also let a machine-learning ranker (LambdaMART) try to use the Spanish score. `It fails` — the answer key is 96% English, so Spanish results just push good English ones down. -> an honest negative result that motivates the smarter fix in notebooks 6–7.

- Notebook 6 — Cross-language attempt 2: meaning, not words (bi-encoder). Instead of translating, we use an AI model that turns any language into "meaning numbers," and re-order BM25's top results by meaning. It works — the first real, statistically-confirmed win (0.317 vs BM25's 0.292). -> carries the same re-ranking trick to notebook 7, now with a stronger model.

- Notebook 7 — Cross-language attempt 3: the strongest model (cross-encoder). Same idea as notebook 6, but with a heavier AI model that reads the query and each paper together. It becomes our best system (0.324) — but only a tiny step above the cheap model from notebook 6. -> closes the story: the gain is real, but it comes from understanding meaning in general, not specifically from crossing languages.

The thread tying them together: 1 teaches -> 2 builds the index -> 3 builds & grades the systems -> 4 proves which improvements are real -> 5 tries the naive cross-language fix and fails -> 6 tries the smart fix and wins -> 7 pushes the smart fix to its strongest form. Every notebook after 3 re-ranks the same BM25 candidates, so all the numbers are directly comparable.
