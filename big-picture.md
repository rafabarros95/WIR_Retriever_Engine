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

- sentence-transformers / our "e5" model = an AI model that turns any text — in any language — into a list of numbers (an embedding) that captures its meaning. Two texts about the same topic end up with similar numbers, even if they use different words or languages. "Dense re-ranking" = take BM25's top results and reorder them by meaning-similarity to the query.

### How they connect (the pipeline)

-> Load the papers + answer key → build the index → search with BM25 → grade with nDCG@10 → try to improve the ranking → re-rank the top results with the AI meaning-model → check with statistics whether the improvement is real.

### The actual story:

Build and tune the basic engine. BM25 works; tuning its dials makes it a bit better. Solid baseline.

We tested two "extra idea" boosts (the hypotheses):

- Boost papers whose authors show up a lot → didn't help.
- Boost papers by publication year → hurt (PL) (we expected this; it's a sanity check that our grading actually detects bad ideas).

*The research question: the papers aren't all English (~20% are other languages, Spanish most common). A plain English search might miss relevant Spanish papers. Can we catch them and score higher?*

- Cross-lingual attempt #1 — the obvious way (notebook 06): translate the query to Spanish, search again, and merge the two result lists (RRF). → It failed. Why? The answer key is 96% English papers, so there were almost no Spanish "right answers" to find. Adding a Spanish result list just shoved good English results down. We proved (with a sweep) that the best amount of Spanish to add was zero.

- Cross-lingual attempt #2 — the smart way (notebook 07): use the AI meaning-model that already understands many languages at once, and re-rank BM25's top results by meaning. → It worked — this became our best system (nDCG@10 0.317 vs BM25's 0.292), and statistics confirm the improvement is real. 

### Two honest findings: 

- the AI model on its own is worse than BM25 — you have to blend the two;

- the improvement actually came from understanding meaning/synonyms in English, not really from the cross-language magic (again, because there were so few non-English right answers to recover).

### The final takeaway:

Best system = BM25 + an AI meaning-based re-ranker. And the "research story" is honest and complete: we measured the language problem, showed the naive cross-language fix fails and explained why, then showed the principled fix (meaning-based AI) actually helps. Problem → naive fix fails → smart fix works.
