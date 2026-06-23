# Design notes

Why `lr_tool` is built the way it is. This complements the README (which says *what*
the tool does) by recording the *why* behind the decisions, the alternatives that were
weighed, and where we'd go next. It's meant for contributors deciding whether to change
something — read the relevant entry before you do, because most of these choices have a
non-obvious reason.

Each entry is roughly: **Decision → Why → Alternatives considered → Status / caveats.**

---

## 1. Retrieval spine: Crossref, not a publisher scraper

**Decision.** The default and only first-class retrieval source is the **Crossref REST
API**. The ACM Digital Library scraper is kept only as an opt-in `--source acm` fallback.

**Why.** ACM's `dl.acm.org` sits behind a Cloudflare managed challenge; a headless,
stealthed browser does not reliably clear it, so the scraper returns **0 records** in
practice. Crossref is the canonical DOI metadata registry that publishers (ACM included)
deposit to at publication time — it's fresh, complete for the core bibliographic fields,
legal to query in bulk, and has no anti-bot wall. The metadata we need is the same data,
available without fighting anyone's edge.

**Alternatives considered.**
- *Bypassing the Cloudflare challenge.* Rejected: it violates ACM's terms of service, it's
  an unwinnable arms race against a moving target, and it's unnecessary because the same
  metadata is freely available elsewhere.
- *Sci-Hub.* Rejected: it serves pirated full text, is the subject of active litigation,
  and wiring it in would make the tool a copyright-infringement vector. We only ever want
  *metadata* anyway, which is openly licensed.
- *Keeping ACM as default with Crossref as backup.* Rejected: the default should be the
  source that actually works.

**Status / caveats.** Crossref is the right spine, but it is **thin on abstracts** (many
publishers, ACM among them, don't deposit them). That gap is what the enrichment layer
exists to close (§2). The ACM scraper remains in-tree, fully isolated under
`retrieval/acm/`, in case a run happens from an environment that *can* reach ACM.

---

## 2. Enrichment: DOI-keyed, multi-source, fill-only

**Decision.** After retrieval, a separate stage joins free scholarly APIs on the
**normalized DOI** to fill gaps Crossref leaves: OpenAlex (abstracts, OA links, citation
counts), Unpaywall (OA links, fallback), Semantic Scholar (TLDR summaries). Enrichment
only ever fills a field that is currently empty — it never overwrites a source's data.

**Why.** Ranking weights the abstract (§5), and synthesis reads abstracts/TLDRs, so a
missing abstract directly degrades both. The DOI is a perfect universal join key: the same
work from Crossref and from OpenAlex collapses to the same normalized DOI, so enrichment is
a clean key-based merge rather than fuzzy title matching. Splitting responsibilities —
OpenAlex for the batchable bulk fields, Unpaywall only as an OA fallback, Semantic Scholar
only for TLDRs — keeps each API's load minimal and respects their batch limits.

**Alternatives considered.**
- *A single enrichment source.* Rejected: no one free API covers abstracts + OA + citations
  + TLDR well; coverage gaps differ, so layering improves fill rate.
- *Scraping abstracts from publisher pages.* Rejected: same Cloudflare/ToS problems as §1.
- *Fuzzy (title/author) joins.* Rejected: DOIs are exact and already present; fuzzy joins
  introduce false merges.

**Status / caveats.** Abstract coverage is bounded by OpenAlex; a paper absent there may
stay abstract-less and rank on title alone.

---

## 3. Enrich abstracts *before* ranking, TLDRs *after*

**Decision.** Abstract/OA/citation enrichment runs over **all** candidates before ranking.
TLDR enrichment runs **after** ranking, over only the top-K slice synthesis will read.

**Why.** The ranker's 0.2 abstract term is worthless if abstracts arrive after scoring, so
that enrichment must precede ranking. TLDRs, by contrast, are only consumed by synthesis,
which only sees the top-K — enriching all candidates with TLDRs would be wasted API calls
on papers that get dropped.

**Alternatives considered.** Enriching everything up front (simpler, but wasteful on TLDRs);
enriching nothing until synthesis (breaks the abstract-aware ranking). The split is the
cost/quality sweet spot.

---

## 4. On-disk DOI cache, with negative markers

**Decision.** `enrichment_cache.json` maps each normalized DOI to what was fetched. It
stores both **positive** entries (abstract, oa_url, cited_by_count, tldr) and **negative**
markers (`openalex_checked`, `unpaywall_checked`, `s2_checked`).

**Why.** Re-runs and overlapping queries hit the same DOIs constantly; caching makes them
fast and offline-friendly. The negative markers are the subtle part: without them, every
re-run would re-query the APIs for DOIs we *already know* return nothing, which is the
common case (most papers have no Unpaywall OA link, etc.). A warm cache run dropped from
seconds of redundant network calls to effectively instant once negative caching was added.

**Status / caveats.** The cache never expires; delete the file to force a clean re-fetch.
For a long-lived deployment we'd add TTLs (§ Future directions).

---

## 5. Ranking: lexical relevance, amplified by citation impact

**Decision.** `rank()` scores each paper by token-frequency **cosine similarity**, fused
late as `relevance = 0.8 × title + 0.2 × abstract` (title-only when no abstract exists),
then multiplies in a **citation-impact** term: `final = relevance × (1 + β × impact)`,
where `impact = log1p(citations) / log1p(max_citations_in_pool)` and `β = 1.0`.

**Why the impact term.** The candidate pool is dozens of near-identically-titled papers
("… program comprehension"), so pure title cosine is almost a coin-flip among them — it
floated 0–2-citation workshop notes above 100-citation seminal work. We already fetch
`cited_by_count` (Crossref + OpenAlex) and were discarding it. Folding it in pulls the
field's important papers to the top, which matters doubly because synthesis only reads the
top slice (§3, §8) — better top-K, better review.

**Why multiplicative, not additive.** A weighted *sum* (`α·relevance + (1−α)·citations`)
would surface a famous-but-off-topic paper whose citations outweigh its low relevance.
Multiplying keeps relevance as a **gate**: a paper that doesn't match scores ~0 no matter
how cited it is. Citations only ever *reorder papers that already match*. The `log1p` plus
pool-relative normalisation stops one mega-cited outlier from flattening everyone else.

**Why lexical cosine for relevance despite being simple.** The ranker is *re-ordering an
already-relevance-filtered set*, not doing primary retrieval — Crossref returns candidates
by relevance, and query-understanding (§6) has already mapped the query into the terms the
papers use. A weak ranker nudging a good candidate list is very different from a weak ranker
searching from scratch. In exchange we get something zero-dependency, instant, deterministic,
and fully offline.

**Why not TF-IDF or BM25.** Both lean on corpus document-frequency statistics (IDF). Our
candidate pool is ~50 short, topically-homogeneous **titles** where within-document term
frequency is almost always 1 and "rare vs common term" carries no signal — IDF is degenerate
on exactly this shape of data, so the added machinery wouldn't earn its keep.

**Known limitation.** Pure lexical matching is blind to synonyms and morphology: it cannot
match *"program understanding"* to *"program comprehension."* This is mitigated upstream by
query-understanding canonicalizing terminology, but not eliminated. Scores therefore look
modest (≈0.15–0.35), which is expected, not a bug.

**Caveat — recency bias.** Citations accrue with age, so the impact term tilts toward
older work. For survey/theory queries that's usually desirable (seminal papers *should*
lead); for a "latest advances" query it can bury fresh work. `β` is the dial — lower it, or
make it intent-aware (§6), if recency matters more than canon. Papers with no citation data
(`cited_by_count = None`) are treated as 0 impact, i.e. ranked on relevance alone.

**Status.** Good enough as a default; the clearest upgrade lever in the codebase (§ Future
directions → semantic ranking).

---

## 6. Query understanding: translate lay → academic terminology

**Decision.** An LLM step rewrites the user's natural-language question into **two** search
phrases at different breadths, plus keywords and an intent label:
- `refined_query` — the bare canonical topic (*"program comprehension"*), for **recall**.
  Still forbids methodological framing words ("metrics", "evaluation", "framework").
- `focus_query` — a narrower phrase that **keeps** the user's specific angle, including the
  qualifier words `refined_query` drops (*"assessing program comprehension"*), for
  **precision**. `null` when the question is a plain topic with no narrowing facet.

**Why two phrases.** The framing-word ban (added because the model used to drift "program
comprehension" → "metrics for program comprehension" and over-narrow) turned out to *also*
strip the legitimate substance of a faceted question — a user asking *what characteristics
mean someone understood an algorithm* was reduced to the bare topic, so retrieval returned
the whole program-comprehension universe and none of the assessment-specific work. Splitting
into a broad phrase (keeps recall, keeps the ban) and a focused phrase (restores the facet)
resolves the tension instead of trading one failure for the other. The two phrases feed the
two-pass retrieval in §7.

**Alternatives considered.** A single phrase at one breadth — always either too broad
(irrelevant papers) or too narrow (lost recall); this was the original design and the source
of the "intent destroyed" failure. Loosening the framing-word ban on `refined_query` itself —
rejected, it reintroduces the over-narrowing the ban prevents.

---

## 7. Retrieval: two-pass (broad + focused), merged by DOI

**Decision.** For a faceted query we issue **two** Crossref `query.bibliographic` searches —
the broad `refined_query` (recall) and the narrower `focus_query` (precision, §6) — and union
the results, deduped by DOI. The pooled candidates are enriched and ranked together; `--limit`
is applied *after* ranking so the best of a richer pool survives. A plain-topic query (no
`focus_query`) runs a single pass exactly as before. Each phrase still has only generic
container words stripped, never reduced to a bag of isolated keywords.

**Why.** A single bare-topic search returns the whole topic and buries the user's actual
angle (§6); a single narrow search loses the seminal broad papers (the original "weak papers"
complaint, §5). Running both and letting ranking arbitrate gets the on-intent *and* the
canonical work. We keep phrases rather than keyword bags because Crossref ranks across *all*
terms, so connective context matters: *"natural language **to** SQL"* retrieves far more
precisely than the three bare tokens. Keywords remain a per-phrase fallback when a phrase
reduces to nothing.

**Status / caveats.** The focused pass costs one extra Crossref request and enriches a
slightly larger pool — accepted, since precision on faceted queries was the whole point.
Merge is first-occurrence-wins on DOI; metadata is identical across passes (both Crossref),
so order doesn't matter — ranking re-sorts the union regardless.

---

## 8. Synthesis input is hard-bounded

**Decision.** `select_for_synthesis` caps the synthesis set at 50 papers *and* enforces a
token budget on the assembled prompt; abstracts are truncated in the payload.

**Why.** A user once requested 1000 papers and the prompt overflowed the model context
window (`456955 > 262144`) and crashed. Synthesis is summarizing a focused set into a few
paragraphs anyway — hundreds of papers are neither synthesizable nor affordable. The cap and
budget make overflow structurally impossible regardless of the requested top-K.

---

## 9. Dates: `None`, never the string `"None"`

**Decision.** Missing publication years are Python `None`, rendered downstream as `n.d.`.

**Why.** Crossref encodes an unknown date as `{"date-parts": [[null]]}`. Naively calling
`str()` on that inner `null` yields the literal string `"None"`, which is *truthy* — so it
defeats every `year or "n.d."` fallback and produces citations like `[Author, None]`. Year
parsing guards the null explicitly. If you touch date handling, preserve this invariant.

---

## 10. Cross-cutting reliability conventions

- **Graceful degradation.** A single malformed record or a failing enrichment source must
  never abort a run — catch, log to stderr, and carry on. Retrieval/processing/enrichment all
  follow this, so partial results beat no results.
- **Polite pool.** Every outbound API call carries a descriptive `User-Agent` and a
  `mailto`/`email`. It buys faster, more reliable service and is required by Unpaywall.
- **stdout discipline.** All progress logs go to **stderr**; only the review or the `--json`
  document goes to **stdout**, so the tool composes cleanly in pipelines.
- **Small files.** Modules are kept ≤150 lines and split along behavioral seams (HTTP vs
  parsing vs orchestration), not arbitrary line counts. This is for readability and
  onboarding, and it's why `enrichment/`, `retrieval/crossref/`, and `retrieval/acm/` are
  packages rather than single files.

---

## Future directions

Roughly ordered by expected value-for-effort.

1. **Semantic ranking (highest leverage).** Replace — or fuse with — the lexical cosine
   using dense embeddings, which directly fixes the synonym/morphology blindness of §5.
   Because the project already depends on Ollama, the embeddings endpoint
   (`nomic-embed-text`, `mxbai-embed-large`, …) gives this with **no new Python dependency**.
   Suggested shape: a `--rank {lexical,semantic,hybrid}` switch where `hybrid` does
   reciprocal-rank fusion over both, keeping the deterministic offline lexical path as the
   default/fallback. Only `ranking.py` changes; `RankedRecord` stays the same.

2. **Cross-encoder re-rank of the top-K.** After a first-pass rank, jointly score
   `(query, title+abstract)` with a small cross-encoder. Trivial cost at ≤50 docs, best
   relevance gains. Pairs naturally with (1).

3. **Semantic Scholar relevance search for recall.** Use S2's `/paper/search` as an
   additional retrieval source (not just enrichment) for conceptual queries where Crossref's
   lexical relevance under-recalls. Merge by DOI with the existing dedup.

4. **More retrieval sources.** arXiv, DBLP, or OpenAlex-as-primary (it already powers
   enrichment). The `BaseTranslator` + router design makes adding one a self-contained change
   (see README → Extending the tool).

5. **Use Crossref's own relevance score.** Crossref returns a `score` per item that we
   currently discard; fusing it with our re-rank is nearly free.

6. **Cache TTLs and query-result caching.** Expire enrichment entries, and cache whole query
   result sets (not just per-DOI enrichment) for repeated runs.

7. **Retrieval filters.** Year range, venue, open-access-only, type filters — cheap to add
   to the Crossref query, useful for focused reviews.

8. **An evaluation harness.** A small set of queries with relevance judgments so ranking
   changes (especially 1–2) can be measured instead of eyeballed. Without this, "better
   ranking" is a vibe; with it, it's a number.

9. **Synthesis grounding.** Verify that every inline citation in the generated review maps to
   a paper actually in the input set, to catch the LLM inventing or mis-attributing claims.
