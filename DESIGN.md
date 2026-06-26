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

## 5. Ranking: semantic + lexical relevance, amplified by citation impact

**Decision.** `rank()` scores relevance as a hybrid of a **semantic** and a **lexical**
signal, then multiplies in a **citation-impact** term:

```
lexical    = 0.8 × title_cosine + 0.2 × abstract_cosine    (token-frequency; title-only if no abstract)
semantic   = cosine(SBERT(query), SBERT(title+abstract))   (dense embeddings; see semantic.py)
relevance  = 0.8 × semantic + 0.2 × lexical                (lexical-only when embeddings unavailable)
final      = relevance × (1 + β × impact),  impact = log1p(citations) / log1p(max_in_pool),  β = 1.0
```

The semantic term uses Sentence-BERT (`all-MiniLM-L6-v2`) via `sentence-transformers`,
run in-process (no server) with vectors cached on disk by DOI. It degrades gracefully:
no `sentence-transformers`, or `--no-semantic`, falls back to the lexical-only path.

**Why semantic.** Token overlap can't tell *"program comprehension"* (software) from
*"reading comprehension program"* (education) or *"democratic peace research program"*
(poli-sci) — they share the tokens `{program, comprehension}`, so a broad query dragged
cross-domain noise into the top results. Dense embeddings place those in different regions
of vector space (measured: real SE-comprehension titles score ≈0.4–0.5 against the query;
the noise titles ≈0.01–0.18), so the noise sinks. This also fixes the long-standing
synonym blindness below (*"program understanding"* ≈ *"program comprehension"*).

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

**Why keep lexical at all.** It's the deterministic, zero-dependency, fully-offline floor:
when `sentence-transformers` isn't installed, the model can't load, or the user passes
`--no-semantic`, ranking still works (lexically) rather than failing. It also contributes
the 0.2 term in the hybrid, so an exact phrase match still counts alongside meaning. We
avoid TF-IDF/BM25 for that lexical term: both lean on document-frequency statistics (IDF),
and our pool of short, topically-homogeneous titles (term frequency ≈ 1, no "rare vs common"
signal) is exactly the shape where IDF is degenerate and wouldn't earn its keep.

**Why blend, not replace.** Pure embedding similarity occasionally rates a topically-vague
paper above an exact match. Keeping 0.2 lexical (and the citation gate) preserves a hard
signal for literal phrase hits and a graceful-degradation path, at no real cost.

**Caveat — recency bias.** Citations accrue with age, so the impact term tilts toward
older work. For survey/theory queries that's usually desirable (seminal papers *should*
lead); for a "latest advances" query it can bury fresh work. `β` is the dial — lower it, or
make it intent-aware (§6), if recency matters more than canon. Papers with no citation data
(`cited_by_count = None`) are treated as 0 impact, i.e. ranked on relevance alone.

**Caveat — embedding cost & determinism.** The first run encodes the pool (CPU is fine for
short titles; vectors cache by DOI, so re-runs are instant). Embedding ranking is *not*
bit-for-bit deterministic across hardware/model versions the way lexical is — `--no-semantic`
is the escape hatch for reproducible/offline runs. SBERT runs on CPU by default; an Intel
XPU / CUDA torch build is auto-detected (see `semantic._pick_device`).

**Status.** Semantic hybrid is the default when `sentence-transformers` is present. The
semantic term also carries the **sense-aware** machinery — augmented embedding + contrastive
down-weighting + the soft relevance floor — when a query was disambiguated (§12). Next levers:
a cross-encoder re-rank of the top-K, or reciprocal-rank fusion instead of the weighted blend
(§ Future directions).

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

## 11. LLM provider: pluggable behind a one-method shim

**Decision.** Query understanding and synthesis get their chat client from `llm.make_client(provider, host, api_key)`. Two providers ship: **ollama** (the official client, local or cloud) and **groq** (a thin `requests` wrapper over Groq's OpenAI-compatible `/chat/completions`). Both expose the identical `chat(model, messages, options)` → `response.message.content` shape the call sites already used, so neither `QueryUnderstanding` nor `Synthesizer` knows which backend is live.

**Why.** The two LLM call sites had a duplicated `_make_client` and assumed the ollama response shape. Adding a second backend behind the *same* shape means the swap is a one-line constructor change in each, with zero branching in the prompt/parse logic. `config.provider_defaults(provider)` centralises the host/key/default-model triple so `--provider groq` "just works" without the user also passing `--model`/`--host`.

**Why a `requests` shim, not the Groq SDK.** Groq's API is OpenAI-compatible, so a 30-line POST + response-adapter needs no new dependency (`requests` is already in-tree) and avoids pulling the `groq`/`openai` SDKs and their transitive deps for one endpoint. The shim mimics ollama's `.message.content` accessor rather than leaking `choices[0].message.content` upward, keeping the abstraction seam clean.

**Caveats.** Groq is cloud-only — selecting it without `GROQ_API_KEY` is a hard error (no local fallback, unlike Ollama). Errors are normalised to `llm.ChatError`, which synthesis catches alongside `ollama.ResponseError`; the Groq branch surfaces the API's own error body (bad model id, rate limit, auth) verbatim. One gotcha baked in: a `requests.Response` with a 4xx/5xx status is *falsy*, so the error handler tests `is not None`, not truthiness, to read the status code.

---

## 12. Disambiguation + sense-aware contrastive ranking

**Decision.** When query understanding judges a query **ambiguous**, it enumerates a list of
`interpretations` — distinct senses that would retrieve *different bodies of literature*. To
make that enumeration reliable, the query-understanding call runs at a **modest temperature
(0.3)**: listing the possible senses is a divergent task, and greedy temp-0 decoding commits to
the single most-salient framing and under-lists alternatives — so a missed sense never becomes a
contrastive negative. Temperature is kept low to protect the convergent fields (`refined_query`/
`keywords`) in the same call.

In an interactive run the interpretations become a pick-list (`cli_display.choose_interpretations`)
and the user may pick **one or several** senses — related readings can be combined. Each chosen
sense drives three things:
1. **retrieval** — its discriminating `refined_query` / `focus_query` / `keywords` drive a search
   pass; multiple chosen senses run a pass each and union their literatures (merged by DOI);
2. **the semantic embedding** — the chosen sense(s) are *appended* to the raw query before encoding
   (augment, never replace), so the pick reaches the dominant 0.8 semantic signal;
3. **contrastive down-weighting** — the senses the user *did not* pick become negative anchors:
   `score = cos(doc, chosen) − contrast × max cos(doc, rejected)` (`semantic.py`), so papers in
   a rejected sense sink and, via the relevance floor, can drop out entirely.
Non-interactive runs (`--json`, piped, `--no-interactive`) auto-pick the primary sense and log
it; unparseable interactive input re-prompts rather than silently overriding the user. A separate
**soft relevance floor** (`--min-relevance`, applied to relevance *before* the citation boost)
trims the off-topic tail regardless of disambiguation.

**Why.** Bag-of-words bibliographic search can't separate the two meanings of a term like
*"algorithmic understanding"* — the computer-science sense (understanding how an algorithm
works) and the sociotechnical sense (algorithmic management / governance / media literacy)
share the literal tokens, so retrieval returns the union and the social cluster floods the
results. The earliest fix (just disambiguating the *search* terms) barely helped: the search
and the 0.2 lexical score moved, but the **0.8 semantic** ranker still embedded the raw query,
which is semantically a twin of every *"understanding algorithmic ‹X›"* title. Routing the pick
into the embedding (augment) and actively pushing *away* from the rejected sense (contrast) is
what finally sinks the wrong-sense papers.

**Why augment, not replace, the embedding.** Replacing the raw query with the bare
`refined_query` was tried and reverted — it discards the user's disambiguating anchors
("characteristics", "well enough") and degraded the review. Augmenting keeps those anchors and
*adds* a sense signal, and only when a sense was actually chosen; unambiguous queries embed the
raw query unchanged.

**Why contrastive (subtract the rejected sense), and why `max`.** A positive anchor alone
raises the right papers but leaves wrong-sense papers in place when they're independently close
to the raw query (e.g. *"Understanding of Algorithmic Decision-Making"*). Subtracting similarity
to the rejected sense is what demotes them. We subtract the *closest* rejected sense (`max`, not
mean) because one strong wrong-sense match is enough to disqualify a paper. Flooring on relevance
*before* the citation multiplier (not on `final_score`) ensures a heavily-cited off-topic paper
can't survive on citations alone.

**Alternatives considered.**
- *A hard relevance floor with no contrastive term.* Rejected as the primary lever: an
  empirical sweep showed the society-noise papers score in the *same* 0.40–0.49 band as the
  good papers, interleaved — a floor only trims the tail, it can't de-interleave. Kept as a
  complementary, tunable trim (`--min-relevance`).
- *Self-reported confidence scores from the LLM.* Rejected: confidence-on-correctness is poorly
  calibrated. Ambiguity *detection* (enumerating distinct senses) is a different, reliable task,
  so that's what the model is asked for instead.
- *Always prompting.* Rejected: most queries are unambiguous; the model returns `[]` and the
  pick-list never appears. Prompting is also gated on a TTY so scripts never hang.

**Status / caveats.**
- **Interpretation quality is the ceiling.** The model first mis-axed the senses (offered
  "CS-ed vs AI/ML interpretability", missing the real *algorithms-in-society* axis); the prompt
  now explicitly nudges the computational-vs-sociotechnical split and forbids reusing the bare
  ambiguous term for any sense. If the model still fails to enumerate the noisy sense, the
  contrastive term has nothing useful to subtract — the LLM step, not the ranker, is the limit.
- **One rejected sense doesn't cover every noise family.** Subtracting *"algorithmic literacy"*
  demotes literacy papers but not *decision-making / management / music* — a different family the
  model didn't list. Those stubborn neighbours persist near the top; fully removing them needs
  either a stricter cut (losing recall) or the model enumerating more rejected senses.
- **Defaults are empirical, not magic.** `--contrast 0.4` / `--min-relevance 0.25` were chosen
  from a `contrast × floor` sweep as the loosest setting that recovers borderline-relevant work
  (the conceptual-vs-algorithmic / skill-vs-understanding cluster) while keeping the synthesis
  top-K mostly clean. They are pool-sensitive (the LLM's `refined_query` varies run to run,
  changing the score distribution), so they're exposed as flags to tune per query. A relative
  floor (keep within Y of the top score) would be more robust than an absolute one — a candidate
  for future work, alongside the evaluation harness that would let us *measure* these dials.

---

## 13. Relevance tiering: label the output, don't filter it

**Decision.** After ranking, a second LLM pass (`tiering.py`) labels each ranked paper with a
relevance **tier** — `high` / `moderate` / `tangential` / `irrelevant` — judged against the
research *goal*, not raw similarity. The tiers are **purely additive**: nothing is ever
removed. They drive two things — a grouped *Relevance Tiers* section in the output (the
score-ranked table is left intact), and the order synthesis fills its budget (highly-relevant
papers first, stable within a tier, via `prioritize_by_tier`). On by default, including under
`--no-synthesis`; `--no-tier` opts out.

**Why label instead of filter.** This tool optimises for **recall** — the cost of dropping a
relevant paper (a false negative) is far higher than carrying a few off-topic ones, because the
synthesizer already ignores list noise. So a precision *filter* that deletes papers is the wrong
shape: a misjudged title/abstract would silently remove a key result. A *labelling* layer gets
the same legibility — the noise is visually quarantined at the bottom of the grouped section —
without paying the recall cost, and it lets synthesis preferentially draw from the on-goal papers.
The tier judgement adds signal the cosine score lacks: similarity can't tell that a heavily-cited
paper sharing the query's vocabulary is actually from the wrong field, but an LLM reading the
abstract against the goal can call it `irrelevant` while still leaving it in the list.

**Why these four tiers.** Three graded "relevant" bands (high/moderate/tangential) plus a
distinct `irrelevant` for genuine wrong-domain noise. The split between `tangential` (same domain,
peripheral) and `irrelevant` (different field, keyword/retrieval noise) matters because they read
very differently to a researcher scanning the list — peripheral-but-on-topic work is worth a
glance; cross-domain noise is not. `intent` is fed to the classifier so, e.g., an
`implementation` query treats purely theoretical/linguistic papers as moderate/tangential rather
than high.

**Why it's recall-safe in the details.** (1) The prompt tells the model to round *up* when
genuinely unsure between two tiers. (2) A paper the model fails to classify at all defaults to
`tangential`, not `irrelevant` — a non-classification shouldn't assert outright irrelevance, but
it also shouldn't inherit a strong label it was never given. (3) Synthesis prioritisation only
*re-orders within* the `top_k` the user already chose; it never pulls a paper from outside that
slice, so the review can't cite a paper that isn't in the shown table. (4) A failed batch leaves
its papers at the default rather than aborting — the pass degrades gracefully like every other
networked stage (§10).

**Why a default stage, even under `--no-synthesis`.** Query understanding (§6) already calls the
LLM unconditionally, so `--no-synthesis` was never an "offline / no-LLM" switch — it means "don't
write the prose review". Tiering sits alongside query understanding as a default LLM stage that
shapes the *list* output, which is exactly what a `--no-synthesis` user is left with. A truly
LLM-free run is `--no-synthesis --no-tier` (and `--no-enrich` to skip the enrichment APIs too).

**Alternatives considered.**
- *An LLM filter that deletes off-topic papers.* Rejected — the recall liability above; the user
  explicitly wants breadth, and the synthesizer already tolerates list noise.
- *Tiering only the synthesis input.* Considered; the grouped section over the *whole* shown list
  is more useful (it tells you the shape of the retrieval), and it's the same LLM call either way.
- *Reusing the cosine score to bucket (thresholds).* Rejected — the score measures similarity, not
  goal-relevance; the wrong-field-but-high-similarity case (§5's cross-domain noise) is exactly
  what a threshold can't catch but the LLM can.

**Status / caveats.** Batched at 50 papers/call (low temperature for consistency); large lists
cost a few calls. Tier quality is bounded by the model's judgement of the goal — same ceiling as
query understanding. Like all LLM output it's mildly non-deterministic, so a borderline paper can
shift one tier between identical runs.

---

## Future directions

Roughly ordered by expected value-for-effort.

1. **Semantic ranking — DONE (§5).** Implemented as an SBERT hybrid (`semantic.py`),
   blended `0.8 semantic + 0.2 lexical`, with `--no-semantic` and a lexical fallback. We
   chose `sentence-transformers` over Ollama embeddings because Ollama *cloud* (the configured
   chat backend) serves chat models only — its embed endpoint 401s — so an in-process SBERT
   avoids forcing a separate local Ollama daemon. Remaining refinements: reciprocal-rank
   fusion instead of the fixed-weight blend; making the blend weight intent-aware.

2. **Cross-encoder re-rank of the top-K.** After the first-pass hybrid rank, jointly score
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
