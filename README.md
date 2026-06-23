# lr_tool — Literature Review Tool

Give it a natural-language research question; get back a ranked list of papers (with
abstracts, open-access links, and citation counts) plus an AI-generated literature
review. Everything runs from one command.

```bash
python cli.py "what makes an algorithm easy to understand"
```

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Architecture (for contributors)](#architecture-for-contributors)
- [Data model](#data-model)
- [Enrichment & caching](#enrichment--caching)
- [Extending the tool](#extending-the-tool)
- [Conventions](#conventions)
- [Known limitations](#known-limitations)

---

## How it works

```
query
  → [1] query understanding   (LLM: lay phrasing → broad + focused academic phrases)
  → [2] retrieval             (Crossref REST API, two-pass broad+focused → merged CSLRecords)
  → [2b] processing           (schema normalization)
  → [2c] enrichment           (OpenAlex / Unpaywall fill abstracts, OA links, citations)
  → [3a] semantic scoring      (SBERT embeddings; skipped with --no-semantic)
  → [3] ranking               (0.8·semantic + 0.2·lexical, × citation impact)
  → [3b] TLDR enrichment       (Semantic Scholar, top-K only)
  → [4] synthesis             (LLM writes the review)
  → Markdown / JSON output
```

1. **Query understanding** — an Ollama LLM rewrites your question into the canonical
   terms scholarly papers actually use (e.g. *"how well someone grasps an algorithm"* →
   *"program comprehension"*), extracts keywords, and classifies intent. For a faceted
   question it also emits a narrower `focus_query` (e.g. *"assessing program comprehension"*)
   that preserves the specific angle the broad term drops.
2. **Retrieval** — queries the **Crossref** REST API, the canonical DOI metadata registry
   that publishers (ACM included) deposit to at publication time. No scraping, no anti-bot
   walls. Faceted queries run **two passes** — the broad topic (recall) and the focused
   phrase (precision) — merged and deduped by DOI, so the user's specific angle reaches the
   results instead of being flattened to the bare topic. An ACM Digital Library scraper
   remains available as a best-effort fallback (`--source acm`), but `dl.acm.org` is behind
   Cloudflare and it usually returns nothing.
3. **Enrichment** — Crossref is thin on abstracts, so we fill the gaps by joining other
   free APIs on the **normalized DOI**:
   - **OpenAlex** → abstracts (rebuilt from an inverted index), OA links, citation counts (batched)
   - **Unpaywall** → OA links, as a fallback when OpenAlex had none
   - **Semantic Scholar** → one-line TLDR summaries for the top-ranked papers
   Results are cached on disk by DOI (`enrichment_cache.json`), so re-runs are fast.
   Disable the whole stage with `--no-enrich`.
4. **Ranking** — scores relevance as a hybrid of a **semantic** signal (SBERT embedding
   cosine, `all-MiniLM-L6-v2`) and a **lexical** signal (`0.8 × title + 0.2 × abstract`
   token cosine), blended `0.8 × semantic + 0.2 × lexical`, then amplified by citation impact
   (`final = relevance × (1 + impact)`). Embeddings separate *"program comprehension"*
   (software) from *"reading comprehension program"* (education) — token overlap can't.
   Runs in-process via `sentence-transformers`, vectors cached by DOI; falls back to
   lexical-only when the library is absent or `--no-semantic` is passed.
5. **Synthesis** — the LLM writes a structured review over a bounded slice of the top
   papers (capped so the prompt can never exceed the model's context window).
6. **Output** — writes a Markdown file and can also emit structured JSON.

---

## Requirements

- Python 3.11+ (the code uses `X | None` unions and `asyncio.timeout`)
- [Ollama](https://ollama.com) running locally **or** an `OLLAMA_API_KEY` for Ollama cloud
- Internet access to the scholarly APIs (Crossref, OpenAlex, Unpaywall, Semantic Scholar)
- `playwright install chromium` is only needed for the optional `--source acm` fallback

---

## Installation

```bash
git clone <repo-url>
cd lr_tool
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
playwright install chromium   # only needed for the ACM fallback
```

`requirements.txt` includes `sentence-transformers` (pulls PyTorch, ~200 MB) for the
semantic ranking signal. The model (`all-MiniLM-L6-v2`, ~80 MB) downloads on first use and
is cached. To skip all of this, drop that line and run with `--no-semantic` — the tool then
ranks lexically with no extra dependency. An Intel XPU (Arc) or CUDA torch build is
auto-detected for faster encoding; plain CPU is fine for the small batches we embed.

---

## Configuration

Create a `.env` file in the project root (it is gitignored):

```env
OLLAMA_API_KEY=your_key_here
CONTACT_EMAIL=you@example.com
```

- **`OLLAMA_API_KEY`** — if set, the tool talks to Ollama cloud (`https://ollama.com`).
  Without it, it falls back to a local Ollama at `http://localhost:11434`.
- **`CONTACT_EMAIL`** — sent to the scholarly APIs as `mailto` (Crossref/OpenAlex) and
  `email` (Unpaywall). It opts you into their faster "polite pool" and is *required* by
  Unpaywall. Optional but recommended; a placeholder is used if unset.

`.env` parsing is intentionally dependency-free — see `config.py`. There is no
`python-dotenv` requirement.

---

## Usage

```
python cli.py [-h] [--source {crossref,acm}] [--model MODEL] [--ollama-host URL]
              [--limit N] [--no-synthesis] [--no-enrich] [--no-semantic]
              [--no-save] [--out-dir DIR] [--workers N] [--json]
              query
```

| Argument | Description |
|---|---|
| `query` | Research query in natural language (required) |
| `--source {crossref,acm}` | Paper source (default: `crossref`; `acm` is a best-effort scraper fallback) |
| `--model MODEL` | Ollama model name (default: `gemma4:31b-cloud`) |
| `--ollama-host URL` | Ollama server URL (default: auto-detected from `.env`) |
| `--limit N` | Max papers to retrieve (default: 50; use `0` for no limit). Crossref returns by relevance, so the first ~50 carry the signal. |
| `--no-synthesis` | Skip LLM synthesis; only show the ranked paper list |
| `--no-enrich` | Skip metadata enrichment (abstracts/OA/TLDR); faster, offline-friendly |
| `--no-semantic` | Skip SBERT embedding ranking; use lexical scoring only (deterministic, no model load) |
| `--no-save` | Do not write a Markdown file to disk |
| `--out-dir DIR` | Directory for the output `.md` file (default: current directory) |
| `--workers N` | Concurrent page fetches; **only used by `--source acm`** (default: 4) |
| `--json` | Emit full structured JSON to stdout — non-interactive, pipeline-friendly |

### Examples

```bash
# Interactive run — prompts you for how many papers to synthesize
python cli.py "natural language to SQL methods"

# Just the ranked list, no synthesis
python cli.py "transformer architectures survey" --no-synthesis

# Non-interactive JSON (scripting / pipelines)
python cli.py "graph neural networks" --json

# Skip enrichment for a fast, offline-friendly run
python cli.py "federated learning privacy" --no-enrich --no-synthesis

# Cap retrieval and pick a different model
python cli.py "vision transformers" --limit 50 --model llama3:8b
```

> **stdout vs stderr:** all progress logs go to **stderr**; only the review (interactive)
> or the JSON document (`--json`) goes to **stdout**. So `python cli.py ... --json > out.json`
> produces a clean JSON file.

---

## Output

**Interactive mode** prints the refined query, the ranked list (score, authors, year, DOI),
and the generated review, then saves a Markdown file named after the query (skip with
`--no-save`).

**`--json` mode** prints one JSON object to stdout:

```json
{
  "structured_query": { "refined_query": "...", "focus_query": "...", "keywords": ["..."], "intent": "theory" },
  "records_retrieved": 50,
  "ranked": [
    {
      "title": "...", "authors": ["..."], "year": "2021", "venue": "...",
      "DOI": "...", "URL": "...", "oa_url": "...", "tldr": "...",
      "cited_by_count": 42, "score": 0.31, "title_score": 0.33,
      "abstract_score": 0.22, "abstract_missing": false
    }
  ],
  "review": "..."
}
```

`records_retrieved` is the full retrieved count; `ranked` contains only the top-K papers.

---

## Architecture (for contributors)

The codebase is deliberately split into small, single-purpose modules (none exceeds
~150 lines). Each stage of the pipeline maps to a file or package, and the orchestration
lives in `pipeline.py`.

> **Why is it built this way?** See [`DESIGN.md`](DESIGN.md) for the rationale behind the
> major decisions (Crossref over scraping, DOI-keyed enrichment, lexical ranking, the
> synthesis cap, …), the alternatives that were weighed, and where the tool could go next.
> Read the relevant entry before changing a stage — most choices have a non-obvious reason.

```
lr_tool/
├── cli.py                  # Entry point: parse args, dispatch JSON vs interactive flow
├── cli_args.py             #   └─ argparse parser definition + TOP_K_DEFAULT
├── cli_display.py          #   └─ terminal rendering (print_ranked) + the top-k prompt
│
├── pipeline.py             # Orchestrates the stages (LiteratureReviewPipeline)
├── search_query.py         #   └─ build a Crossref search string from a query phrase
├── result.py               #   └─ assemble the final result dict (JSON/Markdown shape)
│
├── config.py               # Loads .env; exposes OLLAMA_*, DEFAULT_MODEL, CONTACT_EMAIL, USER_AGENT
├── models.py               # Pydantic data models (CSLRecord, RankedRecord, …)
├── query_understanding.py  # [1]  LLM query refinement → StructuredQuery
│
├── retrieval/
│   ├── base.py             #      BaseTranslator ABC (the source interface)
│   ├── router.py           #      source registry: name → translator class
│   ├── crossref/           # [2]  default source
│   │   ├── client.py       #        HTTP layer (session, retries, endpoint)
│   │   ├── parse.py        #        Crossref item → CSLRecord (+ type/front-matter filters)
│   │   └── translator.py   #        cursor-paging loop + CrossrefTranslator
│   └── acm/                # [2]  best-effort fallback scraper (Cloudflare-prone)
│       ├── config.py       #        endpoints, timeouts, browser fingerprint
│       ├── parse.py        #        BeautifulSoup extraction
│       ├── browser.py      #        navigation + Cloudflare-challenge handling
│       ├── fetch.py        #        per-page fetch with retries + concurrency
│       └── search.py       #        async search loop + ACMScraper
│
├── processing.py           # [2b] normalize raw records to a strict CSL contract
├── enrichment/             # [2c] DOI-keyed metadata enrichment
│   ├── doi.py              #        the one DOI normalizer (join + cache key)
│   ├── http.py             #        shared session helper + timeout
│   ├── cache.py            #        on-disk DOI cache (positive + negative markers)
│   ├── openalex.py         #        abstracts / OA / citations (batched)
│   ├── unpaywall.py        #        OA link fallback (parallel, per-DOI)
│   ├── semantic_scholar.py #        TLDR summaries (batched)
│   └── stages.py           #        public entry points: enrich_pre_rank, enrich_tldr
│
├── ranking.py              # [3]  hybrid (semantic + lexical) × citation-impact ranking
├── semantic.py             #   └─ SBERT embeddings + DOI-cached vectors (semantic signal)
├── synthesis.py            # [4]  LLM review generation + synthesis-set selection
├── urls.py                 #      is.gd URL shortening (used by output)
└── output.py               #      Markdown file generation
```

### Where to start reading

- To understand the **flow**, read `pipeline.py` top to bottom — every stage is a labelled
  method call, and the imports point at the module that does the work.
- To understand **what a paper looks like** at each step, read `models.py` (`CSLRecord`).
- Each package has a docstring in its `__init__.py` explaining the split and the rationale.

---

## Data model

Defined in `models.py` (Pydantic):

| Model | Purpose |
|---|---|
| `StructuredQuery` | LLM output: `refined_query` (broad/recall), `focus_query` (narrow/precision, optional), `keywords`, `intent` |
| `Author` | `given` / `family` name parts |
| `CSLRecord` | One paper, in a CSL-JSON-ish shape. The currency of the whole pipeline. |
| `MetadataMissingness` | Flags (`abstract_missing`, `oa_missing`, `tldr_missing`) the ranker and output consult |
| `RankedRecord` | A `CSLRecord` plus its `title_score` / `abstract_score` / `final_score` |

`CSLRecord.issued` is a **year string or `None`** — never the literal string `"None"`
(see `crossref/parse.py:_parse_year`, which guards a Crossref edge case where the date is
present-but-null). Downstream code relies on `issued or "n.d."` working correctly, so keep
that invariant if you touch year parsing.

---

## Enrichment & caching

The enrichment layer never overwrites a field a source already provided — it only fills
blanks. Every enricher degrades gracefully: on any network/parse failure it logs to stderr
and returns the records untouched, so enrichment can never crash a run.

`enrichment_cache.json` (gitignored, written next to the package) maps each normalized DOI
to whatever was fetched. It stores two kinds of entries:

- **Positive** — `abstract`, `oa_url`, `cited_by_count`, `tldr`.
- **Negative** — boolean markers (`openalex_checked`, `unpaywall_checked`, `s2_checked`)
  that record "we already asked this source and it had nothing", so re-runs don't re-query
  DOIs that are known to return nothing.

Delete the file to force a clean re-fetch.

---

## Extending the tool

### Add a new retrieval source

1. Create a class implementing `BaseTranslator` (`retrieval/base.py`):
   ```python
   class MySource(BaseTranslator):
       def search(self, query: str, limit: int | None = None, workers: int = 2) -> list[CSLRecord]:
           ...
   ```
   Return `CSLRecord`s with a normalized lowercase DOI in `.DOI` (that's the enrichment
   join key) and set `source="mysource"`.
2. Register it in `retrieval/router.py`:
   ```python
   _REGISTRY = {"crossref": CrossrefTranslator, "acm": ACMScraper, "mysource": MySource}
   ```
3. Add it to the `--source` choices in `cli_args.py`.

Enrichment, ranking, synthesis, and output are all source-agnostic — they only need DOIs
and the standard `CSLRecord` fields, so nothing else has to change.

### Add a new enricher

Add a module under `enrichment/` that takes `(records, cache, …)`, fills only missing
fields, uses a `*_checked` negative-cache marker, and never raises. Then call it from
`enrichment/stages.py`.

---

## Conventions

- **Files stay small** — aim for ≤150 lines; split along a natural seam (HTTP vs parsing
  vs orchestration) rather than by line count alone.
- **Logs go to stderr** via `print(..., file=sys.stderr)`; **stdout is reserved** for the
  review / JSON so the tool composes in pipelines.
- **Graceful degradation** — a single bad record or a failing enrichment source must never
  abort the run. Catch, log, and carry on (see `processing.py`, `enrichment/stages.py`).
- **DOIs are normalized everywhere** through `enrichment/doi.py:norm_doi` so join and cache
  keys never drift.
- **Polite pool** — every outbound API call carries the `USER_AGENT` and a `mailto`/`email`.

---

## Known limitations

- **Ranking is a semantic+lexical hybrid.** The SBERT term handles synonyms/paraphrase and
  cross-domain disambiguation; the lexical term is the deterministic, offline fallback (used
  when `sentence-transformers` is absent or `--no-semantic` is passed). Citation impact then
  amplifies relevance, which tilts ranking toward older (more-cited) work — see
  [`DESIGN.md` §5](DESIGN.md) for the recency caveat, the blend/`β` dials, and next steps
  (dense embeddings via the Ollama endpoint you already have).
- **Abstracts depend on OpenAlex** coverage; a paper absent from OpenAlex may stay
  abstract-less (it then contributes only its title score to ranking).
- **The ACM source is unreliable** — Cloudflare's managed challenge usually blocks the
  headless browser. Treat `--source acm` as best-effort only.
- **LLM output is non-deterministic** even at low temperature on cloud models, so the
  refined query and review can vary slightly between identical runs.
