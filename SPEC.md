# ClaimCheck Daily — Project Specification

**Course:** Grad5900 — Agentic AI
**Author:** Bharani Rajendran
**Repo:** https://github.com/bharanirajendran-blip/ClaimCheck_Daily
**Live site:** https://bharanirajendran-blip.github.io/ClaimCheck_Daily/

---

## 1. Project Overview

ClaimCheck Daily is an automated fact-checking pipeline that runs every day, pulls claims from real news and fact-checking RSS feeds, researches each one using AI with live web access, retrieves grounded evidence, produces verified verdicts, and publishes the results as a GitHub Pages website.

The system uses a multi-agent architecture extended with retrieval-augmented generation (RAG), a persistent knowledge graph, and a LLM-as-a-Judge evaluation layer. GPT-4o acts as a high-level Director and independent Verifier; Claude acts as a deep Researcher that fetches and reads live articles. LangGraph wires all components into a stateful, validated pipeline with a self-correcting retry loop. Pydantic enforces data integrity at every step.

---

## 2. Architecture

### 2.1 Agent and Component Roles

**Director (GPT-4o)**
- Reads all harvested claim candidates and selects the top 3 most impactful and verifiable ones for the day
- Enforces source diversity: no two claims from the same outlet, covering at least 2 different topics
- Prefers disputed claims from PolitiFact/FactCheck.org, avoids press releases and paywalled sources
- After retrieval is complete, synthesises a structured verdict grounded in the top-ranked evidence chunks
- Uses OpenAI JSON mode to guarantee parseable structured output

**Researcher (Claude)**
- Receives one claim at a time from the pipeline
- Runs a ReAct tool-use loop: calls `fetch_url` to read live article content, then reasons over it
- Uses extended thinking (3,000 budget tokens, 10,000 max tokens) for deep reasoning over the claim
- Fetches the source URL first, then at most one corroborating source — writes findings immediately after
- Produces a structured research report covering sub-questions, supporting evidence, contradicting evidence, caveats, and key sources

**fetch_url tool**
- Defined in `agent/tools.py` using the Anthropic tool-use API schema
- Fetches a URL using `httpx`, strips HTML with BeautifulSoup, returns clean plain text
- Truncates to 4,000 characters to stay within context window limits
- Claude decides autonomously when and what to fetch — not hardcoded
- Max 5 tool-use rounds per claim to prevent infinite loops

**Evidence Store (`agent/store.py`)**
- Chunks each `ResearchResult` by markdown section heading (splits on `##`, caps at 700 chars)
- Also stores each cited source URL as a discrete chunk for graph edge construction
- Persists to `outputs/evidence_YYYY-MM-DD.json` (daily) and `outputs/evidence_store.json` (cumulative, deduplicated by `chunk_id`)
- Chunks accumulate across runs so retrieval improves over time

**Knowledge Graph (`agent/graph.py`)**
- Maintains a directed graph over `claim` nodes and `source` (domain) nodes using networkx
- Edge types: `CITES` (claim → source), `SUPPORTS` (source → claim), `RELATED_TO` (claim → claim via shared source)
- Updated after every run; serialised to `outputs/knowledge_graph.json`
- `get_related_context()` does a depth-2 traversal to find past claims sharing the same source domains, returning a context string that is prepended to the retrieval query
- This is the GraphRAG pattern: cross-claim, cross-source awareness that flat vector search cannot provide

**HybridRetriever (`agent/retriever.py`)**
- Two-channel retrieval over all cumulative evidence chunks:
  - **Vector channel (60%)** — TF-IDF cosine similarity (sklearn, bigram tokenisation)
  - **Keyword channel (40%)** — BM25 term-frequency scoring (k₁ = 1.5, b = 0.75)
- Both scores min-max normalised before combining
- Retrieval query is enriched with graph context before search
- On retry, the query is replaced by the verifier's refined query derived from `missing_citations`

**Verifier (`agent/verifier.py`)**
- A separate GPT-4o call that acts as an independent judge, evaluating the Director's verdict against the retrieved evidence
- Scores on four dimensions (RAGAS-style rubric):

  | Dimension | Weight | Threshold for retry |
  |---|---|---|
  | Groundedness | 35% | < 0.70 |
  | Citation score | 35% | < 0.70 |
  | No contradiction | 15% | — |
  | No assumption | 15% | — |

- Returns a `VerifierReport` with scores, `unsupported_statements`, `contradictions`, `missing_citations`, and `should_retry`
- Falls back to a heuristic mode if `OPENAI_API_KEY` is not set

**Publisher (`agent/publisher.py`)**
- Renders an HTML verdict card for each claim, including verdict badge, confidence, summary, key evidence, and a collapsible quality score panel with colour-coded progress bars for each verifier dimension
- Generates `docs/YYYY-MM-DD.html`, `docs/index.html`, and `outputs/YYYY-MM-DD.json`

---

### 2.2 Pipeline Flow (LangGraph StateGraph)

```
START
  │
  ▼
harvest_node          ← parse RSS/Atom feeds → candidate Claims
  │
  ▼ (conditional: abort if no candidates)
select_node           ← Director (GPT-4o) picks top 3 claims (diverse sources/topics)
  │
  ▼ (conditional: abort if nothing selected)
research_node         ← Researcher (Claude + fetch_url) investigates each claim
  │
  ▼
store_evidence_node   ← chunk ResearchResults → evidence_store.json (persistent)
  │
  ▼
graph_context_node    ← update knowledge_graph.json; query for related past claims
  │
  ▼
retrieve_node  ◄──────────────────────────────────────────────┐
  │                                                           │
  ▼                                                           │
verdict_node          ← Director synthesises verdict          │
                         grounded in top-k chunks             │
  │                                                           │
  ▼                                                           │
verify_node           ← LLM-as-a-Judge evaluates verdict      │
  │                                                           │
  ├─── retry? ──► revise_query_node ── refined query ─────────┘
  │               (max 2 retries)
  │
  └─── passed ──► publish_node ← HTML + JSON output
                      │
                      ▼
                     END
```

The self-correcting loop (verify → revise_query → retrieve → verdict → verify) exits when all verifier scores pass threshold or when the retry counter for any claim reaches 2.

### 2.3 State Management

All state is carried in a single `PipelineState` Pydantic model flowing through every LangGraph node. Each node receives the full state, updates only its own fields, and returns a partial dict that LangGraph merges back. The full pipeline state is inspectable and validated at every transition.

---

## 3. Data Models (Pydantic)

All models are in `agent/models.py` and use `pydantic.BaseModel`.

| Model | Purpose | Key Fields |
|---|---|---|
| `Claim` | One harvested claim | `id`, `text`, `source`, `url`, `published_at` |
| `ResearchResult` | Claude's research output | `claim_id`, `findings`, `sources` |
| `EvidenceChunk` | A single section chunk from research | `chunk_id`, `claim_id`, `claim_text`, `source_url`, `section`, `text`, `date_slug` |
| `RetrievalHit` | A ranked chunk from the retriever | `chunk`, `vector_score`, `keyword_score`, `hybrid_score` |
| `VerifierReport` | Judge's evaluation of one verdict | `groundedness_score`, `citation_score`, `no_contradiction_score`, `no_assumption_score`, `overall_score`, `unsupported_statements`, `contradictions`, `missing_citations`, `rewrite_suggestion`, `should_retry` |
| `Verdict` | GPT's final judgement | `claim_id`, `verdict` (enum), `confidence` (0–1), `summary`, `key_evidence` |
| `DailyReport` | Full day's output | `claims`, `verdicts`, `retrieval_hits`, `verifier_reports`, `date_slug`, `generated_at` |
| `PipelineState` | LangGraph shared state | all of the above + `evidence_chunks`, `graph_context`, `retry_counts`, config fields |

`VerdictLabel` is a `str, Enum` with six values: `TRUE`, `MOSTLY_TRUE`, `MIXED`, `MOSTLY_FALSE`, `FALSE`, `UNVERIFIABLE`. Pydantic validates GPT's output against this enum and ensures `confidence` is between 0.0 and 1.0.

---

## 4. Repository Structure

```
ClaimCheck_Daily/
├── agent/
│   ├── __init__.py        package exports
│   ├── models.py          Pydantic data models + PipelineState
│   ├── director.py        GPT-4o Director agent
│   ├── researcher.py      Claude Researcher agent (ReAct tool-use loop)
│   ├── tools.py           fetch_url tool — live web article fetcher
│   ├── feeds.py           RSS/Atom feed parser
│   ├── store.py           Persistent evidence store (chunk + persist)
│   ├── retriever.py       HybridRetriever (TF-IDF + BM25)
│   ├── graph.py           Knowledge graph (claims ↔ sources, networkx)
│   ├── verifier.py        LLM-as-a-Judge evaluator (RAGAS rubric)
│   ├── pipeline.py        LangGraph StateGraph orchestration
│   ├── publisher.py       HTML + JSON output renderer
│   └── utils.py           retry decorator, logging, env helpers
│
├── docs/                  GitHub Pages output (auto-generated)
│   ├── _config.yml        Jekyll config
│   ├── index.html         landing page (regenerated each run)
│   └── YYYY-MM-DD.html    one page per daily report
│
├── outputs/               JSON archive (auto-generated; persists across runs)
│   ├── YYYY-MM-DD.json              daily machine-readable verdicts
│   ├── evidence_YYYY-MM-DD.json     daily evidence chunks
│   ├── evidence_store.json          cumulative deduplicated chunk store
│   └── knowledge_graph.json         persistent claim-source graph
│
├── .github/workflows/
│   └── daily.yml          GitHub Actions cron job (08:00 UTC daily)
│
├── feeds.yaml             RSS feed configuration
├── .env.example           environment variable template
├── requirements.txt       Python dependencies
├── run.py                 CLI entry point
├── SPEC.md                this file
└── .gitignore
```

---

## 5. Feed Sources

Configured in `feeds.yaml`. Seven sources across four categories:

| Source | Category |
|---|---|
| AP News — Top Headlines | news |
| Reuters — Top News | news |
| PolitiFact — Latest | politics |
| FactCheck.org | politics |
| Science Daily — Top Science | science |
| WHO — News | health |
| MIT Technology Review | technology |

Each run harvests up to 10 entries per feed (70 candidates max), which the Director narrows to 3.

---

## 6. Output Format

### HTML Report (`docs/YYYY-MM-DD.html`)

A dark-themed GitHub Pages page with one card per claim. Each card shows:
- Verdict badge (colour-coded)
- Claim text and source link
- Confidence percentage
- Summary
- Collapsible key evidence list
- Collapsible quality score panel with colour-coded progress bars for groundedness, citation, no contradiction, no assumption, and overall score (from the LLM-as-a-Judge)

### JSON Archive (`outputs/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-22",
  "generated_at": "2026-03-22T17:30:30Z",
  "results": [
    {
      "claim": "...",
      "source": "...",
      "verdict": "MIXED",
      "confidence": 0.85,
      "summary": "...",
      "key_evidence": ["...", "..."]
    }
  ]
}
```

### Evidence Store (`outputs/evidence_store.json`)

Cumulative JSON array of all `EvidenceChunk` objects from every run, deduplicated by `chunk_id`. Used by the HybridRetriever on subsequent runs — gets richer over time.

### Knowledge Graph (`outputs/knowledge_graph.json`)

networkx node-link JSON format. Contains claim nodes, source-domain nodes, and directional edges (CITES, SUPPORTS, RELATED_TO). Updated after every run.

---

## 7. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | ≥ 0.40.0 | Claude Researcher API (extended thinking + tool use) |
| `openai` | ≥ 1.50.0 | GPT-4o Director and Verifier APIs (JSON mode) |
| `langgraph` | ≥ 0.2.0 | StateGraph pipeline orchestration |
| `langchain-core` | ≥ 0.3.0 | Required by LangGraph |
| `pydantic` | ≥ 2.7.0 | Data validation and state modelling |
| `scikit-learn` | ≥ 1.5.0 | TF-IDF vectoriser (vector channel of HybridRetriever) |
| `numpy` | ≥ 1.26.0 | Array ops for cosine similarity and min-max normalisation |
| `networkx` | ≥ 3.3 | Knowledge graph (GraphRAG) |
| `feedparser` | ≥ 6.0.11 | RSS/Atom feed ingestion |
| `httpx` | ≥ 0.27.0 | HTTP client for fetch_url tool |
| `beautifulsoup4` | ≥ 4.12.0 | HTML stripping for fetch_url tool |
| `python-dotenv` | ≥ 1.0.0 | `.env` loading |
| `pyyaml` | ≥ 6.0.2 | `feeds.yaml` parsing |
| `requests` | ≥ 2.32.0 | HTTP fallback |
| `python-dateutil` | ≥ 2.9.0 | Robust date parsing for feed entries |

---

## 8. Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily.git
cd ClaimCheck_Daily

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 4. Dry run — feed harvest + GPT selection only (no Claude research, no cost)
python run.py --dry-run

# 5. Full pipeline run (~2–3 minutes, ~$0.60–$0.80)
python run.py

# 6. Optional flags
python run.py --feeds my_feeds.yaml      # custom feed file
python run.py --log-level DEBUG          # verbose logging
python run.py --workers 2               # parallel research threads (default: 3)
python run.py --outputs-dir my_outputs  # custom output directory

# 7. View today's report (macOS)
open docs/$(date +%Y-%m-%d).html
```

**Note:** The evidence store and knowledge graph in `outputs/` accumulate across runs. The longer the pipeline has been running, the richer the retrieved evidence and graph context will be.

---

## 9. Known Limitations

**Paywalled articles:** `fetch_url` can only read publicly accessible pages. Paywalled content returns a login page instead of article text. Claude handles this gracefully by falling back to training knowledge for that source.

**Cold-start retrieval:** On the very first run there are no stored chunks yet, so the retrieval step is skipped and the verdict is generated from the researcher's findings alone. Evidence accumulates from the second run onwards.

**Claim extraction:** The feed parser uses the article headline as the claim text. Headlines are not always precise factual claims. A future improvement is to extract specific sub-claims from article bodies.

**No cross-day deduplication:** The same claim can appear across multiple feeds on multiple days. A content-hash dedup layer on ingested candidates would prevent redundant research.

**GitHub Actions workflow (currently disabled):** The `.github/workflows/daily.yml` cron job is disabled. To re-enable, make the repo private, add `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` as GitHub Secrets under Settings → Secrets and variables → Actions, then re-enable the workflow from the Actions tab.

---

## 10. Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-03-02 | Initial skeleton — LangGraph pipeline, Pydantic models, Claude Researcher, GPT Director, GitHub Pages publishing |
| v1.1 | 2026-03-02 | Added `fetch_url` tool-use loop to Researcher — Claude now reads live articles instead of relying solely on training knowledge |
| v1.2 | 2026-03-07 | Tuned for reliability — reduced claims to 3, sequential research, 3k thinking budget, 4k content cap, Director diversity rules, Researcher stop-early prompt |
| v2.0 | 2026-03-22 | Added persistent evidence store, hybrid retrieval (TF-IDF + BM25), GraphRAG knowledge graph, LLM-as-a-Judge verifier with RAGAS rubric, self-correcting retry loop, quality score bars in HTML output |
