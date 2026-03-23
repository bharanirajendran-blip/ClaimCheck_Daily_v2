# ClaimCheck Daily

An automated AI fact-checking pipeline that runs daily, researches claims from real news feeds using live web content, and publishes verdicts to a GitHub Pages website.

**Live site:** https://bharanirajendran-blip.github.io/ClaimCheck_Daily/

---

## What It Does

Every day the pipeline:
1. Pulls headlines from 7 RSS feeds (AP, Reuters, PolitiFact, FactCheck.org, WHO, Science Daily, MIT Tech Review)
2. GPT-4o (Director) selects the 3 most fact-checkable claims — from different sources and topics
3. Claude (Researcher) fetches live source articles and investigates each claim using extended thinking and a ReAct tool-use loop
4. Research findings are chunked and stored in a persistent evidence store
5. A knowledge graph links claims to their sources across runs, giving the retriever cross-run context
6. A hybrid retriever (TF-IDF + BM25) surfaces the most relevant evidence chunks for each claim
7. GPT-4o synthesises a verdict grounded in the retrieved evidence
8. A separate GPT-4o call acts as an independent judge, scoring the verdict on groundedness, citation quality, consistency, and assumption level
9. If scores fall below threshold the retrieval query is refined and the verdict is regenerated (up to 2 retries)
10. Results are published as a rendered HTML page and a JSON archive; quality scores appear on each verdict card

---

## Architecture

```
feeds.yaml
    │
    ▼
harvest_node ──► select_node ──► research_node ──► store_evidence_node
                  (GPT-4o)        (Claude +              │
                                  extended thinking +     │  chunks to
                                  fetch_url tool)         ▼  disk + graph
                                               graph_context_node
                                                    │
                                                    ▼
                                               retrieve_node
                                              (TF-IDF + BM25)
                                                    │
                                                    ▼
                                               verdict_node
                                                (GPT-4o +
                                               RAG chunks)
                                                    │
                                                    ▼
                                               verify_node
                                           (LLM-as-a-Judge)
                                                    │
                              ┌─────── retry? ──────┘
                              │   (revise_query_node
                              │    → retrieve → verdict
                              │    → verify, max 2×)
                              │
                              └─────── passed ──► publish_node ──► END
```

Built with **LangGraph** (StateGraph), **Pydantic v2** (validated state at every step), **scikit-learn** (TF-IDF), and **networkx** (knowledge graph).

### Components

| Component | Role | Model |
|---|---|---|
| Director | Claim selection, RAG-grounded verdict synthesis | GPT-4o |
| Researcher | Live web research with ReAct tool-use loop | Claude + extended thinking |
| fetch_url tool | Fetches and strips article HTML to plain text | httpx + BeautifulSoup |
| Evidence Store | Chunks research findings by markdown section; persists across runs | — |
| Knowledge Graph | Links claims → sources across runs; injects cross-run context | networkx |
| HybridRetriever | TF-IDF cosine (60%) + BM25 keyword (40%); min-max normalised | scikit-learn |
| Verifier | Independent LLM-as-a-Judge scoring on 4-dimension RAGAS rubric | GPT-4o |
| Publisher | HTML + JSON output with quality score bars | — |

### How the Researcher tool-use loop works

Claude doesn't rely on training knowledge alone. When researching a claim it:
1. Calls `fetch_url` on the source article URL
2. Reads the returned plain text
3. May fetch one more corroborating source
4. Produces a structured analysis with sub-questions, evidence, caveats, and key sources

This ReAct loop runs for up to 5 rounds until Claude returns a final response.

### Hybrid retrieval

After research is stored, the pipeline retrieves the top 6 most relevant evidence chunks per claim using a two-channel hybrid search:

- **Vector channel (60%)** — TF-IDF cosine similarity with bigram tokenisation
- **Keyword channel (40%)** — BM25 term-frequency scoring (k₁ = 1.5, b = 0.75)

Both scores are min-max normalised before combining. The retrieval query is enriched with context from the knowledge graph (related past claims that share source domains), so the retriever surfaces corroborating or contradicting evidence from prior runs.

### LLM-as-a-Judge verification

After each verdict is generated, a separate GPT-4o call evaluates it against the retrieved evidence on four dimensions:

| Dimension | Weight | Meaning |
|---|---|---|
| Groundedness | 35% | Every claim in the verdict is supported by a retrieved chunk |
| Citation score | 35% | Key sources are properly referenced |
| No contradiction | 15% | Verdict does not contradict the evidence |
| No assumption | 15% | Verdict makes no unsupported leaps |

If groundedness or citation falls below 0.70, the verdict is flagged for retry. The verifier identifies missing citation hints, which are used to refine the retrieval query before regenerating the verdict.

---

## Verdict Labels

| Label | Meaning |
|---|---|
| ✅ TRUE | Claim is accurate |
| 🟢 MOSTLY TRUE | Accurate with minor caveats |
| 🟡 MIXED | Partially true, partially false |
| 🟠 MOSTLY FALSE | Misleading or largely inaccurate |
| ❌ FALSE | Claim is inaccurate |
| ❔ UNVERIFIABLE | Insufficient evidence to judge |

---

## Project Structure

```
ClaimCheck_Daily/
├── agent/
│   ├── models.py       Pydantic data models + LangGraph PipelineState
│   ├── director.py     GPT-4o Director agent
│   ├── researcher.py   Claude Researcher agent (ReAct tool-use loop)
│   ├── tools.py        fetch_url tool — live web article fetcher
│   ├── feeds.py        RSS/Atom feed parser
│   ├── store.py        Persistent evidence store (chunk + persist)
│   ├── retriever.py    HybridRetriever (TF-IDF + BM25)
│   ├── graph.py        Knowledge graph (claims ↔ sources, networkx)
│   ├── verifier.py     LLM-as-a-Judge (RAGAS rubric)
│   ├── pipeline.py     LangGraph StateGraph orchestration
│   ├── publisher.py    HTML + JSON renderer
│   └── utils.py        retry, logging, env helpers
├── docs/               GitHub Pages output (auto-generated)
├── outputs/            JSON archive + evidence store (auto-generated)
│   ├── YYYY-MM-DD.json          Daily verdicts
│   ├── evidence_YYYY-MM-DD.json Daily evidence chunks
│   ├── evidence_store.json      Cumulative deduplicated chunk store
│   └── knowledge_graph.json     Persistent claim-source graph
├── feeds.yaml          RSS feed configuration
├── run.py              CLI entry point
└── SPEC.md             Full technical specification
```

---

## Setup

### Requirements

You need two API keys:

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com → API Keys |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |

### Install & Run

```bash
# 1. Clone the repo
git clone https://github.com/bharanirajendran-blip/ClaimCheck_Daily.git
cd ClaimCheck_Daily

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 4. Dry run — feed harvest + GPT claim selection only (no Claude research, no cost)
python run.py --dry-run

# 5. Full pipeline run (~2–3 minutes, ~$0.60–$0.80)
python run.py
```

### Optional flags

```bash
python run.py --log-level DEBUG          # verbose logging
python run.py --dry-run                  # harvest + select only, skip research
python run.py --workers 2                # parallel research threads (default: 3)
python run.py --feeds my_feeds.yaml      # custom feed file
python run.py --outputs-dir my_outputs   # custom output directory
```

### Output files

| File | Description |
|---|---|
| `docs/YYYY-MM-DD.html` | Dark-themed report page; open in any browser |
| `docs/index.html` | Landing page with links to all past reports |
| `outputs/YYYY-MM-DD.json` | Machine-readable verdicts |
| `outputs/evidence_store.json` | Cumulative chunked evidence (grows across runs) |
| `outputs/knowledge_graph.json` | Persistent claim-source graph (grows across runs) |

```bash
open docs/$(date +%Y-%m-%d).html      # macOS — open today's report
```

---

## GitHub Actions Automation (Currently Disabled)

The repo includes `.github/workflows/daily.yml` which runs the pipeline automatically at 08:00 UTC and publishes to GitHub Pages.

**It is currently disabled.** To re-enable safely:
1. Make the repo **private** (Settings → General → Change visibility)
2. Add secrets under Settings → Secrets and variables → Actions:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
3. Go to **Actions tab → ClaimCheck Daily → Enable workflow**

---

## Troubleshooting

**`ModuleNotFoundError: sklearn` or `networkx`**
Run `pip install -r requirements.txt` — scikit-learn and networkx are required for hybrid retrieval and the knowledge graph.

**`max_tokens must be greater than thinking.budget_tokens`**
Make sure you have the latest `researcher.py` — `max_tokens=10000`, `thinking_budget=3000`.

**`Repository not found` when pushing**
Create the repo on github.com first (empty, no README), then push.

**`src refspec main does not match any`**
No commits yet — run `git add . && git commit -m "initial"` first.
Or rename the branch: `git branch -m master main`.

**Paywalled articles return login page content**
`fetch_url` can only read public pages. Claude notes this and falls back to training knowledge for that source.

**`ModuleNotFoundError: bs4`**
Run `pip install beautifulsoup4`.

---

## Tech Stack

- [Anthropic Claude](https://www.anthropic.com) — deep claim research with extended thinking + ReAct tool-use loop
- [OpenAI GPT-4o](https://openai.com) — claim selection, RAG-grounded verdict synthesis, LLM-as-a-Judge evaluation
- [LangGraph](https://langchain-ai.github.io/langgraph/) — stateful multi-agent pipeline with conditional retry loop
- [Pydantic v2](https://docs.pydantic.dev) — data validation and state modelling
- [scikit-learn](https://scikit-learn.org) — TF-IDF vectoriser for hybrid retrieval
- [networkx](https://networkx.org) — knowledge graph over claims and sources
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — HTML stripping for web fetch tool
- [GitHub Pages](https://pages.github.com) — automated publishing

---

## Course

Grad5900 — Agentic AI
