"""
graph.py — Knowledge Graph over Claims and Sources (Week 6: GraphRAG)

Builds and queries a lightweight directed graph using networkx.
No Neo4j server required — the graph is serialised to JSON and
loaded fresh each run, so it accumulates over time alongside the
evidence store.

Graph structure:
  Node types:
    claim   — a fact-checked claim (node id = claim_id)
    source  — a news source / outlet (node id = domain)

  Edge types:
    claim  → source  (CITES)         claim referenced this source
    claim  → claim   (RELATED_TO)    two claims share a common source domain
    source → claim   (SUPPORTS)      a source contributed evidence for a claim

Why this matters (Week 6 concept):
  Standard RAG retrieves the top-k most similar chunks for a single query.
  GraphRAG lets the agent "see" relationships across the entire dataset:
    - "Have we checked a claim from this source before?"
    - "Which past claims share the same source domain as this one?"
    - "Is there a known pattern of misleading claims from outlet X?"
  That cross-claim, cross-source context is prepended to the retrieval
  query so the retriever can surface corroborating or contradicting
  evidence from prior runs, not just today's fetches.

Persistence:
  outputs/knowledge_graph.json — serialised node/edge list, updated
  after every daily run. Grows organically; no manual curation needed.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx

from .models import EvidenceChunk

logger = logging.getLogger(__name__)

GRAPH_FILE = "knowledge_graph.json"


# ── Public API ─────────────────────────────────────────────────────────────────

def update_graph(
    chunks: list[EvidenceChunk],
    outputs_dir: str | Path,
) -> None:
    """
    Add new claim/source nodes and edges from today's evidence chunks
    into the persistent knowledge graph.
    """
    path = Path(outputs_dir) / GRAPH_FILE
    G = _load(path)

    for chunk in chunks:
        cid    = chunk.claim_id
        domain = _domain(chunk.source_url)

        # Add claim node (upsert)
        if not G.has_node(cid):
            G.add_node(cid, type="claim", text=chunk.claim_text[:120])

        if domain:
            # Add source node (upsert)
            if not G.has_node(domain):
                G.add_node(domain, type="source")

            # claim → source (CITES)
            G.add_edge(cid, domain, rel="CITES")
            # source → claim (SUPPORTS)
            G.add_edge(domain, cid, rel="SUPPORTS")

    # Add RELATED_TO edges between claims that share a source domain
    _add_related_edges(G)

    _save(G, path)
    logger.info("[graph] Graph updated → %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())


def get_related_context(
    claim_text: str,
    claim_id: str,
    outputs_dir: str | Path,
    max_related: int = 3,
) -> str:
    """
    Query the graph for past claims related to this one via shared sources.
    Returns a short context string to prepend to the retrieval query,
    giving the retriever cross-run awareness (the core GraphRAG value).
    """
    path = Path(outputs_dir) / GRAPH_FILE
    if not path.exists():
        return ""

    G = _load(path)
    if not G.has_node(claim_id):
        # New claim not yet in graph — try keyword match on existing claim texts
        return _fuzzy_context(G, claim_text, max_related)

    # Find claims reachable via shared source domains (depth-2 traversal)
    related: list[str] = []
    for source_node in G.successors(claim_id):
        if G.nodes[source_node].get("type") == "source":
            for related_claim in G.predecessors(source_node):
                if (
                    related_claim != claim_id
                    and G.nodes[related_claim].get("type") == "claim"
                ):
                    related.append(G.nodes[related_claim].get("text", ""))

    # Deduplicate and trim
    seen: set[str] = set()
    unique: list[str] = []
    for t in related:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) >= max_related:
            break

    if not unique:
        return ""

    lines = "\n".join(f"- {t}" for t in unique)
    return f"Related past claims (same sources):\n{lines}"


def graph_stats(outputs_dir: str | Path) -> dict:
    """Return summary stats about the knowledge graph for the eval harness."""
    path = Path(outputs_dir) / GRAPH_FILE
    if not path.exists():
        return {"nodes": 0, "edges": 0, "claims": 0, "sources": 0}
    G = _load(path)
    claims  = [n for n, d in G.nodes(data=True) if d.get("type") == "claim"]
    sources = [n for n, d in G.nodes(data=True) if d.get("type") == "source"]
    return {
        "nodes":   G.number_of_nodes(),
        "edges":   G.number_of_edges(),
        "claims":  len(claims),
        "sources": len(sources),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _add_related_edges(G: nx.DiGraph) -> None:
    """Connect claims that share at least one source domain via RELATED_TO edges."""
    source_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "source"]
    for src in source_nodes:
        linked_claims = [n for n in G.predecessors(src)
                         if G.nodes[n].get("type") == "claim"]
        for i, c1 in enumerate(linked_claims):
            for c2 in linked_claims[i + 1:]:
                if not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, rel="RELATED_TO", via=src)


def _fuzzy_context(G: nx.DiGraph, claim_text: str, max_related: int) -> str:
    """Keyword overlap fallback when the claim is not yet a node."""
    query_words = set(re.findall(r"[a-z]{4,}", claim_text.lower()))
    scored: list[tuple[float, str]] = []
    for node, data in G.nodes(data=True):
        if data.get("type") != "claim":
            continue
        node_words = set(re.findall(r"[a-z]{4,}", data.get("text", "").lower()))
        overlap = len(query_words & node_words) / max(len(query_words), 1)
        if overlap > 0.1:
            scored.append((overlap, data.get("text", "")))

    scored.sort(reverse=True)
    related = [t for _, t in scored[:max_related]]
    if not related:
        return ""
    lines = "\n".join(f"- {t}" for t in related)
    return f"Potentially related past claims:\n{lines}"


def _domain(url: str) -> str:
    """Extract the netloc domain from a URL, stripping 'www.'."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return re.sub(r"^www\.", "", parsed.netloc)
    except Exception:
        return ""


def _load(path: Path) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=False)


def _save(G: nx.DiGraph, path: Path) -> None:
    data = nx.node_link_data(G)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
