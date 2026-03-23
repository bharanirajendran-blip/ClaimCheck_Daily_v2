"""
retriever.py — Hybrid Evidence Retriever 

Combines two retrieval signals over the persistent evidence store:

  Channel 1 — TF-IDF cosine similarity (vector channel)
    Captures semantic overlap, paraphrasing, and related terminology.
    Strong when the query and stored text use different but related words.

  Channel 2 — BM25 term frequency (keyword channel)
    Rewards exact term matches with length normalisation.
    Critical for fact-checking: "claims", "study finds", "according to"
    and named entities need exact-match weight, not just semantic drift.

Hybrid score = 0.60 * vector_score + 0.40 * keyword_score

Both channels are min-max normalised before combining so neither
dominates due to raw scale differences.

Self-correcting retrieval :
  The pipeline calls HybridRetriever.search() twice when the verifier
  triggers a retry — once with the original claim text, once with a
  refined query that appends the verifier's missing-citation hints.
  The retriever itself is stateless; query refinement happens in pipeline.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import EvidenceChunk, RetrievalHit


class HybridRetriever:
    """
    Stateless retriever built from a flat list of EvidenceChunks.
    Instantiated fresh each node call so it always reflects the
    current cumulative store without any caching complexity.
    """

    VECTOR_WEIGHT  = 0.60
    KEYWORD_WEIGHT = 0.40
    KIND_WEIGHTS = {
        "raw_source": 1.15,
        "summary": 0.95,
        "source_metadata": 0.75,
    }

    # BM25 parameters
    _K1 = 1.5   # term saturation
    _B  = 0.75  # length normalisation

    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = chunks
        self._texts = [c.text for c in chunks]

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams for better coverage
            max_features=10000,
        )
        if self._texts:
            self._matrix = self._vectorizer.fit_transform(self._texts)
        else:
            self._matrix = None

        # Pre-compute average document length for BM25
        self._avg_len = (
            sum(len(_tokenize(t)) for t in self._texts) / max(len(self._texts), 1)
        )

    def search(self, query: str, top_k: int = 6) -> list[RetrievalHit]:
        """Return top_k chunks ranked by hybrid score."""
        if not self.chunks or self._matrix is None:
            return []

        # ── Channel 1: TF-IDF cosine ──────────────────────────────────────
        q_vec = self._vectorizer.transform([query])
        raw_vector: np.ndarray = (self._matrix @ q_vec.T).toarray().ravel()

        # ── Channel 2: BM25 ───────────────────────────────────────────────
        query_terms = Counter(_tokenize(query))
        raw_keyword = np.array(
            [self._bm25(query_terms, chunk.text) for chunk in self.chunks]
        )

        # ── Normalise both to [0, 1] then blend ──────────────────────────
        v_norm = _minmax(raw_vector)
        k_norm = _minmax(raw_keyword)
        hybrid = self.VECTOR_WEIGHT * v_norm + self.KEYWORD_WEIGHT * k_norm
        kind_boosts = np.array([
            self.KIND_WEIGHTS.get(chunk.chunk_kind, 1.0) for chunk in self.chunks
        ])
        hybrid = hybrid * kind_boosts

        hits = [
            RetrievalHit(
                chunk=chunk,
                vector_score=float(v_norm[i]),
                keyword_score=float(k_norm[i]),
                hybrid_score=float(hybrid[i]),
            )
            for i, chunk in enumerate(self.chunks)
        ]
        hits.sort(
            key=lambda h: (
                h.hybrid_score,
                1 if h.chunk.chunk_kind == "raw_source" else 0,
                h.vector_score,
            ),
            reverse=True,
        )
        return self._diversify_hits(hits, top_k=top_k)

    # ── BM25 ──────────────────────────────────────────────────────────────

    def _bm25(self, query_terms: Counter[str], doc_text: str) -> float:
        tokens  = Counter(_tokenize(doc_text))
        doc_len = sum(tokens.values())
        score   = 0.0
        for term, _ in query_terms.items():
            tf  = tokens.get(term, 0)
            idf = self._idf(term)
            num = tf * (self._K1 + 1)
            den = tf + self._K1 * (1 - self._B + self._B * doc_len / max(self._avg_len, 1))
            score += idf * (num / max(den, 1e-9))
        return score

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = sum(1 for c in self.chunks if term in _tokenize(c.text))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _diversify_hits(self, hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        """
        Select top-k chunks balancing two diversity axes:
          1. Kind diversity  — bias toward raw_source (up to 4), then fill with others
          2. Domain diversity — prefer chunks from distinct source domains so the
                               verdict rests on multiple independent sources

        Selection pass:
          - First, pick the best raw_source chunk from each unique domain (up to 4).
          - Then fill remaining slots with the next-best chunks, preferring new domains.
          - Finally, pad with any remaining unseen chunks if slots are still open.
        """
        def _domain(url: str) -> str:
            """Extract bare domain (e.g. 'factcheck.org') from a URL string."""
            url = url or ""
            url = re.sub(r"^https?://", "", url).split("/")[0]
            # strip leading www.
            return re.sub(r"^www\.", "", url).lower() or "_unknown_"

        selected: list[RetrievalHit] = []
        seen_ids: set[str] = set()
        seen_domains: set[str] = set()

        raw_hits   = [h for h in hits if h.chunk.chunk_kind == "raw_source"]
        other_hits = [h for h in hits if h.chunk.chunk_kind != "raw_source"]

        # Pass 1: best raw_source chunk per domain, up to 4 slots
        for hit in raw_hits:
            if len(selected) >= min(4, top_k):
                break
            d = _domain(hit.chunk.source_url)
            if hit.chunk.chunk_id in seen_ids:
                continue
            # Prefer new domain; allow repeat domain only if we'd otherwise stall
            if d not in seen_domains or len(seen_domains) >= 3:
                selected.append(hit)
                seen_ids.add(hit.chunk.chunk_id)
                seen_domains.add(d)

        # If pass 1 left raw_source slots unfilled (all same domain), top-up
        for hit in raw_hits:
            if len(selected) >= min(4, top_k):
                break
            if hit.chunk.chunk_id not in seen_ids:
                selected.append(hit)
                seen_ids.add(hit.chunk.chunk_id)
                seen_domains.add(_domain(hit.chunk.source_url))

        # Pass 2: fill remaining slots with non-raw chunks, preferring new domains
        for hit in other_hits:
            if len(selected) >= top_k:
                break
            if hit.chunk.chunk_id in seen_ids:
                continue
            d = _domain(hit.chunk.source_url)
            if d not in seen_domains:
                selected.append(hit)
                seen_ids.add(hit.chunk.chunk_id)
                seen_domains.add(d)

        # Pass 3: pad with any remaining unseen chunks regardless of domain
        for hit in hits:
            if len(selected) >= top_k:
                break
            if hit.chunk.chunk_id not in seen_ids:
                selected.append(hit)
                seen_ids.add(hit.chunk.chunk_id)

        return selected[:top_k]


# ── Utilities ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def build_retrieval_query(claim_text: str, graph_context: str = "") -> str:
    """
    Assemble the retrieval query from the claim + any cross-claim
    context returned by the knowledge graph node.
    """
    base = claim_text.strip()
    if graph_context:
        return f"{base}\n\nRelated prior context:\n{graph_context}"
    return base
