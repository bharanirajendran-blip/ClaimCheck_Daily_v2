"""
Feed parser — ingests RSS/Atom feeds defined in feeds.yaml
and extracts candidate Claim objects.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator

import feedparser
import yaml

from .models import Claim

logger = logging.getLogger(__name__)

# Heuristics: headline patterns that often contain checkable claims
CLAIM_SIGNALS = re.compile(
    r"\b(says|claims|reports|announces|confirms|denies|alleges|warns|"
    r"according to|study finds|data shows|experts say)\b",
    re.IGNORECASE,
)


def load_feeds(feeds_path: str | Path = "feeds.yaml") -> list[dict]:
    with open(feeds_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("feeds", [])


def harvest_claims(
    feeds_path: str | Path = "feeds.yaml",
    max_per_feed: int = 10,
) -> list[Claim]:
    """Parse all feeds and return candidate claims."""
    feeds = load_feeds(feeds_path)
    claims: list[Claim] = []

    for feed_cfg in feeds:
        name = feed_cfg.get("name", feed_cfg["url"])
        url = feed_cfg["url"]
        logger.info("Parsing feed: %s", name)

        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:max_per_feed]:
                for claim in _extract_claims(entry, feed_name=name):
                    claims.append(claim)
        except Exception as exc:
            logger.warning("Failed to parse feed %s: %s", name, exc)

    logger.info("Harvested %d candidate claims from %d feeds.", len(claims), len(feeds))
    return claims


def _extract_claims(entry, feed_name: str) -> Iterator[Claim]:
    """Yield 0-or-more Claim objects from a single feed entry."""
    title: str = getattr(entry, "title", "").strip()
    link: str = getattr(entry, "link", "")
    published: str = getattr(entry, "published", None)
    summary: str = getattr(entry, "summary", "").strip()

    # Use title as the primary claim text if it looks checkable
    text = title or summary
    if not text:
        return

    # Stable ID based on content hash
    claim_id = hashlib.md5(text.encode()).hexdigest()[:8]

    yield Claim(
        id=claim_id,
        text=text,
        source=feed_name,
        url=link,
        published_at=published,
        feed_name=feed_name,
    )
