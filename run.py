#!/usr/bin/env python3
"""
ClaimCheck Daily — CLI entry point
------------------------------------
Usage:
    python run.py                          # run with defaults
    python run.py --feeds my_feeds.yaml    # custom feed file
    python run.py --log-level DEBUG        # verbose output
    python run.py --dry-run                # harvest + select, skip research
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing agents (so API keys are available)
load_dotenv()

from agent.pipeline import run_pipeline  # noqa: E402  (after dotenv)
from agent.models import Claim      # noqa: E402
from agent.utils import require_env      # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="claimcheck",
        description="ClaimCheck Daily — automated fact-checking pipeline",
    )
    p.add_argument(
        "--feeds",
        default="feeds.yaml",
        help="Path to feeds.yaml (default: feeds.yaml)",
    )
    p.add_argument(
        "--docs-dir",
        default="docs",
        help="Output directory for GitHub Pages HTML (default: docs)",
    )
    p.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Output directory for JSON results (default: outputs)",
    )
    p.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("RESEARCH_WORKERS", "3")),
        help="Parallel research threads (default: 3)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Harvest + Director selection only; skip Claude research & publishing",
    )
    p.add_argument(
        "--claim",
        help="Investigate a single claim text instead of using daily feeds",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        # Lightweight check — just show what would be selected
        from agent.feeds import harvest_claims
        from agent.director import Director
        from agent.utils import setup_logging

        setup_logging(args.log_level)
        if args.claim:
            claim = Claim(
                id=hashlib.md5(args.claim.encode()).hexdigest()[:8],
                text=args.claim,
                source="Manual Input",
                feed_name="Manual Input",
            )
            print("\nDry-run: manual claim selected:\n")
            print(f"  [{claim.id}] {claim.text[:200]}")
            return 0
        try:
            require_env("OPENAI_API_KEY")
        except EnvironmentError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1
        candidates = harvest_claims(args.feeds)
        director = Director()
        selected = director.select_claims(candidates)
        print(f"\nDry-run: {len(selected)} claims selected:\n")
        for c in selected:
            print(f"  [{c.id}] {c.text[:100]}")
        return 0

    # Validate required env vars before full pipeline work
    try:
        require_env("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    except EnvironmentError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    report = run_pipeline(
        feeds_path=args.feeds,
        docs_dir=args.docs_dir,
        outputs_dir=args.outputs_dir,
        max_workers=args.workers,
        log_level=args.log_level,
        manual_claim=args.claim,
    )

    print(
        f"\n✅  Published {len(report.verdicts)} verdicts → "
        f"{args.docs_dir}/{report.date_slug}.html"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
