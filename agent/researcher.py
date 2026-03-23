"""
Researcher — Claude-powered deep research agent with web fetch tool
--------------------------------------------------------------------
Responsibilities:
  1. Accept a single Claim from the Director
  2. Use Claude (claude-opus-4-5) with:
       - Extended thinking (10k budget tokens) for deep reasoning
       - fetch_url tool so Claude can read live article content
  3. Run an agentic tool-use loop:
       - Claude reasons, decides to fetch a URL
       - We fetch it, return the text as a tool_result
       - Claude continues reasoning with real content
       - Loop ends when Claude returns a final text response
  4. Return a ResearchResult (findings + extracted sources)

The tool-use loop means Claude is no longer limited to training knowledge —
it can read the actual source articles published in 2025/2026.
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass, field

import anthropic

from .models import Claim, ResearchResult
from .tools import TOOL_DEFINITIONS, execute_tool
from .utils import retry

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a meticulous research analyst for ClaimCheck Daily.
    You have access to a fetch_url tool to read live web articles.

    Your task is to evaluate a claim by fetching and reading its source article.

    IMPORTANT RULES:
    - Fetch the source URL provided first. That is your primary evidence.
    - You may fetch ONE additional corroborating source if needed.
    - After fetching the source article, STOP fetching and write your final analysis.
    - Do NOT chase dead links or keep fetching if URLs return errors.
    - Base your verdict on what you successfully read. If the source article loaded,
      you have enough to write a thorough analysis — do so immediately.

    Structure your final response as follows:

    ## Sub-questions
    [numbered list]

    ## Evidence Assessment
    [detailed prose, balanced, based on what you read]

    ## Supporting evidence
    [bullet points with sources]

    ## Contradicting evidence
    [bullet points with sources]

    ## Caveats & Missing Context
    [prose]

    ## Key Sources
    [list of {"title": "...", "url": "...", "reliability": "high|medium|low"}]
""")

MAX_TOOL_ROUNDS = 5  # enough rounds to fetch + synthesize findings


def _block_to_dict(block) -> dict | None:
    """
    Convert an Anthropic SDK content block to a plain dict for the messages API.
    Avoids the Pydantic v2 'by_alias' serialization conflict when passing
    SDK objects directly back into the messages list.
    """
    # Prefer Anthropic SDK serialization so block shape stays API-compatible
    # (important for thinking blocks + signatures).
    try:
        dumped = block.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass

    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": block.text}
    if t == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if t == "thinking":
        signature = getattr(block, "signature", None)
        thinking = getattr(block, "thinking", None)
        if signature and thinking:
            return {"type": "thinking", "thinking": thinking, "signature": signature}
        logger.warning("Skipping malformed thinking block while replaying assistant turn.")
        return None
    if t:
        return {"type": t}
    return None


@dataclass
class Researcher:
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"))
    max_tokens: int = 10000        # must be greater than thinking_budget
    use_extended_thinking: bool = True
    thinking_budget: int = 3000    # enough to process fetched article content
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, claim: Claim) -> ResearchResult:
        """Deep-research a single claim using tool-use loop, return findings."""
        logger.info("Researcher investigating claim %s: %s", claim.id, claim.text[:80])

        raw_text = self._run_tool_loop(claim)
        sources = self._extract_sources(raw_text)

        return ResearchResult(
            claim_id=claim.id,
            findings=raw_text,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Agentic tool-use loop
    # ------------------------------------------------------------------

    @retry(times=3, delay=3)
    def _run_tool_loop(self, claim: Claim) -> str:  # noqa: C901
        """
        Agentic loop:
          1. Send claim + tools to Claude
          2. If Claude calls fetch_url → execute it → append tool_result → repeat
          3. When Claude returns a final text response → extract and return it
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Claim to investigate:\n\n\"{claim.text}\"\n\n"
                    f"Original source: {claim.source}\n"
                    f"Source URL: {claim.url or 'not available'}\n"
                    f"Published: {claim.published_at or 'unknown'}\n\n"
                    "Please fetch the source article first, then research this claim thoroughly."
                ),
            }
        ]

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            kwargs: dict = dict(
                model=self.model,
                max_tokens=self.max_tokens,
                system=RESEARCH_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            if self.use_extended_thinking:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
                kwargs["temperature"] = 1  # required for extended thinking

            response = self._client.messages.create(**kwargs)
            logger.debug("Round %d stop_reason=%s", round_num, response.stop_reason)

            # ── Claude finished — collect all text blocks ─────────────
            if response.stop_reason == "end_turn":
                parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(parts).strip()

            # ── Claude wants to use a tool ────────────────────────────
            if response.stop_reason == "tool_use":
                # Serialize SDK content blocks to plain dicts to avoid
                # Pydantic v2 / Anthropic SDK 'by_alias' serialization conflict
                assistant_content = [
                    block_dict
                    for block_dict in (_block_to_dict(b) for b in response.content)
                    if block_dict is not None
                ]
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # Execute every tool call Claude requested
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(
                            "[tool] Claude calling %s with %s", block.name, block.input
                        )
                        result_text = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                # Return all tool results in one user turn
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason — break and return whatever text we have
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            parts = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            return "\n".join(parts).strip()

        logger.warning("Tool loop hit max rounds (%d) for claim %s", MAX_TOOL_ROUNDS, claim.id)
        return "Research incomplete: maximum tool rounds reached."

    # ------------------------------------------------------------------
    # Source extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(raw: str) -> list[dict]:
        """Best-effort extraction of JSON source objects from the Key Sources section."""
        import json
        import re

        sources = []
        pattern = r'\{[^{}]*"title"[^{}]*\}'
        for match in re.finditer(pattern, raw, re.DOTALL):
            try:
                sources.append(json.loads(match.group()))
            except json.JSONDecodeError:
                pass
        return sources
