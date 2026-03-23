"""
ClaimCheck Daily — Agent package
Architecture:
  - Director  : GPT-4o orchestrates the daily pipeline (claim selection, verdict synthesis)
  - Researcher : Claude does the deep research + source verification per claim
  - Publisher  : Renders findings to docs/ for GitHub Pages
"""

from .director import Director
from .researcher import Researcher
from .publisher import Publisher
from .pipeline import run_pipeline

__all__ = ["Director", "Researcher", "Publisher", "run_pipeline"]
