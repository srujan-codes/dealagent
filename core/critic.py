"""Critic agent for DealAgent v3-2.

After the multi-agent committee produces scores, the Critic reviews each
dimension and flags weak evidence:

  - HIGH_SCORE_WEAK_SOURCE     score >=7 but cited tier is 4-5
  - HIGH_SCORE_LOW_TRIANGULATION  score >=7 but triangulation < 0.3
  - SINGLE_SOURCE              the dimension is supported by only one URL
  - SOURCE_TIER_INCONSISTENT   specialist claimed a tier that doesn't match the URL
  - LOW_CONFIDENCE             aggregate confidence below threshold

The flags surface in the UI as warnings on each score card. If at least
one HIGH_SCORE_WEAK_SOURCE or HIGH_SCORE_LOW_TRIANGULATION fires, the
overall confidence is downgraded.

This is a self-review loop without the cost of re-running the entire
committee — it inspects what's already there and surfaces the gaps.
"""
from __future__ import annotations

from typing import Any, Dict, List

from core import credibility

WEAK_TIER_THRESHOLD = 4
HIGH_SCORE_THRESHOLD = 7.0
LOW_TRIANG_THRESHOLD = 0.30
SINGLE_SOURCE_THRESHOLD = 1


def _count_unique_sources_per_dim(research: Dict[str, List[Dict]]) -> Dict[str, int]:
    dim_to_cat = {
        "team": "founder_signals",
        "market": "market_signals",
        "traction": "traction_signals",
        "risk": "risk_signals",
    }
    out = {}
    for dim, cat in dim_to_cat.items():
        urls = set()
        for s in research.get(cat, []):
            u = s.get("url", "")
            if u:
                urls.add(u)
        out[dim] = len(urls)
    return out


def review(scoring: Dict[str, Any], research: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Run the critic over the scoring output. Returns:

    {
      "flags": {dim: [list of flag-strings]},
      "confidence_downgraded": bool,
      "warnings_count": int,
      "summary": str,
    }
    """
    scores = scoring.get("scores", {}) or {}
    counts = _count_unique_sources_per_dim(research)
    flags: Dict[str, List[str]] = {}
    downgrade = False
    notes: List[str] = []

    for dim in ("team", "market", "traction", "risk"):
        s = scores.get(dim, {}) or {}
        raw = float(s.get("raw_score", s.get("score", 0)) or 0)
        tier = int(s.get("source_tier", 4) or 4)
        triang = float(s.get("triangulation", 0) or 0)
        source = s.get("source", "")

        dim_flags: List[str] = []

        # Cited source is weak but score is high
        if raw >= HIGH_SCORE_THRESHOLD and tier >= WEAK_TIER_THRESHOLD:
            dim_flags.append("HIGH_SCORE_WEAK_SOURCE")
            downgrade = True
            notes.append(f"{dim}: score {raw} but cited only T{tier}")

        # High score on poorly-triangulated evidence
        if raw >= HIGH_SCORE_THRESHOLD and triang < LOW_TRIANG_THRESHOLD:
            dim_flags.append("HIGH_SCORE_LOW_TRIANGULATION")
            downgrade = True
            notes.append(f"{dim}: score {raw} with triangulation {triang:.2f}")

        # Only one source in entire dimension
        if counts.get(dim, 0) <= SINGLE_SOURCE_THRESHOLD:
            dim_flags.append("SINGLE_SOURCE")
            notes.append(f"{dim}: only {counts.get(dim, 0)} unique source")

        # Source-tier mismatch: tier in scoring doesn't match the URL's classified tier
        if source:
            real_tier = credibility.classify_source(source)
            if real_tier != tier:
                dim_flags.append(f"SOURCE_TIER_MISMATCH:claimed_T{tier}_actual_T{real_tier}")
                notes.append(f"{dim}: tier mismatch (claimed T{tier}, actual T{real_tier})")

        if dim_flags:
            flags[dim] = dim_flags

    warnings_count = sum(len(v) for v in flags.values())

    if warnings_count == 0:
        summary = "Critic review: no warnings. Scores appear well-supported."
    elif warnings_count <= 2:
        summary = f"Critic review: {warnings_count} minor warning(s)."
    else:
        summary = f"Critic review: {warnings_count} warnings. Consider re-research for flagged dimensions."

    return {
        "flags": flags,
        "confidence_downgraded": downgrade,
        "warnings_count": warnings_count,
        "summary": summary,
        "notes": notes[:8],
    }


def maybe_downgrade_confidence(
    recommendation: Dict[str, Any], critic_review: Dict[str, Any]
) -> Dict[str, Any]:
    """If critic flagged high-score-weak-evidence issues, downgrade the
    recommendation confidence one notch."""
    if not critic_review.get("confidence_downgraded"):
        return recommendation
    rec = dict(recommendation)
    current = (rec.get("confidence") or "MEDIUM").upper()
    if current == "HIGH":
        rec["confidence"] = "MEDIUM"
        rec["_downgraded_by_critic"] = True
    elif current == "MEDIUM":
        rec["confidence"] = "LOW"
        rec["_downgraded_by_critic"] = True
    return rec
