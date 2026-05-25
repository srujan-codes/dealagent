"""Claim grounding / provenance for DealAgent v3-3.

For each score, find the actual research signal whose URL was cited and
attach its full snippet + title + tier as 'evidence'. This lets the UI
render an expandable "source trail" under each score card — so judges
can see the exact text the agent used to make each claim.

Also computes a 'lexical overlap' score between the score's reasoning
sentence and the snippet, so we can highlight which fraction of the
reasoning is directly grounded in the source vs synthesized.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STOP = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "but", "not", "are", "was", "were", "will", "can", "more", "than",
    "their", "its", "is", "of", "to", "in", "on", "at", "by", "an", "as",
    "be", "or", "a", "i", "we", "they", "he", "she", "it", "all", "any",
    "may", "due", "very", "much", "one", "two",
})


def _words(s: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "") if w.lower() not in _STOP and len(w) > 2]


def _overlap_ratio(reasoning: str, snippet: str) -> float:
    """Fraction of reasoning words also present in the snippet."""
    r = _words(reasoning)
    s = set(_words(snippet))
    if not r:
        return 0.0
    hits = sum(1 for w in r if w in s)
    return round(hits / len(r), 3)


def _find_signal_by_url(research: Dict[str, List[Dict]], url: str) -> Optional[Dict[str, Any]]:
    if not url:
        return None
    for sigs in research.values():
        for s in sigs:
            if s.get("url") == url:
                return s
    return None


def ground_scores(
    scoring: Dict[str, Any], research: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """Attach evidence to each score in-place. Returns the mutated dict + a
    top-level provenance summary."""
    scores = scoring.get("scores", {}) or {}
    summary = {"grounded_count": 0, "ungrounded_count": 0, "avg_overlap": 0.0}
    overlaps: List[float] = []

    for dim in ("team", "market", "traction", "risk"):
        s = scores.get(dim, {}) or {}
        url = s.get("source", "")
        reasoning = s.get("reasoning", "")
        sig = _find_signal_by_url(research, url)
        if sig:
            overlap = _overlap_ratio(reasoning, sig.get("snippet", ""))
            s["evidence"] = {
                "url": url,
                "snippet": sig.get("snippet", ""),
                "title": sig.get("title", ""),
                "tier": int(sig.get("tier", s.get("source_tier", 4)) or 4),
                "source_type": sig.get("source_type", "web"),
                "overlap_ratio": overlap,
                "grounded": overlap >= 0.15,
            }
            overlaps.append(overlap)
            if overlap >= 0.15:
                summary["grounded_count"] += 1
            else:
                summary["ungrounded_count"] += 1
        else:
            s["evidence"] = {
                "url": url,
                "snippet": "",
                "title": "",
                "tier": int(s.get("source_tier", 4) or 4),
                "source_type": "missing",
                "overlap_ratio": 0.0,
                "grounded": False,
                "note": "Source URL not found in research corpus",
            }
            summary["ungrounded_count"] += 1

        # Also ground risk sub-scores if present
        breakdown = (s.get("breakdown") or {}) if dim == "risk" else {}
        for sub_key, sub in list(breakdown.items()):
            if not isinstance(sub, dict):
                # Synthesizer may have returned a bare number; coerce
                breakdown[sub_key] = {"score": sub if isinstance(sub, (int, float)) else 5, "reasoning": ""}
                sub = breakdown[sub_key]
            sub_reasoning = sub.get("reasoning", "")
            best_sig = _best_signal_for_reasoning(research, sub_reasoning)
            if best_sig:
                ov = _overlap_ratio(sub_reasoning, best_sig.get("snippet", ""))
                sub["evidence"] = {
                    "url": best_sig.get("url", ""),
                    "title": best_sig.get("title", ""),
                    "snippet": best_sig.get("snippet", ""),
                    "tier": int(best_sig.get("tier", 4) or 4),
                    "overlap_ratio": ov,
                    "grounded": ov >= 0.15,
                }
        scores[dim] = s

    if overlaps:
        summary["avg_overlap"] = round(sum(overlaps) / len(overlaps), 3)
    scoring["scores"] = scores
    scoring["provenance_summary"] = summary
    return scoring


def _best_signal_for_reasoning(
    research: Dict[str, List[Dict]], reasoning: str
) -> Optional[Dict[str, Any]]:
    """Find the research signal whose snippet has the most lexical overlap with
    the reasoning sentence. Used for risk-sub-score grounding where the
    specialist didn't return a per-sub source."""
    target_words = set(_words(reasoning))
    if not target_words:
        return None
    best = None
    best_score = 0
    for sigs in research.values():
        for s in sigs:
            snippet_words = set(_words(s.get("snippet", "")))
            overlap = len(target_words & snippet_words)
            if overlap > best_score:
                best_score = overlap
                best = s
    if best_score < 2:
        return None
    return best
