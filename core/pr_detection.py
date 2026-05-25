"""PR-shine detection for DealAgent v2.

When a startup hires a PR firm, the firm typically writes a single
talking-point doc and seeds it across many outlets. The result: 5-10
"independent" articles that all use the same phrases. This module
detects that pattern using pairwise n-gram overlap.

We compute the average Jaccard similarity on word bigrams across
every pair of research snippets. High similarity means the snippets
are saying the same thing in the same words → likely PR coordination.
We combine that with the tier distribution (T4-T5 heavy = PR-prone)
to produce a single PR-shine score in [0.0, 1.0].

The score modulates the truth_discount in the scoring agent: a high
PR-shine pulls the truth_score even further toward neutral than the
tier-only discount would.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

# Words that appear in almost every business snippet — exclude from
# similarity matching so legitimate-looking content doesn't trigger.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "but",
    "not", "are", "was", "were", "will", "can", "more", "than", "their", "its",
    "is", "of", "to", "in", "on", "at", "by", "an", "as", "be", "or", "a",
    "company", "companies", "business", "billion", "million", "first", "new",
    "i", "you", "we", "they", "he", "she", "it",
})


def _tokens(text: str) -> List[str]:
    """Lowercase, drop punctuation, drop stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _bigrams(text: str) -> set:
    toks = _tokens(text)
    return set(zip(toks, toks[1:]))


def _trigrams(text: str) -> set:
    toks = _tokens(text)
    return set(zip(toks, toks[1:], toks[2:]))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _all_snippets(research: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """Flatten the research dict into a list of {snippet, url, tier} entries."""
    out: List[Dict[str, Any]] = []
    for sigs in research.values():
        for s in sigs:
            snip = s.get("snippet", "")
            if snip and len(snip) > 20:
                out.append({
                    "snippet": snip,
                    "url": s.get("url", ""),
                    "tier": s.get("tier", 4),
                })
    return out


def detect_pr_shine(research: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Return a dict with pr_shine_score in [0, 1] plus explanation fields.

    High score = strong evidence of coordinated PR. Low score = sources
    look organic and independent.
    """
    snippets = _all_snippets(research)
    n = len(snippets)
    if n < 3:
        return {
            "pr_shine_score": 0.0,
            "level": "insufficient_data",
            "snippets_analyzed": n,
            "avg_pairwise_bigram_similarity": 0.0,
            "max_pairwise_bigram_similarity": 0.0,
            "tier_penalty": 0.0,
            "method": "jaccard bigram + tier distribution",
            "notes": "fewer than 3 substantial snippets — cannot assess",
        }

    # Pairwise bigram similarity
    bigrams = [_bigrams(s["snippet"]) for s in snippets]
    similarities: List[float] = []
    max_sim = 0.0
    max_pair = ("", "")
    for i in range(n):
        for j in range(i + 1, n):
            # Skip pairs where both snippets are from the same URL (duplicates)
            if snippets[i]["url"] and snippets[i]["url"] == snippets[j]["url"]:
                continue
            sim = _jaccard(bigrams[i], bigrams[j])
            similarities.append(sim)
            if sim > max_sim:
                max_sim = sim
                max_pair = (snippets[i]["url"], snippets[j]["url"])

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Empirical scaling: avg jaccard of 0.10+ is very suspicious, 0.05+ suspicious.
    # Cap contribution from similarity at 0.7 of the total score.
    similarity_component = min(0.7, avg_sim * 10)

    # Tier penalty: how much of the corpus is in T4-T5 (PR-prone tiers)
    tier_counts = {t: 0 for t in (1, 2, 3, 4, 5)}
    for s in snippets:
        tier_counts[int(s.get("tier") or 4)] += 1
    pr_tier_fraction = (tier_counts[4] + tier_counts[5]) / n
    # T4-T5 heavy contributes up to 0.3 of the score
    tier_penalty = min(0.3, pr_tier_fraction * 0.4)

    pr_shine = round(min(1.0, similarity_component + tier_penalty), 3)

    if pr_shine >= 0.5:
        level = "high"
    elif pr_shine >= 0.25:
        level = "medium"
    else:
        level = "low"

    return {
        "pr_shine_score": pr_shine,
        "level": level,
        "snippets_analyzed": n,
        "avg_pairwise_bigram_similarity": round(avg_sim, 4),
        "max_pairwise_bigram_similarity": round(max_sim, 4),
        "most_similar_pair": list(max_pair) if max_sim > 0.05 else [],
        "tier_penalty": round(tier_penalty, 3),
        "tier_counts": {f"T{k}": v for k, v in tier_counts.items()},
        "method": "jaccard bigram + tier distribution",
    }


def apply_pr_shine_to_truth_discount(base_discount: float, pr_shine: float) -> float:
    """Combine the tier-based discount with PR-shine to get a final discount.

    base_discount is already in [0.4, 1.0] from triangulation. We further
    pull it down when PR shine is high. Final discount = base * (1 - pr_shine * 0.4).
    A maximally-suspicious PR shine (1.0) shaves up to 40% off the base discount.
    """
    return round(max(0.2, base_discount * (1.0 - pr_shine * 0.4)), 3)
