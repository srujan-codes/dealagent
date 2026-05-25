"""Semantic memory for DealAgent v3-5.

After enough reports accumulate in ClickHouse, every new analysis can be
informed by similar past deals: 'this company looks structurally like
{X}, which we scored {Y}'. This is the foundation of an agent that
gets sharper over time.

For embeddings we'd ideally use sentence-transformers, but to keep the
dependency surface small we use TF-IDF-style Jaccard similarity on word
tokens from the verdict + key insight + company name. Good enough for
small corpora (<10k reports) and zero extra deps.

Public API:
  find_similar_past_reports(company, verdict, key_insight, k=3)
    Returns top-k {company_name, similarity, overall_score, verdict, created_at}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core import clickhouse_client


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "have", "has",
    "but", "not", "are", "was", "were", "will", "can", "more", "than",
    "their", "its", "is", "of", "to", "in", "on", "at", "by", "an", "as",
    "be", "or", "a", "i", "we", "they", "he", "she", "it", "all", "any",
    "may", "due", "very", "much", "one", "company", "companies",
})


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")
            if w.lower() not in _STOP and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _make_signature(company: str, verdict: str, key_insight: str = "") -> set:
    """Build the token signature for similarity comparison."""
    return _tokens(company + " " + verdict + " " + key_insight)


def find_similar_past_reports(
    company: str,
    verdict: str,
    key_insight: str = "",
    k: int = 3,
    exclude_same_company: bool = True,
) -> List[Dict[str, Any]]:
    """Return top-k most similar past reports from ClickHouse."""
    client = clickhouse_client._get_client()
    if client is None:
        return []

    try:
        rows = client.query(
            "SELECT company_name, overall_score, verdict, created_at "
            "FROM dd_reports ORDER BY created_at DESC LIMIT 200"
        ).result_rows
    except Exception:
        return []

    target_sig = _make_signature(company, verdict, key_insight)
    if not target_sig:
        return []

    company_lower = (company or "").lower().strip()
    scored: List[Dict[str, Any]] = []
    for row in rows:
        past_company, past_score, past_verdict, past_ts = row
        if exclude_same_company and (past_company or "").lower().strip() == company_lower:
            continue
        past_sig = _make_signature(past_company, past_verdict, "")
        sim = _jaccard(target_sig, past_sig)
        if sim <= 0:
            continue
        scored.append({
            "company_name": past_company,
            "overall_score": round(float(past_score), 2),
            "verdict": past_verdict,
            "created_at": past_ts.isoformat() if hasattr(past_ts, "isoformat") else str(past_ts),
            "similarity": round(sim, 3),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:k]


def memory_context_for_synthesizer(
    similar: List[Dict[str, Any]]
) -> str:
    """Format the top-k similar past reports for inclusion in a synthesizer prompt."""
    if not similar:
        return ""
    lines = ["SIMILAR PAST DEALS WE'VE ANALYZED (for context only — do not over-anchor):"]
    for r in similar:
        lines.append(
            f"  - {r['company_name']} (score {r['overall_score']}/10, "
            f"similarity {r['similarity']}): {r['verdict'][:120]}"
        )
    return "\n".join(lines)
