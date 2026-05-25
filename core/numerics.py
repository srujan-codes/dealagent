"""Numerical claim extraction + cross-source contradiction detection.

PR firms can shade narratives, but they can't easily make every source
agree on a hard number. If Bloomberg says $50M ARR and TechCrunch says
$80M ARR, that's a 60% discrepancy — and the agent should flag it.

This module:
  1. Scans every snippet for numerical claims (revenue, funding, headcount,
     growth, valuation, market share).
  2. Normalizes them to a common scale ($M for money, integer for counts).
  3. Groups by metric type and source URL.
  4. Detects contradictions where multiple sources cite different values
     for the same metric.
  5. Returns a structured "key_numbers" dict for UI display.

Returns a list of normalized claims + a list of detected contradictions.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Money normalization
# ---------------------------------------------------------------------------
# Match: $50M, $50 million, $1.2B, $1.2 billion, USD 500M, $500,000, 50M USD
_MONEY_RE = re.compile(
    r"""
    (?:US?\$|USD\s*)?\$?\s*
    (\d{1,3}(?:[,.]\d{3})*(?:\.\d+)?)   # number
    \s*
    (?:(M|B|K|million|billion|thousand))?  # scale
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _to_millions(num_str: str, scale: str) -> float:
    """Normalize money expression to millions of USD."""
    try:
        n = float(num_str.replace(",", ""))
    except ValueError:
        return 0.0
    scale = (scale or "").lower()
    if scale in ("b", "billion"):
        return n * 1000.0
    if scale in ("k", "thousand"):
        return n / 1000.0
    if scale in ("m", "million"):
        return n
    # No scale: if number is huge (>10000), assume raw USD
    if n > 10_000_000:
        return n / 1_000_000.0
    if n > 10_000:
        # Probably "$500,000" meant 0.5M
        return n / 1_000_000.0
    # Otherwise assume already in millions implicitly
    return n


# ---------------------------------------------------------------------------
# Metric-extraction patterns
# ---------------------------------------------------------------------------
# Each pattern captures a money-bearing phrase. We then re-run MONEY_RE inside
# the captured span to normalize. Metric type comes from the trigger keyword.

_METRIC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("revenue", re.compile(
        r"((?:annual\s+revenue|revenue|ARR|topline|sales)[^.]{0,80}?(?:US?\$|USD|\$)\s*[\d.,]+\s*(?:M|B|K|million|billion|thousand)?)",
        re.IGNORECASE)),
    ("funding", re.compile(
        r"((?:raised|raising|funding|round|investment|secured)[^.]{0,80}?(?:US?\$|USD|\$)\s*[\d.,]+\s*(?:M|B|K|million|billion|thousand)?)",
        re.IGNORECASE)),
    ("valuation", re.compile(
        r"((?:valuation|valued\s+at|worth)[^.]{0,60}?(?:US?\$|USD|\$)\s*[\d.,]+\s*(?:M|B|K|million|billion|thousand)?)",
        re.IGNORECASE)),
]

_HEADCOUNT_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d{2,})\s+(?:employees|staff|engineers|developers|workers|people|team\s+members)",
    re.IGNORECASE,
)

_GROWTH_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:YoY|year[- ]over[- ]year|growth|increase|grew|gain)",
    re.IGNORECASE,
)

_MARKET_SHARE_RE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:market\s+share|of\s+the\s+market)",
    re.IGNORECASE,
)


def _extract_money_from_phrase(phrase: str) -> float:
    """Pull the first money expression from a phrase, return in $M."""
    m = _MONEY_RE.search(phrase)
    if not m:
        return 0.0
    return _to_millions(m.group(1), m.group(2) or "")


def extract_from_snippet(snippet: str, source_url: str, source_tier: int) -> List[Dict[str, Any]]:
    """Return all numerical claims found in a single snippet."""
    out: List[Dict[str, Any]] = []
    if not snippet:
        return out

    for metric, pattern in _METRIC_PATTERNS:
        for match in pattern.finditer(snippet):
            phrase = match.group(1)
            amount = _extract_money_from_phrase(phrase)
            if amount > 0:
                out.append({
                    "metric": metric,
                    "value_millions_usd": round(amount, 2),
                    "raw_phrase": phrase.strip()[:200],
                    "source_url": source_url,
                    "source_tier": source_tier,
                })

    # Headcount
    for m in _HEADCOUNT_RE.finditer(snippet):
        try:
            n = int(m.group(1).replace(",", ""))
            if 5 <= n <= 5_000_000:
                out.append({
                    "metric": "headcount",
                    "value": n,
                    "raw_phrase": m.group(0).strip(),
                    "source_url": source_url,
                    "source_tier": source_tier,
                })
        except ValueError:
            pass

    # Growth
    for m in _GROWTH_RE.finditer(snippet):
        try:
            pct = float(m.group(1))
            if 0.1 <= pct <= 1000:
                out.append({
                    "metric": "growth_pct",
                    "value": pct,
                    "raw_phrase": m.group(0).strip(),
                    "source_url": source_url,
                    "source_tier": source_tier,
                })
        except ValueError:
            pass

    # Market share
    for m in _MARKET_SHARE_RE.finditer(snippet):
        try:
            pct = float(m.group(1))
            if 0.1 <= pct <= 100:
                out.append({
                    "metric": "market_share_pct",
                    "value": pct,
                    "raw_phrase": m.group(0).strip(),
                    "source_url": source_url,
                    "source_tier": source_tier,
                })
        except ValueError:
            pass

    return out


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def detect_contradictions(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each metric type, if multiple sources disagree by >20%, flag it."""
    grouped: Dict[str, List[Dict]] = {}
    for c in claims:
        grouped.setdefault(c["metric"], []).append(c)

    contradictions: List[Dict[str, Any]] = []
    for metric, items in grouped.items():
        # Get the numeric value per claim
        values = []
        for it in items:
            v = it.get("value_millions_usd") if "value_millions_usd" in it else it.get("value")
            if isinstance(v, (int, float)) and v > 0:
                values.append((v, it))
        if len(values) < 2:
            continue
        v_nums = [v for v, _ in values]
        lo, hi = min(v_nums), max(v_nums)
        if lo <= 0:
            continue
        pct_diff = (hi - lo) / lo * 100.0
        if pct_diff < 20:
            continue  # Within tolerance — not a contradiction
        contradictions.append({
            "metric": metric,
            "low_value": lo,
            "high_value": hi,
            "percent_difference": round(pct_diff, 1),
            "low_source": next(it for v, it in values if v == lo).get("source_url"),
            "high_source": next(it for v, it in values if v == hi).get("source_url"),
            "sources_count": len(values),
        })
    return contradictions


def summarize_claims(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a UI-friendly summary: for each metric, the values + sources."""
    grouped: Dict[str, List[Dict]] = {}
    for c in claims:
        grouped.setdefault(c["metric"], []).append(c)

    summary: Dict[str, Any] = {}
    for metric, items in grouped.items():
        # Sort by source tier (T1 first) and take top 4 distinct values
        items = sorted(items, key=lambda x: x.get("source_tier", 9))
        seen_values = set()
        unique = []
        for it in items:
            v = it.get("value_millions_usd") if "value_millions_usd" in it else it.get("value")
            key = round(v, 1) if isinstance(v, float) else v
            if key in seen_values:
                continue
            seen_values.add(key)
            unique.append(it)
            if len(unique) >= 4:
                break
        summary[metric] = unique
    return summary


def extract_all_numerics(
    research: Dict[str, List[Dict]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Top-level: scan every snippet, return (claims, contradictions, summary)."""
    claims: List[Dict[str, Any]] = []
    for sigs in research.values():
        for s in sigs:
            url = s.get("url", "")
            tier = int(s.get("tier", 4) or 4)
            snippet = s.get("snippet", "")
            claims.extend(extract_from_snippet(snippet, url, tier))
    contradictions = detect_contradictions(claims)
    summary = summarize_claims(claims)
    return claims, contradictions, summary
