"""Source credibility tiering and triangulation logic for DealAgent v2.

The core idea: not all web sources are equal. A claim that only appears on
a startup's own press release is far weaker than the same claim cross-
referenced by the WSJ, an SEC filing, and a court judgment.

We tier every research signal by the credibility of its publisher,
weight each signal accordingly, and require dimension claims to be
triangulated across multiple tiers before counting as "high confidence".

Tier scheme:
  T1 (1.00) — Regulatory / first-party legally-accountable sources
              (sec.gov, courts, uspto, github.com — provable facts)
  T2 (0.70) — Established journalism with editorial standards
              (wsj.com, ft.com, bloomberg.com, nytimes.com, reuters.com, ...)
  T3 (0.50) — Tech / industry press
              (techcrunch.com, theinformation.com, axios.com, ...)
  T4 (0.30) — Analyst / aggregator / wikipedia / linkedin posts
              (linkedin.com, wikipedia.org, similarweb.com, ...)
  T5 (0.10) — Blogs, press releases, social, content farms
              (medium.com, substack.com, facebook.com, twitter.com, ...)

The triangulation score for a dimension is:
   sum(tier_weight for each unique tier represented) / 5.0
A dimension with sources in all 5 tiers scores 1.0. A dimension with only
T5 sources scores 0.02. The truth_score in scoring is computed roughly as:
   truth_score = raw_score * (0.4 + 0.6 * avg_triangulation_score)
so a maximally-triangulated claim survives intact, but a PR-only claim
gets discounted to ~40% of its raw value.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIER_WEIGHTS: Dict[int, float] = {1: 1.00, 2: 0.70, 3: 0.50, 4: 0.30, 5: 0.10}

TIER_LABELS: Dict[int, str] = {
    1: "Regulatory / first-party",
    2: "Established journalism",
    3: "Tech / industry press",
    4: "Analyst / aggregator",
    5: "Blog / social / press release",
}

# Domain → tier map. Order matters: more specific suffixes first.
_TIER_BY_DOMAIN: List[Tuple[str, int]] = [
    # T1 — Regulatory / first-party / legally-accountable
    ("sec.gov",                 1),
    ("data.sec.gov",            1),
    ("courtlistener.com",       1),
    ("supremecourt.gov",        1),
    ("supremecourt.uk",         1),
    ("ecf.uscourts.gov",        1),
    ("uspto.gov",               1),
    ("patentcenter.uspto.gov",  1),
    ("epo.org",                 1),
    ("github.com",              1),
    ("gitlab.com",              1),
    ("fda.gov",                 1),
    ("federalreserve.gov",      1),
    ("ftc.gov",                 1),
    ("doj.gov",                 1),
    ("companieshouse.gov.uk",   1),
    ("eur-lex.europa.eu",       1),

    # T2 — Established journalism with editorial standards
    ("wsj.com",                 2),
    ("ft.com",                  2),
    ("bloomberg.com",           2),
    ("nytimes.com",             2),
    ("reuters.com",             2),
    ("economist.com",           2),
    ("ap.org",                  2),
    ("apnews.com",              2),
    ("bbc.com",                 2),
    ("bbc.co.uk",               2),
    ("cnbc.com",                2),
    ("forbes.com",              2),
    ("washingtonpost.com",      2),
    ("npr.org",                 2),
    ("thehindu.com",            2),
    ("livemint.com",            2),
    ("economictimes.indiatimes.com", 2),

    # T3 — Tech / industry press
    ("techcrunch.com",          3),
    ("theinformation.com",      3),
    ("axios.com",               3),
    ("theverge.com",            3),
    ("wired.com",               3),
    ("arstechnica.com",         3),
    ("venturebeat.com",         3),
    ("crunchbase.com",          3),
    ("pitchbook.com",           3),
    ("sifted.eu",               3),
    ("inc.com",                 3),
    ("businessinsider.com",     3),
    ("fastcompany.com",         3),

    # T4 — Analyst / aggregator / professional networks / Wikipedia
    ("linkedin.com",            4),
    ("wikipedia.org",           4),
    ("similarweb.com",          4),
    ("g2.com",                  4),
    ("glassdoor.com",           4),
    ("teamblind.com",           4),
    ("ycombinator.com",         4),
    ("news.ycombinator.com",    4),
    ("6sense.com",              4),
    ("statista.com",            4),
    ("matrixbcg.com",           4),
    ("pestel-analysis.com",     4),
    ("chargeflow.io",           4),

    # T5 — Blogs / social / press releases / content farms
    ("medium.com",              5),
    ("substack.com",            5),
    ("facebook.com",            5),
    ("instagram.com",           5),
    ("twitter.com",             5),
    ("x.com",                   5),
    ("reddit.com",              5),
    ("youtube.com",             5),
    ("prnewswire.com",          5),
    ("businesswire.com",        5),
    ("globenewswire.com",       5),
]


def _normalize_host(url: str) -> str:
    """Lower-case host, drop leading www., return ''. Never raises."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def classify_source(url: str) -> int:
    """Return the tier (1-5) for a given URL. Unknown → tier 4 (neutral)."""
    if not url:
        return 4
    host = _normalize_host(url)
    if not host:
        return 4

    # Direct match or suffix match
    for domain, tier in _TIER_BY_DOMAIN:
        if host == domain or host.endswith("." + domain):
            return tier

    # Heuristic: a .gov/.gov.* domain is regulatory (T1)
    if host.endswith(".gov") or ".gov." in host:
        return 1

    # Heuristic: a company's own domain (matches the typical pattern
    # "{slug}.com" or "{slug}.io" and was a single-word brand) is T5
    # since first-party marketing material is essentially a press release.
    # We don't know the company here, so default unknowns to T4 (neutral).
    return 4


def annotate_signal(signal: Dict) -> Dict:
    """Return a copy of the signal dict with `tier` and `tier_weight` added."""
    tier = classify_source(signal.get("url", ""))
    out = dict(signal)
    out["tier"] = tier
    out["tier_weight"] = TIER_WEIGHTS[tier]
    out["tier_label"] = TIER_LABELS[tier]
    return out


def annotate_research(research: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Tier-annotate every signal in the research dict."""
    return {
        category: [annotate_signal(s) for s in signals]
        for category, signals in research.items()
    }


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def triangulation_score(signals: Iterable[Dict]) -> float:
    """Score 0.0–1.0 representing how well-triangulated a set of signals is.

    Formula: sum of unique tier weights present, normalized so a set covering
    every tier (1+2+3+4+5 = 0.5+0.5+0.5+0.5+0.5? no — actually 1+0.7+0.5+0.3+0.1
    = 2.6) maps to 1.0. So divisor is 2.6.
    """
    unique_tiers = set()
    for s in signals:
        tier = s.get("tier") or classify_source(s.get("url", ""))
        unique_tiers.add(tier)
    if not unique_tiers:
        return 0.0
    score = sum(TIER_WEIGHTS[t] for t in unique_tiers) / 2.6
    return round(min(1.0, score), 3)


def triangulation_per_dimension(research: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Compute triangulation score for each of the 4 research dimensions."""
    return {cat: triangulation_score(sigs) for cat, sigs in research.items()}


# ---------------------------------------------------------------------------
# Truth-score discount
# ---------------------------------------------------------------------------

def truth_discount(triangulation: float) -> float:
    """Map triangulation [0,1] to a multiplier [0.4, 1.0].

    A claim with zero triangulation (T5-only sources) keeps 40% of its raw
    score. A claim triangulated across all 5 tiers keeps 100%. Linear in between.
    """
    return round(0.4 + 0.6 * max(0.0, min(1.0, triangulation)), 3)


def discount_score(raw_score: float, triangulation: float) -> float:
    """Apply the truth discount to a raw score."""
    return round(raw_score * truth_discount(triangulation), 2)


# ---------------------------------------------------------------------------
# Convenience: tier distribution summary
# ---------------------------------------------------------------------------

def tier_distribution(research: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Count how many signals fall in each tier. Useful for UI badges."""
    counts = {f"T{t}": 0 for t in (1, 2, 3, 4, 5)}
    for signals in research.values():
        for s in signals:
            t = s.get("tier") or classify_source(s.get("url", ""))
            counts[f"T{t}"] += 1
    return counts
