"""Temporal trajectory analysis for DealAgent v3-4.

For each scoring dimension, this module computes:
  - latest_date:  most recent date mentioned in supporting snippets
  - median_date:  median date across all dated signals
  - recency:      "fresh" (<90d), "recent" (<365d), "aging" (<3y), "stale" (>3y)
  - momentum:     "up", "flat", "down", or "unknown"
    derived from comparing recent-half vs older-half signal counts +
    GitHub repo activity (if present)

Returns a per-dimension trajectory dict that the UI can show as small
↑ / → / ↓ indicators next to each score.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from core.pr_detection import _extract_year_month


_TODAY = dt.date.today()


def _ym_to_date(ym: Optional[tuple]) -> Optional[dt.date]:
    if ym is None:
        return None
    y, m = ym
    try:
        return dt.date(y, m, 15)
    except Exception:
        return None


def _classify_recency(d: Optional[dt.date]) -> str:
    if d is None:
        return "unknown"
    age_days = (_TODAY - d).days
    if age_days < 0:
        age_days = 0
    if age_days <= 90:
        return "fresh"
    if age_days <= 365:
        return "recent"
    if age_days <= 365 * 3:
        return "aging"
    return "stale"


def _momentum_from_github(signals: List[Dict[str, Any]]) -> Optional[str]:
    """If GitHub repo signals are present, derive momentum from commits_90d
    relative to repo age."""
    repo_signals = [s for s in signals if s.get("source_type") == "github_repo"]
    if not repo_signals:
        return None
    total_commits_90d = 0
    repos_with_recent_pushes = 0
    for r in repo_signals:
        m = r.get("metrics") or {}
        total_commits_90d += int(m.get("commits_90d", 0) or 0)
        push = m.get("last_push", "")
        try:
            push_date = dt.datetime.strptime(push[:10], "%Y-%m-%d").date()
            if (_TODAY - push_date).days <= 90:
                repos_with_recent_pushes += 1
        except Exception:
            pass
    if total_commits_90d >= 30 and repos_with_recent_pushes >= 1:
        return "up"
    if total_commits_90d > 0:
        return "flat"
    return "down"


def _momentum_from_dates(dates: List[dt.date]) -> str:
    """Compare distribution of dates: are most recent vs older?"""
    if len(dates) < 4:
        return "unknown"
    dates = sorted(dates)
    midpoint = dates[len(dates) // 2]
    older = sum(1 for d in dates if d <= midpoint)
    newer = sum(1 for d in dates if d > midpoint)
    if newer > older + 2:
        return "up"
    if older > newer + 2:
        return "down"
    return "flat"


def analyze(research: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Per-dimension temporal trajectory.

    Returns {
      'team': {recency, momentum, latest_date, dated_count, ...},
      'market': {...},
      'traction': {...},
      'risk': {...},
      'overall_freshness': float [0-1],  # fraction of fresh+recent across all
    }
    """
    dim_to_cat = {
        "team":     "founder_signals",
        "market":   "market_signals",
        "traction": "traction_signals",
        "risk":     "risk_signals",
    }
    out: Dict[str, Any] = {}
    fresh_count = 0
    total_dated = 0

    for dim, cat in dim_to_cat.items():
        sigs = research.get(cat, []) or []
        dates: List[dt.date] = []
        for s in sigs:
            d = _ym_to_date(_extract_year_month(s.get("snippet", "")))
            if d:
                dates.append(d)
        latest = max(dates) if dates else None
        median = sorted(dates)[len(dates)//2] if dates else None

        # Momentum: prefer GitHub if any github_repo signals; else date distribution
        gh_mom = _momentum_from_github(sigs)
        momentum = gh_mom if gh_mom else _momentum_from_dates(dates)

        recency = _classify_recency(latest)
        if recency in ("fresh", "recent"):
            fresh_count += 1
        if dates:
            total_dated += 1

        out[dim] = {
            "recency": recency,
            "momentum": momentum,
            "latest_date": latest.isoformat() if latest else None,
            "median_date": median.isoformat() if median else None,
            "dated_count": len(dates),
            "total_signals": len(sigs),
            "github_signal_present": any(s.get("source_type") in ("github_repo", "github_org") for s in sigs),
        }

    out["overall_freshness"] = round(fresh_count / 4, 2)
    return out
