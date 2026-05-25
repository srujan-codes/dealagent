"""GitHub agent — pulls T1 engineering signal for tech companies.

Why this matters: marketing copy is easy to fake. A company's GitHub activity
is much harder to fake without actually shipping code. Commit cadence,
contributor count, recent activity — these are first-party, auditable facts.

We try several lookup strategies because the company's GitHub org name
isn't always exactly their company name (e.g. "Anthropic" → "anthropics",
"Stripe" → "stripe", "Notion" → "makenotion"). Strategy:

  1. Look up the org by the slugified company name
  2. If that 404s, fall back to /search/repositories?q={company} and grab
     the most-starred result's owner

Output is shaped to plug straight into the existing research dict as a
new 'engineering_signals' category with tier=1 annotations.

Uses unauthenticated GitHub REST API (60 req/hr/IP) by default. If
GITHUB_TOKEN is in the env, the rate climbs to 5000/hr.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from core import config

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def _headers() -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "DealAgent-v2",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "", name.lower())


async def _get(client: httpx.AsyncClient, path: str, **params) -> Optional[Any]:
    try:
        r = await client.get(GITHUB_API + path, params=params, headers=_headers())
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Org / repo discovery
# ---------------------------------------------------------------------------

async def discover_org(client: httpx.AsyncClient, company: str) -> Optional[Dict[str, Any]]:
    """Find the most likely GitHub org for a company. Returns org dict or None."""
    candidates = [_slug(company), _slug(company) + "s", _slug(company) + "ai",
                  _slug(company) + "labs", _slug(company) + "inc",
                  "make" + _slug(company), _slug(company) + "hq"]

    # Try direct org lookup first
    for cand in candidates:
        if not cand:
            continue
        org = await _get(client, f"/orgs/{cand}")
        if org:
            return org

    # Fall back to repo search to find the most-active GitHub presence
    search = await _get(
        client,
        "/search/repositories",
        q=f"{company} in:name,description",
        sort="stars",
        order="desc",
        per_page=5,
    )
    if search and search.get("items"):
        # Take the owner of the most-starred match
        top = search["items"][0]
        owner = (top.get("owner") or {}).get("login")
        if owner:
            org = await _get(client, f"/orgs/{owner}")
            if org:
                return org
            # If it's a user account, fall back to a minimal dict
            user = await _get(client, f"/users/{owner}")
            if user:
                return user

    return None


async def list_org_repos(
    client: httpx.AsyncClient, login: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """List top repos for an org/user, sorted by recent push."""
    # Try as org first, then as user
    repos = await _get(client, f"/orgs/{login}/repos", sort="updated", per_page=limit)
    if not repos:
        repos = await _get(client, f"/users/{login}/repos", sort="updated", per_page=limit)
    return repos or []


# ---------------------------------------------------------------------------
# Activity metrics
# ---------------------------------------------------------------------------

async def repo_activity(
    client: httpx.AsyncClient, full_name: str
) -> Dict[str, Any]:
    """Last-90-days activity summary for one repo."""
    # GitHub's stats endpoints are notoriously cold — they 202 first then 200 on retry.
    # We just count commits in the last 90 days via /commits with since=
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 90 * 86400))
    commits = await _get(client, f"/repos/{full_name}/commits", since=cutoff, per_page=100)
    commit_count = len(commits) if isinstance(commits, list) else 0

    contributors = await _get(client, f"/repos/{full_name}/contributors", per_page=100)
    contributor_count = len(contributors) if isinstance(contributors, list) else 0

    return {
        "commits_90d": commit_count,
        "contributors_total": contributor_count,
    }


# ---------------------------------------------------------------------------
# Public API: build engineering signals for a company
# ---------------------------------------------------------------------------

async def engineering_signals(company: str) -> List[Dict[str, Any]]:
    """Return a list of T1 engineering signals shaped like Nimble's research items."""
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        org = await discover_org(client, company)
        if not org:
            return out

        login = org.get("login", "")
        org_url = org.get("html_url", f"https://github.com/{login}")
        public_repos = org.get("public_repos") or org.get("total_repos") or 0
        followers = org.get("followers", 0)

        out.append({
            "title": f"{org.get('name') or login} on GitHub",
            "snippet": (
                f"{public_repos} public repos · {followers} followers"
                + (f" · bio: {org.get('description','')[:120]}" if org.get("description") else "")
            ),
            "url": org_url,
            "tier": 1,
            "tier_weight": 1.0,
            "tier_label": "Regulatory / first-party",
            "source_type": "github_org",
        })

        # Sample top repos for activity
        repos = await list_org_repos(client, login, limit=5)
        for repo in repos[:3]:
            full = repo.get("full_name", "")
            if not full:
                continue
            activity = await repo_activity(client, full)
            stars = repo.get("stargazers_count", 0)
            updated = (repo.get("pushed_at") or "")[:10]
            out.append({
                "title": f"GitHub: {full}",
                "snippet": (
                    f"{stars}★ · {activity['commits_90d']} commits in last 90d · "
                    f"{activity['contributors_total']} contributors · last push {updated}"
                ),
                "url": repo.get("html_url", f"https://github.com/{full}"),
                "tier": 1,
                "tier_weight": 1.0,
                "tier_label": "Regulatory / first-party",
                "source_type": "github_repo",
                "metrics": {
                    "stars": stars,
                    "commits_90d": activity["commits_90d"],
                    "contributors": activity["contributors_total"],
                    "last_push": updated,
                },
            })
    return out
