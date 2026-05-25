"""CourtListener agent — T1 legal-risk signal.

CourtListener (https://www.courtlistener.com) is the Free Law Project's
public archive of US federal and state court filings. Their REST API
exposes dockets, opinions, and parties — all source-of-truth data.

Why this matters for VC due diligence: PR firms cannot hide lawsuits.
If a company has been sued for fraud, IP infringement, employment
violations, or breach of contract, it shows up here. This is exactly
the kind of risk signal the open web tends to bury.

CourtListener offers anonymous read access at low rate limits (no token
needed for basic search). With a free COURTLISTENER_TOKEN you get
higher quotas.

We hit /api/rest/v4/search/?type=r&q={company} which searches RECAP
dockets (federal trial-court filings). Returns dockets ordered by
date filed descending.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx

CL_BASE = "https://www.courtlistener.com"
CL_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json", "User-Agent": "DealAgent-v2"}
    if CL_TOKEN:
        h["Authorization"] = f"Token {CL_TOKEN}"
    return h


async def legal_signals(company: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch the most recent lawsuits/dockets mentioning the company."""
    out: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(
                f"{CL_BASE}/api/rest/v4/search/",
                params={"type": "r", "q": company, "order_by": "dateFiled desc"},
                headers=_headers(),
            )
            if r.status_code != 200:
                return out
            data = r.json()
    except Exception:
        return out

    results = data.get("results", [])[:limit]
    for d in results:
        case_name = d.get("caseName") or d.get("case_name") or "Unnamed docket"
        court = d.get("court") or d.get("court_citation_string") or ""
        date_filed = d.get("dateFiled") or d.get("date_filed") or ""
        docket_num = d.get("docketNumber") or d.get("docket_number") or ""
        abs_url = d.get("absolute_url") or ""
        url = f"{CL_BASE}{abs_url}" if abs_url else f"{CL_BASE}/?q={company}"

        out.append({
            "title": f"Court: {case_name[:80]}",
            "snippet": (
                f"{court} · docket {docket_num} · filed {date_filed[:10] if date_filed else 'unknown'}"
            ),
            "url": url,
            "tier": 1,
            "tier_weight": 1.0,
            "tier_label": "Regulatory / first-party",
            "source_type": "courtlistener_docket",
            "court": court,
            "date_filed": date_filed,
        })
    return out
