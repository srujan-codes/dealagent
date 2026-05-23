"""Nimble web search client — fires parallel queries via asyncio.gather."""
import asyncio
from typing import List, Dict, Any
import httpx

from core import config


async def _search_one(client: httpx.AsyncClient, query: str) -> List[Dict[str, Any]]:
    """Single Nimble search. Returns list of {title, snippet, url}. Never raises."""
    if not config.have_nimble():
        return _mock_results(query)

    headers = {
        "Authorization": f"Bearer {config.NIMBLE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "num_results": 8,
        "country": "US",
        "locale": "en",
    }

    try:
        resp = await client.post(
            config.NIMBLE_URL,
            json=payload,
            headers=headers,
            timeout=config.HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            return _mock_results(query)
        data = resp.json()
        body = data.get("body") or data.get("results") or []
        out: List[Dict[str, Any]] = []
        for item in body[:8]:
            meta = item.get("metadata", item)
            out.append({
                "title": meta.get("title", "") or meta.get("name", ""),
                "snippet": meta.get("snippet", "") or meta.get("description", ""),
                "url": meta.get("url", "") or meta.get("link", ""),
            })
        return out
    except Exception:
        return _mock_results(query)


def _mock_results(query: str) -> List[Dict[str, Any]]:
    """Deterministic mock so the demo never breaks if Nimble is unreachable."""
    return [
        {
            "title": f"Result about {query[:60]}",
            "snippet": "Public web context unavailable — using fallback signal so the pipeline can continue.",
            "url": "https://example.com/source",
        }
    ]


async def parallel_search(company: str) -> Dict[str, List[Dict[str, Any]]]:
    """Fire 4 Nimble searches in parallel. Returns dict of categorized signals."""
    queries = {
        "founder_signals": f"{company} founder CEO background exit history LinkedIn",
        "traction_signals": f"{company} startup funding revenue growth traction metrics",
        "market_signals": f"{company} competitors market size industry analysis",
        "risk_signals": f"{company} news risk layoffs lawsuit controversy github",
    }
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_search_one(client, q) for q in queries.values()],
            return_exceptions=False,
        )
    return dict(zip(queries.keys(), results))
