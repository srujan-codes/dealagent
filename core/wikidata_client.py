"""Wikidata + Wikipedia structured-facts client.

Wikidata is the structured-data backbone of Wikipedia. Every notable
company has an item with properties like founding date, headquarters,
CEO, employee count, total funding. These facts are heavily-watched
by editors — PR firms can't easily slip falsehoods in unnoticed.

We resolve a company name → Wikidata item via the search API, then
fetch its claims via the entity API. Returns a dict of structured
facts shaped for direct display in the UI.

All endpoints are free, no auth, polite User-Agent only.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx

UA = "DealAgent srujan-codes/dealagent v2 (research@dealagent.ai)"

# Wikidata properties we care about. P-numbers from wikidata.org.
PROPS = {
    "P571":  "inception",           # founding date
    "P159":  "headquarters",        # HQ location
    "P169":  "ceo",                 # chief executive officer
    "P112":  "founder",             # founder(s)
    "P1128": "employees",           # number of employees
    "P2226": "market_cap",          # market capitalization
    "P2403": "assets_total",        # total assets
    "P452":  "industry",            # industry
    "P17":   "country",             # country
    "P856":  "official_website",    # official site
    "P414":  "stock_exchange",      # stock exchange
    "P249":  "ticker_symbol",       # ticker symbol
    "P2218": "net_worth",           # net worth (rare on company items)
}


def _headers() -> Dict[str, str]:
    return {"User-Agent": UA, "Accept": "application/json"}


async def _search_entity(client: httpx.AsyncClient, name: str) -> Optional[str]:
    """Return the Wikidata Q-id for the best matching entity (a company), or None."""
    try:
        r = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": "en",
                "format": "json",
                "type": "item",
                "limit": 8,
            },
            headers=_headers(),
        )
        if r.status_code != 200:
            return None
        results = r.json().get("search", [])
        if not results:
            return None
        # Prefer results whose description mentions company-like words
        company_keywords = ("company", "corporation", "enterprise", "firm",
                            "business", "startup", "organization", "platform",
                            "service", "subsidiary")
        for it in results:
            desc = (it.get("description") or "").lower()
            if any(k in desc for k in company_keywords):
                return it.get("id")
        # Fallback: first result
        return results[0].get("id")
    except Exception:
        return None


async def _fetch_entity(client: httpx.AsyncClient, qid: str) -> Optional[Dict[str, Any]]:
    try:
        r = await client.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "format": "json",
                "props": "claims|labels|descriptions|sitelinks",
                "languages": "en",
            },
            headers=_headers(),
        )
        if r.status_code != 200:
            return None
        return r.json().get("entities", {}).get(qid)
    except Exception:
        return None


async def _resolve_label(client: httpx.AsyncClient, qid: str) -> str:
    """Resolve a Q-id to its English label."""
    entity = await _fetch_entity(client, qid)
    if not entity:
        return qid
    return entity.get("labels", {}).get("en", {}).get("value", qid)


def _extract_value(snak: Dict, prop_id: str) -> Optional[Any]:
    """Extract the human-readable value from a Wikidata snak."""
    dv = snak.get("datavalue") or {}
    val = dv.get("value")
    if val is None:
        return None
    dtype = dv.get("type")

    if dtype == "string":
        return val
    if dtype == "monolingualtext":
        return val.get("text")
    if dtype == "time":
        # ISO time like "+2008-04-01T00:00:00Z"
        t = val.get("time", "")
        if t.startswith("+"):
            t = t[1:]
        return t[:10] if len(t) >= 10 else t
    if dtype == "quantity":
        amount = val.get("amount", "0")
        return amount.lstrip("+")
    if dtype == "wikibase-entityid":
        # Returns the Q-id; caller should resolve to label
        return val.get("id")
    if dtype == "globe-coordinate":
        return f"{val.get('latitude','?')}, {val.get('longitude','?')}"
    return str(val)[:200]


async def _extract_claims(
    client: httpx.AsyncClient, entity: Dict[str, Any]
) -> Dict[str, Any]:
    """Pull every property we care about from the entity's claims."""
    out: Dict[str, Any] = {}
    claims = entity.get("claims", {}) or {}
    qid_to_resolve: List[str] = []

    for prop_id, key in PROPS.items():
        if prop_id not in claims:
            continue
        statements = claims[prop_id]
        if not statements:
            continue
        # Take the first (most-preferred) statement
        main = statements[0].get("mainsnak", {})
        v = _extract_value(main, prop_id)
        if v is None:
            continue
        out[key] = v
        # Note Q-ids that need resolving to labels
        if isinstance(v, str) and re.match(r"^Q\d+$", v):
            qid_to_resolve.append(v)

    # Batch-resolve Q-id values to labels
    resolved: Dict[str, str] = {}
    for qid in set(qid_to_resolve):
        label = await _resolve_label(client, qid)
        resolved[qid] = label

    for key, v in list(out.items()):
        if isinstance(v, str) and v in resolved:
            out[key] = resolved[v]

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def company_facts(name: str) -> Dict[str, Any]:
    """Resolve company name → Wikidata facts. Returns {} on miss."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        qid = await _search_entity(client, name)
        if not qid:
            return {}
        entity = await _fetch_entity(client, qid)
        if not entity:
            return {}
        facts = await _extract_claims(client, entity)
        if not facts:
            return {}
        # Add Wikidata metadata
        facts["_wikidata_id"] = qid
        facts["_wikidata_url"] = f"https://www.wikidata.org/wiki/{qid}"
        facts["_label"] = entity.get("labels", {}).get("en", {}).get("value", name)
        facts["_description"] = entity.get("descriptions", {}).get("en", {}).get("value", "")
        wp = (entity.get("sitelinks", {}) or {}).get("enwiki", {}) or {}
        if wp.get("title"):
            facts["_wikipedia_url"] = f"https://en.wikipedia.org/wiki/{wp['title'].replace(' ', '_')}"
        return facts


def facts_as_signals(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Wikidata facts into research-signal shape (T1) so they can
    feed the scoring pipeline alongside Nimble/GitHub/SEC/Court."""
    if not facts:
        return []
    wd_url = facts.get("_wikidata_url", "https://www.wikidata.org/")
    label = facts.get("_label", "Unknown")
    description = facts.get("_description", "")
    title = f"Wikidata: {label}"
    fact_parts = []
    for key in ("inception", "headquarters", "ceo", "founder", "employees",
                "industry", "country", "ticker_symbol", "stock_exchange"):
        if key in facts:
            fact_parts.append(f"{key}={facts[key]}")
    snippet = (description + " · " if description else "") + "; ".join(fact_parts[:6])
    return [{
        "title": title,
        "snippet": snippet[:350],
        "url": wd_url,
        "tier": 1,
        "tier_weight": 1.0,
        "tier_label": "Regulatory / first-party",
        "source_type": "wikidata_facts",
    }]
