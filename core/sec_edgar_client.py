"""SEC EDGAR agent — T1 regulatory signal for public companies.

SEC filings are the gold standard for due diligence: lying on a 10-K,
10-Q, S-1, or Form D is securities fraud with personal liability for
the executives who sign. We treat any EDGAR signal as the highest
credibility tier (T1).

EDGAR is free and requires no auth. The API just demands a polite
User-Agent header per SEC's fair-use policy.

Flow:
  1. Resolve the company name to a CIK (Central Index Key)
     via the public company tickers JSON.
  2. Fetch the company's recent filings via
     https://data.sec.gov/submissions/CIK{padded}.json
  3. Pick out the most recent 10-K, 10-Q, S-1, 8-K, Form D filings.

If the company isn't a SEC registrant (private startup), we return
an empty list — the rest of the pipeline degrades gracefully.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import httpx


EDGAR_BASE = "https://data.sec.gov"
SEC_BASE = "https://www.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
UA = "DealAgent srujan-codes/dealagent v2 (research@dealagent.ai)"

# Filing types most relevant to VC due diligence, in priority order.
INTERESTING_FORMS = ["10-K", "10-Q", "S-1", "S-1/A", "20-F", "8-K", "DEF 14A", "Form D"]


def _headers() -> Dict[str, str]:
    return {"User-Agent": UA, "Accept": "application/json"}


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


_TICKERS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


async def _load_tickers(client: httpx.AsyncClient) -> Dict[str, Dict[str, Any]]:
    """Load the SEC's full company tickers map once and cache it."""
    global _TICKERS_CACHE
    if _TICKERS_CACHE is not None:
        return _TICKERS_CACHE
    try:
        r = await client.get(TICKERS_URL, headers=_headers())
        if r.status_code != 200:
            return {}
        data = r.json()
        # SEC returns {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        out: Dict[str, Dict[str, Any]] = {}
        for v in data.values():
            if not isinstance(v, dict):
                continue
            title = v.get("title", "")
            ticker = v.get("ticker", "")
            cik = v.get("cik_str")
            if not cik:
                continue
            key_title = _normalize(title)
            key_ticker = _normalize(ticker)
            if key_title:
                out[key_title] = v
            if key_ticker:
                out[key_ticker] = v
        _TICKERS_CACHE = out
        return out
    except Exception:
        return {}


def _pad_cik(cik) -> str:
    return f"{int(cik):010d}"


async def _resolve_cik(client: httpx.AsyncClient, company: str) -> Optional[Dict[str, Any]]:
    """Try multiple variations of the company name to find a CIK."""
    tickers = await _load_tickers(client)
    if not tickers:
        return None

    norm = _normalize(company)
    if norm in tickers:
        return tickers[norm]

    # Try with common suffixes added
    for suffix in ("inc", "corp", "corporation", "holdings", "group", "company", "co"):
        cand = norm + suffix
        if cand in tickers:
            return tickers[cand]

    # Try partial substring match (e.g. "Stripe" → "Stripe Inc")
    for key, val in tickers.items():
        if norm and norm in key and len(norm) >= 4:
            return val

    return None


async def _recent_filings(client: httpx.AsyncClient, cik: int) -> List[Dict[str, Any]]:
    """Get the company's submissions feed and pluck the interesting form types."""
    padded = _pad_cik(cik)
    try:
        r = await client.get(
            f"{EDGAR_BASE}/submissions/CIK{padded}.json", headers=_headers()
        )
        if r.status_code != 200:
            return []
        sub = r.json()
    except Exception:
        return []

    recent = (sub.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    seen_forms: set = set()
    out: List[Dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form not in INTERESTING_FORMS:
            continue
        if form in seen_forms:
            # Only keep the most recent of each type
            continue
        seen_forms.add(form)

        acc = accession_numbers[i].replace("-", "") if i < len(accession_numbers) else ""
        doc = primary_docs[i] if i < len(primary_docs) else ""
        date = filing_dates[i] if i < len(filing_dates) else ""
        desc = descriptions[i] if i < len(descriptions) else ""

        url = f"{SEC_BASE}/Archives/edgar/data/{cik}/{acc}/{doc}" if acc and doc else ""
        out.append({
            "form": form,
            "filing_date": date,
            "description": desc,
            "url": url,
            "accession": accession_numbers[i] if i < len(accession_numbers) else "",
        })
        if len(out) >= 8:
            break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def regulatory_signals(company: str) -> List[Dict[str, Any]]:
    """Return a list of T1 SEC EDGAR signals shaped like Nimble research items."""
    out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        match = await _resolve_cik(client, company)
        if not match:
            return out
        cik = match.get("cik_str")
        title = match.get("title", company)
        ticker = match.get("ticker", "")

        # Cover signal: the company itself is SEC-registered
        out.append({
            "title": f"SEC EDGAR registrant: {title}",
            "snippet": (
                f"CIK {cik} · ticker {ticker}. "
                f"SEC-registered public company — filings have legal accountability."
            ),
            "url": f"{SEC_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=40",
            "tier": 1,
            "tier_weight": 1.0,
            "tier_label": "Regulatory / first-party",
            "source_type": "sec_registrant",
        })

        filings = await _recent_filings(client, cik)
        for f in filings:
            out.append({
                "title": f"SEC {f['form']} filed {f['filing_date']}",
                "snippet": (
                    f"{f.get('description', '')[:160]} · "
                    f"Accession {f.get('accession', '')}"
                ),
                "url": f["url"],
                "tier": 1,
                "tier_weight": 1.0,
                "tier_label": "Regulatory / first-party",
                "source_type": f"sec_{f['form'].lower().replace(' ', '_').replace('/', '_')}",
                "filing_date": f["filing_date"],
                "form": f["form"],
            })
    return out
