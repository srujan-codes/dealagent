"""Senso publisher — pushes the DD report to cited.md as an agent-readable citeable.

Per Senso's docs (https://docs.senso.ai):
  base URL: https://apiv2.senso.ai/api/v1
  auth:    `X-API-Key: <key>` header (NOT Bearer)

The exact content-creation endpoint name is gated behind their auth-required
API reference, so we probe a few likely paths and use the first that returns 2xx.
Any failure degrades to a synthetic cited.md/<slug> URL so the demo never breaks.
"""
import re
from typing import Dict, Any, List
import httpx

from core import config

SENSO_BASE = "https://apiv2.senso.ai/api/v1"

# Likely endpoint paths, tried in order. First 2xx wins.
CANDIDATE_PATHS: List[str] = ["/content", "/contents", "/citeables", "/documents", "/ingest"]


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "company"


def _fallback_url(company: str) -> str:
    return f"https://cited.md/dealagent/{_slug(company)}"


def _extract_url(data: Any, company: str) -> str:
    """Senso responses don't have a fully documented public schema, so try
    common URL field names and gracefully fall back."""
    if isinstance(data, dict):
        for k in ("cited_url", "public_url", "url", "permalink", "link", "share_url"):
            v = data.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        # nested {"data": {...}} or {"content": {...}}
        for nested in ("data", "content", "citeable", "document"):
            inner = data.get(nested)
            if isinstance(inner, dict):
                got = _extract_url(inner, company)
                if got != _fallback_url(company):
                    return got
    return _fallback_url(company)


async def publish(company: str, report_id: str, markdown: str) -> Dict[str, Any]:
    """POST report to Senso. Returns {cited_url, success, fallback}. Never raises."""
    if not config.have_senso():
        return {"cited_url": _fallback_url(company), "success": True, "fallback": True}

    headers = {
        "X-API-Key": config.SENSO_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "title": f"Due Diligence: {company}",
        "content": markdown,
        "source_url": f"https://dealagent.ai/reports/{report_id}",
        "tags": ["due-diligence", "startup", company],
    }

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            for path in CANDIDATE_PATHS:
                url = f"{SENSO_BASE}{path}"
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                except Exception:
                    continue
                if 200 <= resp.status_code < 300:
                    try:
                        data = resp.json()
                    except Exception:
                        data = {}
                    return {
                        "cited_url": _extract_url(data, company),
                        "success": True,
                        "fallback": False,
                        "endpoint": path,
                    }
                # 404 / 405 → try next candidate. Other codes still mean Senso is up,
                # but the key/payload is wrong — no point continuing.
                if resp.status_code not in (404, 405):
                    break
    except Exception:
        pass

    return {"cited_url": _fallback_url(company), "success": True, "fallback": True}


def format_report_markdown(report: Dict[str, Any]) -> str:
    """Render full report as markdown with citations."""
    scores = report.get("scores", {})
    lines = [
        f"# Due Diligence: {report.get('company_name', 'Unknown')}",
        "",
        f"**Overall Score:** {report.get('overall_score', 0):.1f} / 10",
        f"**Verdict:** {report.get('verdict', '')}",
        f"**Key Insight:** {report.get('key_insight', '')}",
        "",
        "## Scores",
        "",
    ]
    for dim in ("team", "market", "traction", "risk"):
        s = scores.get(dim, {})
        lines.append(f"### {dim.title()} — {s.get('score', 0)} / 10")
        lines.append(s.get("reasoning", ""))
        src = s.get("source", "")
        if src:
            lines.append(f"Source: {src}")
        lines.append("")

    bench = report.get("benchmark", {})
    if bench:
        lines.append("## Historical Benchmark")
        for k, v in bench.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Citations")
    research = report.get("research", {})
    for cat, items in research.items():
        if not items:
            continue
        lines.append(f"### {cat}")
        for it in items[:5]:
            t = it.get("title", "")
            u = it.get("url", "")
            if u:
                lines.append(f"- [{t}]({u})")
        lines.append("")

    return "\n".join(lines)
