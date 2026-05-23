"""Senso publisher — pushes the DD report to cited.md as an agent-readable citeable."""
import re
from typing import Dict, Any
import httpx

from core import config


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "company"


def _fallback_url(company: str) -> str:
    return f"https://cited.md/dealagent/{_slug(company)}"


async def publish(company: str, report_id: str, markdown: str) -> Dict[str, Any]:
    """POST report to Senso. Returns {cited_url, success}. Never raises."""
    if not config.have_senso():
        return {"cited_url": _fallback_url(company), "success": True, "fallback": True}

    headers = {
        "Authorization": f"Bearer {config.SENSO_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "title": f"Due Diligence: {company}",
        "content": markdown,
        "source_url": f"https://dealagent.ai/reports/{report_id}",
        "tags": ["due-diligence", "startup", company],
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                config.SENSO_URL,
                json=payload,
                headers=headers,
                timeout=config.HTTP_TIMEOUT,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                url = (
                    data.get("cited_url")
                    or data.get("url")
                    or data.get("public_url")
                    or _fallback_url(company)
                )
                return {"cited_url": url, "success": True, "fallback": False}
            return {"cited_url": _fallback_url(company), "success": True, "fallback": True}
    except Exception:
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
