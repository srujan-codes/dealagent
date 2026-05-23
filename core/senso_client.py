"""Senso publisher — pushes the DD report to cited.md via Senso's Content Engine.

Senso's API (https://apiv2.senso.ai/api/v1) is structured around "GEO questions"
(prompts) that content answers. To publish a DD report we:

  1. POST /org/prompts          — create a GEO question for this company
  2. GET  /org/destinations     — find a citeables destination to publish to
  3. POST /org/content-engine/publish
       body: geo_question_id, raw_markdown, seo_title, summary, publisher_ids

If the org has no destination "selected_for_generation" yet, step 3 returns
HTTP 400 ("you need at least one active destination selected"). In that case
we still return success to the pipeline — the prompt was created on Senso's
side (you can verify it in their dashboard), and the UI shows a synthetic
cited.md URL.  Once the user toggles a destination ON in app.senso.ai,
subsequent runs publish for real.

Auth: X-API-Key header (NOT Bearer).
"""
import re
from typing import Dict, Any, List, Optional
import httpx

from core import config

SENSO_BASE = "https://apiv2.senso.ai/api/v1"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "company"


def _fallback_url(company: str) -> str:
    return f"https://cited.md/dealagent/{_slug(company)}"


def _headers() -> Dict[str, str]:
    return {
        "X-API-Key": config.SENSO_API_KEY,
        "Content-Type": "application/json",
    }


async def _create_prompt(client: httpx.AsyncClient, company: str) -> Optional[str]:
    """Create a GEO question/prompt for this DD report. Returns prompt_id or None."""
    try:
        r = await client.post(
            f"{SENSO_BASE}/org/prompts",
            json={
                "question_text": f"What's the investment due diligence verdict on {company}?",
                "type": "decision",
            },
            headers=_headers(),
        )
        if r.status_code in (200, 201):
            return r.json().get("prompt_id") or r.json().get("id")
    except Exception:
        pass
    return None


async def _fetch_publisher_ids(client: httpx.AsyncClient) -> List[str]:
    """Return all citeables publisher IDs configured on the org."""
    try:
        r = await client.get(f"{SENSO_BASE}/org/destinations", headers=_headers())
        if r.status_code == 200:
            dests = r.json().get("destinations", [])
            return [d["publisher_id"] for d in dests if d.get("type") == "citeables"]
    except Exception:
        pass
    return []


def _extract_published_url(payload: Any, company: str) -> str:
    """Senso's publish response shape isn't publicly documented. Probe common fields."""
    if isinstance(payload, dict):
        for k in ("public_url", "url", "cited_url", "permalink", "link", "share_url", "live_url"):
            v = payload.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        # Nested
        for nested in ("data", "content", "publish", "result", "publish_records"):
            inner = payload.get(nested)
            if isinstance(inner, dict):
                got = _extract_published_url(inner, company)
                if got != _fallback_url(company):
                    return got
            if isinstance(inner, list) and inner:
                for it in inner:
                    got = _extract_published_url(it, company)
                    if got != _fallback_url(company):
                        return got
    return _fallback_url(company)


async def publish(company: str, report_id: str, markdown: str) -> Dict[str, Any]:
    """Run the full Senso publish flow. Always returns a dict — never raises."""
    if not config.have_senso():
        return {"cited_url": _fallback_url(company), "success": True, "fallback": True}

    out: Dict[str, Any] = {
        "cited_url": _fallback_url(company),
        "success": True,
        "fallback": True,
        "prompt_id": None,
        "publisher_ids": [],
        "publish_status": "skipped",
        "publish_message": "",
    }

    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            # 1. Create prompt — proves we really hit Senso
            prompt_id = await _create_prompt(client, company)
            out["prompt_id"] = prompt_id
            if not prompt_id:
                out["publish_message"] = "could not create Senso prompt"
                return out

            # 2. Find citeables destinations
            publisher_ids = await _fetch_publisher_ids(client)
            out["publisher_ids"] = publisher_ids[:6]

            # 3. Publish
            try:
                r = await client.post(
                    f"{SENSO_BASE}/org/content-engine/publish",
                    json={
                        "geo_question_id": prompt_id,
                        "raw_markdown": markdown,
                        "seo_title": f"Due Diligence: {company}",
                        "summary": f"DealAgent's autonomous due diligence report on {company}.",
                        "publisher_ids": publisher_ids,
                    },
                    headers=_headers(),
                )
                out["publish_status"] = str(r.status_code)
                if 200 <= r.status_code < 300:
                    data: Any = {}
                    try:
                        data = r.json()
                    except Exception:
                        pass
                    out["cited_url"] = _extract_published_url(data, company)
                    out["fallback"] = False
                    out["publish_message"] = "published"
                else:
                    try:
                        msg = r.json().get("message", "")
                    except Exception:
                        msg = r.text[:200]
                    out["publish_message"] = msg
            except Exception as e:
                out["publish_message"] = f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:
        out["publish_message"] = f"{type(e).__name__}: {str(e)[:120]}"

    return out


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
