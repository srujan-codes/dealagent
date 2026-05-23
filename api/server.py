"""FastAPI server — serves UI, runs pipeline, streams progress via SSE."""
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.pipeline import run_dealagent
from core.senso_client import format_report_markdown


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="DealAgent", version="1.0.0")


# ---------------------------------------------------------------------------
# In-memory report cache — keyed by slug, populated on every analysis.
# Used by GET /r/{slug} so the published-report link in the UI always resolves
# even if Senso hasn't published yet (destination not selected in Senso dash).
# ---------------------------------------------------------------------------
_REPORT_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_MAX = 256  # rolling cap so we don't grow forever


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "company"


def _save_report(report: Dict[str, Any]) -> None:
    """Cache the rendered HTML for a report so /r/{slug} can serve it."""
    slug = _slug(report.get("company_name", "company"))
    html = _render_report_html(report)
    _REPORT_CACHE[slug] = {
        "html": html,
        "saved_at": time.time(),
        "report": report,
    }
    # rolling LRU-ish eviction
    if len(_REPORT_CACHE) > _CACHE_MAX:
        oldest = min(_REPORT_CACHE, key=lambda k: _REPORT_CACHE[k]["saved_at"])
        _REPORT_CACHE.pop(oldest, None)


def _local_report_url(slug: str, request_host: str = "localhost:8000") -> str:
    return f"http://{request_host}/r/{slug}"


def _maybe_override_cited_url(report: Dict[str, Any], host: str) -> Dict[str, Any]:
    """If Senso publish fell back, replace the synthetic cited.md URL with our
    own /r/{slug} URL so the link in the UI actually resolves to a real page."""
    if report.get("publish_fallback"):
        slug = _slug(report.get("company_name", "company"))
        report["cited_url"] = _local_report_url(slug, host)
        report["served_locally"] = True
    else:
        report["served_locally"] = False
    return report


class AnalyzeBody(BaseModel):
    company_name: str


@app.get("/")
async def index() -> HTMLResponse:
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>DealAgent</h1><p>Frontend not found.</p>", status_code=200)


@app.get("/health")
async def health():
    return {"status": "running"}


@app.post("/api/analyze")
async def analyze(body: AnalyzeBody):
    """Synchronous endpoint — returns the full report JSON when done."""
    report = await run_dealagent(body.company_name)
    _save_report(report)
    report = _maybe_override_cited_url(report, "localhost:8000")
    return JSONResponse(report)


@app.get("/api/analyze/stream")
async def analyze_stream(company: str):
    """SSE stream of progress events. Terminates with a `report` event."""
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(stage: str, message: str) -> None:
        await queue.put({"stage": stage, "message": message})

    async def runner():
        try:
            report = await run_dealagent(company, progress_callback=progress)
            _save_report(report)
            report = _maybe_override_cited_url(report, "localhost:8000")
            await queue.put({"stage": "report", "report": report})
        except Exception as e:
            await queue.put({"stage": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put(None)  # sentinel

    async def event_stream():
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/r/{slug}")
async def serve_report(slug: str) -> HTMLResponse:
    """Render a cached report as a public-style HTML page."""
    entry = _REPORT_CACHE.get(slug.lower())
    if not entry:
        raise HTTPException(status_code=404, detail=f"No report cached for /{slug}. Run an analysis first.")
    return HTMLResponse(entry["html"])


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
_REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>DealAgent · Due Diligence · {company}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@500;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0a0f; --surface: #111118; --surface-2: #161620; --border: #22222e;
    --text: #e5e7eb; --muted: #8b8ba0; --purple: #7c6dfa; --teal: #00d4aa;
    --red: #ff5a6b; --orange: #ffaa3b; --green: #3ddc97;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: 'Space Mono', monospace; min-height: 100vh; }}
  body::before {{ content: ""; position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(124,109,250,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,109,250,0.04) 1px, transparent 1px);
    background-size: 32px 32px; pointer-events: none; z-index: 0; }}
  .wrap {{ position: relative; z-index: 1; max-width: 880px; margin: 0 auto;
    padding: 56px 24px 80px; }}
  .badge {{ display: inline-block; font-size: 11px; letter-spacing: 2px;
    color: var(--purple); border: 1px solid var(--purple); padding: 6px 12px;
    border-radius: 999px; text-transform: uppercase; margin-bottom: 18px; }}
  h1 {{ font-family: 'Syne', sans-serif; font-size: 56px; font-weight: 800;
    margin: 0 0 8px; letter-spacing: -1.5px;
    background: linear-gradient(90deg, #fff, var(--purple));
    -webkit-background-clip: text; background-clip: text; color: transparent; }}
  .verdict {{ background: rgba(124,109,250,0.1); border: 1px solid var(--purple);
    border-radius: 999px; padding: 8px 16px; font-size: 13px; color: var(--purple);
    display: inline-block; margin-bottom: 18px; }}
  .overall {{ font-family: 'Syne', sans-serif; font-size: 64px; font-weight: 800;
    color: var(--teal); line-height: 1; margin: 24px 0 6px; }}
  .insight {{ color: var(--text); font-size: 15px; line-height: 1.55;
    max-width: 700px; margin: 0 0 32px; }}
  .scores {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;
    margin-bottom: 24px; }}
  .score {{ background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; }}
  .score .lbl {{ font-size: 11px; letter-spacing: 2px; color: var(--muted);
    text-transform: uppercase; }}
  .score .v {{ font-family: 'Syne', sans-serif; font-size: 48px; font-weight: 800;
    line-height: 1; margin: 8px 0 12px; }}
  .score .reason {{ font-size: 13px; color: var(--text); line-height: 1.55; }}
  .score .src {{ display: block; font-size: 11px; color: var(--purple);
    margin-top: 10px; word-break: break-all; text-decoration: none; }}
  .score .src:hover {{ text-decoration: underline; }}
  .section-h {{ font-size: 11px; letter-spacing: 2px; color: var(--muted);
    text-transform: uppercase; margin: 28px 0 12px; }}
  .meta {{ background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; margin-bottom: 18px; }}
  .meta .row {{ display: flex; justify-content: space-between; gap: 10px;
    font-size: 13px; padding: 6px 0; border-bottom: 1px solid var(--border); }}
  .meta .row:last-child {{ border-bottom: none; }}
  .meta .k {{ color: var(--muted); }}
  .meta .v {{ color: var(--text); word-break: break-all; text-align: right; }}
  .citations {{ background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; }}
  .citations h3 {{ font-family: 'Syne', sans-serif; font-size: 14px;
    color: var(--purple); margin: 12px 0 6px; text-transform: capitalize; }}
  .citations ul {{ list-style: none; padding: 0; margin: 0; }}
  .citations li {{ font-size: 12px; line-height: 1.55; padding: 4px 0; }}
  .citations a {{ color: var(--text); text-decoration: none;
    border-bottom: 1px dotted var(--muted); }}
  .citations a:hover {{ color: var(--purple); border-bottom-color: var(--purple); }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border);
    color: var(--muted); font-size: 11px; line-height: 1.6; }}
  .footer a {{ color: var(--purple); }}
</style>
</head>
<body>
<div class="wrap">
  <span class="badge">DealAgent · Autonomous Due Diligence</span>
  <h1>{company}</h1>
  <div class="verdict">{verdict}</div>

  <div class="overall">{overall} / 10</div>
  <p class="insight">{key_insight}</p>

  <div class="section-h">Scores</div>
  <div class="scores">{score_cards}</div>

  <div class="section-h">Report Metadata</div>
  <div class="meta">{meta_rows}</div>

  <div class="section-h">Citations</div>
  <div class="citations">{citations}</div>

  <div class="footer">
    Generated by DealAgent — 5 autonomous agents, 6 sponsor integrations:
    <strong>Nimble</strong> (live web), <strong>Groq</strong> (Llama 3.1 8B),
    <strong>ClickHouse</strong> (benchmark), <strong>Senso</strong> (publish),
    <strong>Datadog</strong> (LLM Obs), <strong>x402</strong> (agent payment).<br>
    Source code · <a href="https://github.com/srujan-codes/dealagent">github.com/srujan-codes/dealagent</a>
  </div>
</div>
</body>
</html>"""


def _escape(s: Any) -> str:
    s = "" if s is None else str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def _color_for(score: float, is_risk: bool = False) -> str:
    v = (10 - score) if is_risk else score
    if v >= 7:
        return "var(--green)"
    if v >= 4:
        return "var(--orange)"
    return "var(--red)"


def _render_score_card(label: str, key: str, data: Dict[str, Any]) -> str:
    score = float(data.get("score", 0) or 0)
    is_risk = key == "risk"
    color = _color_for(score, is_risk)
    suffix = " (lower = better)" if is_risk else ""
    src = data.get("source", "") or ""
    src_html = ""
    if src and src.startswith("http"):
        src_html = f'<a class="src" href="{_escape(src)}" target="_blank" rel="noopener">{_escape(src)}</a>'
    return f"""<div class="score">
        <div class="lbl">{_escape(label)}{suffix}</div>
        <div class="v" style="color:{color}">{score:.0f}</div>
        <div class="reason">{_escape(data.get('reasoning', ''))}</div>
        {src_html}
    </div>"""


def _render_report_html(report: Dict[str, Any]) -> str:
    company = report.get("company_name", "Unknown")
    scores = report.get("scores", {})
    score_cards = "\n".join(
        _render_score_card(label, key, scores.get(key, {}))
        for key, label in [
            ("team", "Team"), ("market", "Market"),
            ("traction", "Traction"), ("risk", "Risk"),
        ]
    )

    bench = report.get("benchmark", {})
    if bench.get("total_in_db", 0) > 0:
        bench_text = (
            f"{bench.get('this_company_vs_avg','')} avg by "
            f"{abs(bench.get('delta', 0)):.2f} (n={bench.get('total_in_db', 0)})"
        )
    else:
        bench_text = bench.get("note", "—")

    payment = report.get("payment", {}) or {}
    tx_hash = payment.get("tx_hash", "")
    explorer = payment.get("explorer_url", "")
    if explorer:
        tx_cell = f'<a href="{_escape(explorer)}" target="_blank" rel="noopener" style="color:var(--teal)">{_escape(tx_hash[:32])}…</a>'
    else:
        tx_cell = _escape(tx_hash[:32] + "…" if tx_hash else "—")

    meta_rows = "\n".join([
        f'<div class="row"><span class="k">Report ID</span><span class="v">{_escape(report.get("report_id", "—"))}</span></div>',
        f'<div class="row"><span class="k">Benchmark</span><span class="v">{_escape(bench_text)}</span></div>',
        f'<div class="row"><span class="k">Sources analyzed</span><span class="v">{_escape(report.get("sources_count", "—"))}</span></div>',
        f'<div class="row"><span class="k">Senso prompt</span><span class="v">{_escape(report.get("senso_prompt_id") or "—")}</span></div>',
        f'<div class="row"><span class="k">Payment tx</span><span class="v">{tx_cell}</span></div>',
        f'<div class="row"><span class="k">Payment amount</span><span class="v">{payment.get("amount_usd", 0)} USDC on {_escape(payment.get("network","—"))}</span></div>',
        f'<div class="row"><span class="k">Total runtime</span><span class="v">{report.get("timing_ms", {}).get("total", 0)/1000:.2f}s</span></div>',
    ])

    citations_html = []
    research = report.get("research", {}) or {}
    for cat, items in research.items():
        if not items:
            continue
        cat_label = cat.replace("_", " ").replace(" signals", "").title()
        lis = []
        for it in items[:5]:
            url = it.get("url", "")
            title = it.get("title", "") or url
            if url and url.startswith("http"):
                lis.append(f'<li><a href="{_escape(url)}" target="_blank" rel="noopener">{_escape(title)}</a></li>')
        if lis:
            citations_html.append(f'<h3>{_escape(cat_label)}</h3><ul>{"".join(lis)}</ul>')

    return _REPORT_TEMPLATE.format(
        company=_escape(company),
        verdict=_escape(report.get("verdict", "")),
        overall=f"{float(report.get('overall_score', 0)):.1f}",
        key_insight=_escape(report.get("key_insight", "")),
        score_cards=score_cards,
        meta_rows=meta_rows,
        citations="\n".join(citations_html) or "<p>No citations available.</p>",
    )
