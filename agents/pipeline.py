"""The 5-agent DealAgent pipeline + orchestrator.

Agents:
 1. Research (Nimble)        — parallel web search
 2. Scoring  (Claude)        — structured JSON scores with citations
 3. Benchmark (ClickHouse)   — historical comparison
 4. Publisher (Senso)        — publish to cited.md
 5. Payment  (x402)          — agent-to-agent USDC on Base
"""
import asyncio
import json
import re
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from core import config
from core import nimble_client
from core import clickhouse_client
from core import senso_client
from core import x402_client

try:
    from ddtrace import tracer
    _DD_OK = True
except Exception:
    tracer = None
    _DD_OK = False


ProgressCb = Optional[Callable[[str, str], Awaitable[None]]]


async def _emit(cb: ProgressCb, stage: str, message: str) -> None:
    if cb is None:
        return
    try:
        await cb(stage, message)
    except Exception:
        pass


def _span(name: str):
    """Context manager: real Datadog span if available, no-op otherwise."""
    if _DD_OK and tracer is not None:
        return tracer.trace(name, service=config.DD_SERVICE)

    class _Noop:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_tag(self, *a, **kw):
            pass

    return _Noop()


# -----------------------------------------------------------------------------
# Agent 1 — Research
# -----------------------------------------------------------------------------
async def research_agent(company: str, cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "research", f"Research Agent firing — 4 parallel Nimble searches on {company}...")
    with _span("dealagent.research") as span:
        span.set_tag("company", company)
        signals = await nimble_client.parallel_search(company)
        total = sum(len(v) for v in signals.values())
        span.set_tag("signals.count", total)
    await _emit(cb, "research", f"Research Agent complete — pulled {total} signals across 4 dimensions.")
    return signals


# -----------------------------------------------------------------------------
# Agent 2 — Scoring (Claude)
# -----------------------------------------------------------------------------
SCORING_PROMPT = """You are a senior VC analyst scoring a startup on 4 dimensions: team, market, traction, risk.

Company: {company}

Research signals (from live web search):
{research}

Score each dimension 0-10. For risk, higher score = MORE risk (so worse).
EVERY score must cite a specific source URL from the research signals above.
Return ONLY valid JSON, no prose, no markdown fences.

JSON schema:
{{
  "scores": {{
    "team":     {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url from research>"}},
    "market":   {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url from research>"}},
    "traction": {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url from research>"}},
    "risk":     {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url from research>"}}
  }},
  "verdict": "<one sharp investment verdict sentence>",
  "key_insight": "<the single most important finding>"
}}
"""


def _fallback_scores() -> Dict[str, Any]:
    return {
        "scores": {
            "team":     {"score": 5, "reasoning": "Data unavailable.", "source": ""},
            "market":   {"score": 5, "reasoning": "Data unavailable.", "source": ""},
            "traction": {"score": 5, "reasoning": "Data unavailable.", "source": ""},
            "risk":     {"score": 5, "reasoning": "Data unavailable.", "source": ""},
        },
        "verdict": "Insufficient data — manual review recommended.",
        "key_insight": "Scoring agent could not reach Claude API.",
    }


def _research_for_prompt(research: Dict[str, Any], cap: int = 6) -> str:
    chunks = []
    for cat, items in research.items():
        chunks.append(f"## {cat}")
        for it in items[:cap]:
            t = it.get("title", "")
            s = it.get("snippet", "")
            u = it.get("url", "")
            chunks.append(f"- {t} | {s} | {u}")
    return "\n".join(chunks)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


async def scoring_agent(company: str, research: Dict[str, Any], cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "scoring", "Scoring Agent firing — Claude analyzing signals with citations...")
    with _span("dealagent.scoring") as span:
        span.set_tag("company", company)

        if not config.have_anthropic():
            await _emit(cb, "scoring", "Scoring Agent complete — fallback scores (no Anthropic key).")
            return _fallback_scores()

        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
            prompt = SCORING_PROMPT.format(
                company=company,
                research=_research_for_prompt(research),
            )
            msg = await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            text = _strip_json_fence(text)
            parsed = json.loads(text)
            # sanity
            scores = parsed.get("scores", {})
            for dim in ("team", "market", "traction", "risk"):
                if dim not in scores:
                    scores[dim] = {"score": 5, "reasoning": "Missing.", "source": ""}
            parsed["scores"] = scores
            span.set_tag("ok", True)
            await _emit(cb, "scoring", "Scoring Agent complete — 4 dimensions scored with sources.")
            return parsed
        except Exception as e:
            span.set_tag("error", str(e)[:200])
            await _emit(cb, "scoring", f"Scoring Agent complete — fallback scores ({type(e).__name__}).")
            return _fallback_scores()


def _overall(scores: Dict[str, Any]) -> float:
    team = float(scores.get("team", {}).get("score", 0))
    market = float(scores.get("market", {}).get("score", 0))
    traction = float(scores.get("traction", {}).get("score", 0))
    risk = float(scores.get("risk", {}).get("score", 0))
    return round((team + market + traction + (10 - risk)) / 4.0, 2)


# -----------------------------------------------------------------------------
# Agent 3 — Benchmark
# -----------------------------------------------------------------------------
async def benchmark_agent(overall_score: float, cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "benchmark", "Benchmark Agent firing — ClickHouse historical comparison...")
    with _span("dealagent.benchmark") as span:
        span.set_tag("overall_score", overall_score)
        # ClickHouse calls are sync; offload to a thread so we don't block the loop.
        result = await asyncio.to_thread(clickhouse_client.benchmark, overall_score)
    total = result.get("total_in_db", 0)
    if total:
        msg = f"Benchmark Agent complete — {total} historical deals, {result.get('this_company_vs_avg', 'n/a')} avg by {abs(result.get('delta', 0))}."
    else:
        msg = f"Benchmark Agent complete — {result.get('note', 'no history yet')}."
    await _emit(cb, "benchmark", msg)
    return result


# -----------------------------------------------------------------------------
# Agent 4 — Publisher
# -----------------------------------------------------------------------------
async def publisher_agent(report: Dict[str, Any], cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "publish", "Publisher Agent firing — pushing report to cited.md via Senso...")
    with _span("dealagent.publish") as span:
        markdown = senso_client.format_report_markdown(report)
        out = await senso_client.publish(
            company=report["company_name"],
            report_id=report["report_id"],
            markdown=markdown,
        )
        span.set_tag("cited_url", out.get("cited_url", ""))
    await _emit(cb, "publish", f"Publisher Agent complete — live at {out['cited_url']}")
    return out


# -----------------------------------------------------------------------------
# Agent 5 — Payment
# -----------------------------------------------------------------------------
async def payment_agent(report_id: str, cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "payment", "Payment Agent firing — x402 micropayment on Base...")
    with _span("dealagent.payment") as span:
        result = await asyncio.to_thread(x402_client.pay, report_id)
        # log to ClickHouse (best effort)
        await asyncio.to_thread(
            clickhouse_client.insert_payment,
            report_id,
            result["amount_usd"],
            result["tx_hash"],
            result["payer"],
        )
        span.set_tag("tx_hash", result["tx_hash"])
    await _emit(cb, "payment", f"Payment Agent complete — {result['amount_usd']} USDC sent, tx {result['tx_hash'][:14]}…")
    return result


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
async def run_dealagent(company_name: str, progress_callback: ProgressCb = None) -> Dict[str, Any]:
    """End-to-end DealAgent run. Never raises — always returns a report dict."""
    company_name = (company_name or "").strip() or "Unknown"
    report_id = uuid.uuid4().hex[:16]

    with _span("dealagent.pipeline") as span:
        span.set_tag("company", company_name)
        span.set_tag("report_id", report_id)

        await _emit(progress_callback, "start", f"DealAgent kicking off due diligence on {company_name}...")

        # 1. Research
        research = await research_agent(company_name, progress_callback)

        # 2. Scoring
        scoring = await scoring_agent(company_name, research, progress_callback)
        scores = scoring.get("scores", {})
        overall_score = _overall(scores)

        # 3. Benchmark
        benchmark = await benchmark_agent(overall_score, progress_callback)

        # Build the report so far (publisher needs it)
        report: Dict[str, Any] = {
            "report_id": report_id,
            "company_name": company_name,
            "scores": scores,
            "overall_score": overall_score,
            "verdict": scoring.get("verdict", ""),
            "key_insight": scoring.get("key_insight", ""),
            "benchmark": benchmark,
            "research": research,
        }

        # 4. Publish
        pub = await publisher_agent(report, progress_callback)
        report["cited_url"] = pub.get("cited_url", "")
        report["publish_fallback"] = pub.get("fallback", False)

        # Persist report to ClickHouse before payment (best effort)
        await asyncio.to_thread(clickhouse_client.insert_report, report)

        # 5. Payment
        payment = await payment_agent(report_id, progress_callback)
        report["payment"] = payment

        # Sources count for the UI
        sources_count = sum(len(v) for v in research.values())
        report["sources_count"] = sources_count

        span.set_tag("overall_score", overall_score)
        span.set_tag("sources_count", sources_count)

    await _emit(progress_callback, "complete", "DealAgent complete — all 5 agents fired.")
    return report
