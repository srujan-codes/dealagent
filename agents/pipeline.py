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
import time
import uuid
from typing import Any, Awaitable, Callable, Coroutine, Dict, Optional

from core import config
from core import nimble_client
from core import clickhouse_client
from core import senso_client
from core import x402_client

# ---------------------------------------------------------------------------
# Datadog LLM Observability
# ---------------------------------------------------------------------------
# When DD_API_KEY is set, we wire each agent into Datadog's LLM Observability
# product so judges can see prompts/completions/latency in real time at
# https://app.datadoghq.com/llm/traces. Without the key the same calls become
# no-ops so the pipeline still runs everywhere.
LLMObs = None
_LLMOBS_ENABLED = False

try:
    from ddtrace.llmobs import LLMObs as _LLMObs  # type: ignore
    LLMObs = _LLMObs
    if config.have_datadog():
        try:
            LLMObs.enable(
                ml_app=config.DD_SERVICE,
                api_key=config.DD_API_KEY,
                site=config.DD_SITE,
                agentless_enabled=True,
            )
            _LLMOBS_ENABLED = True
        except Exception:
            _LLMOBS_ENABLED = False
except Exception:
    LLMObs = None
    _LLMOBS_ENABLED = False


ProgressCb = Optional[Callable[[str, str], Awaitable[None]]]


async def _emit(cb: ProgressCb, stage: str, message: str) -> None:
    if cb is None:
        return
    try:
        await cb(stage, message)
    except Exception:
        pass


class _NoopSpan:
    """Stand-in span used whenever Datadog isn't initialized."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def set_tag(self, *a, **kw):
        pass


def _workflow(name: str):
    if _LLMOBS_ENABLED and LLMObs is not None:
        try:
            return LLMObs.workflow(name=name)
        except Exception:
            return _NoopSpan()
    return _NoopSpan()


def _task(name: str):
    if _LLMOBS_ENABLED and LLMObs is not None:
        try:
            return LLMObs.task(name=name)
        except Exception:
            return _NoopSpan()
    return _NoopSpan()


def _llm(name: str, model_name: str, model_provider: str):
    if _LLMOBS_ENABLED and LLMObs is not None:
        try:
            return LLMObs.llm(name=name, model_name=model_name, model_provider=model_provider)
        except Exception:
            return _NoopSpan()
    return _NoopSpan()


def _annotate(**kw):
    """LLMObs.annotate() — silent no-op when Datadog isn't enabled."""
    if _LLMOBS_ENABLED and LLMObs is not None:
        try:
            LLMObs.annotate(**kw)
        except Exception:
            pass


async def _timed(name: str, coro: Coroutine, sink: Dict[str, float]) -> Any:
    """Run coro, record wall-clock duration (ms) into sink[name]."""
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        sink[name] = round((time.perf_counter() - t0) * 1000, 1)


# -----------------------------------------------------------------------------
# Agent 1 — Research
# -----------------------------------------------------------------------------
async def research_agent(company: str, cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "research", f"Research Agent firing — 4 parallel Nimble searches on {company}...")
    with _task("dealagent.research"):
        signals = await nimble_client.parallel_search(company)
        total = sum(len(v) for v in signals.values())
        _annotate(
            input_data=f"company={company}",
            output_data=f"{total} signals across 4 dimensions",
            tags={"company": company, "agent": "research", "tool": "nimble"},
        )
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
    await _emit(cb, "scoring", f"Scoring Agent firing — Groq ({config.GROQ_MODEL}) analyzing signals...")
    with _llm("dealagent.scoring", model_name=config.GROQ_MODEL, model_provider="groq"):
        if not config.have_groq():
            _annotate(
                input_data="(no Groq key configured)",
                output_data="fallback 5/5/5/5",
                tags={"company": company, "fallback": True},
            )
            await _emit(cb, "scoring", "Scoring Agent complete — fallback scores (no Groq key).")
            return _fallback_scores()

        try:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=config.GROQ_API_KEY)
            prompt = SCORING_PROMPT.format(
                company=company,
                research=_research_for_prompt(research),
            )
            completion = await client.chat.completions.create(
                model=config.GROQ_MODEL,
                max_tokens=1500,
                temperature=0.3,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or ""
            text_clean = _strip_json_fence(text)
            parsed = json.loads(text_clean)
            # sanity
            scores = parsed.get("scores", {})
            for dim in ("team", "market", "traction", "risk"):
                if dim not in scores:
                    scores[dim] = {"score": 5, "reasoning": "Missing.", "source": ""}
            parsed["scores"] = scores

            # Annotate the LLM span with the full prompt+completion for Datadog
            usage = getattr(completion, "usage", None)
            metadata = {
                "temperature": 0.3,
                "max_tokens": 1500,
                "response_format": "json_object",
            }
            metrics = {}
            if usage is not None:
                metrics["input_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
                metrics["output_tokens"] = getattr(usage, "completion_tokens", 0) or 0
                metrics["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
            _annotate(
                input_data=[{"role": "user", "content": prompt}],
                output_data=[{"role": "assistant", "content": text}],
                metadata=metadata,
                metrics=metrics,
                tags={"company": company, "agent": "scoring"},
            )

            await _emit(cb, "scoring", "Scoring Agent complete — 4 dimensions scored with sources.")
            return parsed
        except Exception as e:
            _annotate(
                input_data=str(prompt) if "prompt" in locals() else "(prompt failed to build)",
                output_data=f"ERROR: {type(e).__name__}: {str(e)[:200]}",
                tags={"company": company, "agent": "scoring", "error": True},
            )
            await _emit(cb, "scoring", f"Scoring Agent complete — fallback scores ({type(e).__name__}: {str(e)[:80]}).")
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
    with _task("dealagent.benchmark"):
        # ClickHouse calls are sync; offload to a thread so we don't block the loop.
        result = await asyncio.to_thread(clickhouse_client.benchmark, overall_score)
        _annotate(
            input_data=f"overall_score={overall_score}",
            output_data=json.dumps(result),
            tags={"agent": "benchmark", "tool": "clickhouse"},
        )
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
    with _task("dealagent.publish"):
        markdown = senso_client.format_report_markdown(report)
        out = await senso_client.publish(
            company=report["company_name"],
            report_id=report["report_id"],
            markdown=markdown,
        )
        _annotate(
            input_data=f"company={report['company_name']}, report_id={report['report_id']}",
            output_data=out.get("cited_url", ""),
            tags={"agent": "publish", "tool": "senso", "fallback": out.get("fallback", False)},
        )
    await _emit(cb, "publish", f"Publisher Agent complete — live at {out['cited_url']}")
    return out


# -----------------------------------------------------------------------------
# Agent 5 — Payment
# -----------------------------------------------------------------------------
async def payment_agent(report_id: str, cb: ProgressCb) -> Dict[str, Any]:
    await _emit(cb, "payment", "Payment Agent firing — x402 micropayment on Base...")
    with _task("dealagent.payment"):
        result = await asyncio.to_thread(x402_client.pay, report_id)
        # log to ClickHouse (best effort)
        await asyncio.to_thread(
            clickhouse_client.insert_payment,
            report_id,
            result["amount_usd"],
            result["tx_hash"],
            result["payer"],
        )
        _annotate(
            input_data=f"report_id={report_id}",
            output_data=f"tx={result['tx_hash']} amount=${result['amount_usd']} USDC",
            tags={"agent": "payment", "tool": "x402", "network": result.get("network", "base")},
        )
    await _emit(cb, "payment", f"Payment Agent complete — {result['amount_usd']} USDC sent, tx {result['tx_hash'][:14]}…")
    return result


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
async def run_dealagent(company_name: str, progress_callback: ProgressCb = None) -> Dict[str, Any]:
    """End-to-end DealAgent run. Never raises — always returns a report dict."""
    company_name = (company_name or "").strip() or "Unknown"
    report_id = uuid.uuid4().hex[:16]
    timing: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    with _workflow("dealagent.pipeline"):
        _annotate(
            input_data=f"company={company_name}",
            tags={"company": company_name, "report_id": report_id},
        )
        await _emit(progress_callback, "start", f"DealAgent kicking off due diligence on {company_name}...")

        # 1. Research
        research = await _timed(
            "research",
            research_agent(company_name, progress_callback),
            timing,
        )

        # 2. Scoring
        scoring = await _timed(
            "scoring",
            scoring_agent(company_name, research, progress_callback),
            timing,
        )
        scores = scoring.get("scores", {})
        overall_score = _overall(scores)

        # 3. Benchmark
        benchmark = await _timed(
            "benchmark",
            benchmark_agent(overall_score, progress_callback),
            timing,
        )

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
        pub = await _timed(
            "publish",
            publisher_agent(report, progress_callback),
            timing,
        )
        report["cited_url"] = pub.get("cited_url", "")
        report["publish_fallback"] = pub.get("fallback", False)

        # Persist report to ClickHouse before payment (best effort)
        await asyncio.to_thread(clickhouse_client.insert_report, report)

        # 5. Payment
        payment = await _timed(
            "payment",
            payment_agent(report_id, progress_callback),
            timing,
        )
        report["payment"] = payment

        # Sources count for the UI
        sources_count = sum(len(v) for v in research.values())
        report["sources_count"] = sources_count

        timing["total"] = round((time.perf_counter() - t_total_start) * 1000, 1)
        report["timing_ms"] = timing
        report["datadog_enabled"] = _LLMOBS_ENABLED
        report["datadog_url"] = config.datadog_llmobs_url() if _LLMOBS_ENABLED else ""

        _annotate(
            output_data=f"overall_score={overall_score}, sources={sources_count}, total_ms={timing['total']}",
            metadata={"verdict": report.get("verdict", "")[:200]},
            tags={"company": company_name, "overall_score": overall_score},
        )

    # Flush LLM Observability spans so they appear in Datadog immediately
    if _LLMOBS_ENABLED and LLMObs is not None:
        try:
            LLMObs.flush()
        except Exception:
            pass

    await _emit(progress_callback, "complete", f"DealAgent complete — all 5 agents fired in {timing['total']/1000:.1f}s.")
    return report
