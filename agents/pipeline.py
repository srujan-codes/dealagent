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
from core import credibility       # v2: source tiering + triangulation
from core import github_client     # v2: T1 engineering signals
from core import sec_edgar_client  # v2: T1 regulatory signals
from core import courtlistener_client  # v2: T1 legal-risk signals
from core import pr_detection      # v2: coordinated-PR detection + time bursts
from core import wikidata_client   # v2.1 A: T1 structured facts
from core import numerics          # v2.1 B: numerical claims + contradictions
from core import specialists       # v3-1: multi-agent decomposition
from core import critic            # v3-2: self-critique / evidence-quality review
from core import grounding         # v3-3: claim grounding / provenance
from core import trajectory        # v3-4: temporal momentum per dimension
from core import memory            # v3-5: semantic memory of past reports

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
            # integrations_enabled=False skips ddtrace's auto-patching of
            # openai/anthropic/etc. We intentionally do all instrumentation
            # manually via workflow/task/llm decorators below. This also
            # avoids a name collision: ddtrace's openai-agents auto-patch
            # tries to import from `agents.tracing`, which clashes with our
            # own top-level `agents/` package.
            LLMObs.enable(
                ml_app=config.DD_SERVICE,
                api_key=config.DD_API_KEY,
                site=config.DD_SITE,
                agentless_enabled=True,
                integrations_enabled=False,
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
    """v2: Nimble + GitHub + SEC + CourtListener in parallel.

    Nimble provides T2-T5 web breadth. GitHub/SEC/CourtListener provide
    T1 first-party signals that PR firms cannot fake. All four data sources
    fire concurrently; T1 results get merged into the appropriate Nimble
    category so the scoring agent sees one unified research dict.
    """
    await _emit(cb, "research", f"Research Agent firing — Nimble + GitHub + SEC + CourtListener in parallel on {company}...")
    with _task("dealagent.research"):
        # Fire all 5 source agents concurrently (v2 + v2.1 Wikidata)
        nimble_task = nimble_client.parallel_search(company)
        github_task = github_client.engineering_signals(company)
        sec_task = sec_edgar_client.regulatory_signals(company)
        court_task = courtlistener_client.legal_signals(company)
        wikidata_task = wikidata_client.company_facts(company)
        nimble_res, gh_res, sec_res, court_res, wd_res = await asyncio.gather(
            nimble_task, github_task, sec_task, court_task, wikidata_task,
            return_exceptions=True,
        )

        signals = nimble_res if isinstance(nimble_res, dict) else {
            "founder_signals": [], "traction_signals": [],
            "market_signals": [], "risk_signals": [],
        }

        # v2: annotate every Nimble signal with credibility tier (T1-T5)
        signals = credibility.annotate_research(signals)

        # Merge T1 signals into the appropriate Nimble categories
        # GitHub → team + traction (engineering velocity tells both stories)
        if isinstance(gh_res, list) and gh_res:
            signals.setdefault("founder_signals", []).extend(gh_res[:2])
            signals.setdefault("traction_signals", []).extend(gh_res[2:])
        # SEC EDGAR → traction (filing health) + risk (8-K disclosures)
        if isinstance(sec_res, list) and sec_res:
            signals.setdefault("traction_signals", []).extend(sec_res[:3])
            signals.setdefault("risk_signals", []).extend(sec_res[3:6])
        # CourtListener → all risk
        if isinstance(court_res, list) and court_res:
            signals.setdefault("risk_signals", []).extend(court_res[:5])
        # Wikidata structured facts → team (CEO, founder) + traction (employees, funding)
        wikidata_facts: Dict[str, Any] = {}
        if isinstance(wd_res, dict) and wd_res:
            wd_signals = wikidata_client.facts_as_signals(wd_res)
            signals.setdefault("founder_signals", []).extend(wd_signals)
            signals.setdefault("traction_signals", []).extend(wd_signals)
            wikidata_facts = wd_res

        # Tally
        total = sum(len(v) for v in signals.values())
        tier_dist = credibility.tier_distribution(signals)
        gh_count = len(gh_res) if isinstance(gh_res, list) else 0
        sec_count = len(sec_res) if isinstance(sec_res, list) else 0
        court_count = len(court_res) if isinstance(court_res, list) else 0
        _annotate(
            input_data=f"company={company}",
            output_data=(
                f"{total} signals · Nimble={sum(len(v) for v in (nimble_res.values() if isinstance(nimble_res, dict) else []))}, "
                f"GitHub={gh_count}, SEC={sec_count}, Court={court_count} · tiers: {tier_dist}"
            ),
            tags={"company": company, "agent": "research", "tools": "nimble+github+sec+courtlistener"},
        )
    wd_label = ("Wikidata ✓" if wikidata_facts else "Wikidata ∅")
    await _emit(
        cb, "research",
        f"Research Agent complete — {total} signals (Nimble + {gh_count} GitHub + {sec_count} SEC + {court_count} Court + {wd_label}) · tiers: {tier_dist}",
    )
    # Return signals + a tuple of extras so the orchestrator can surface them.
    return {"_signals": signals, "_wikidata_facts": wikidata_facts}


# -----------------------------------------------------------------------------
# Agent 2 — Scoring (Claude)
# -----------------------------------------------------------------------------
SCORING_PROMPT = """You are a senior VC analyst writing a detailed due diligence memo.

Company: {company}

Every research signal below is tagged with a credibility tier:
  T1 (regulatory / first-party):  sec.gov, courts, USPTO, GitHub, Wikidata. Facts with LEGAL accountability.
  T2 (established journalism):    WSJ, FT, Bloomberg, Reuters, NYT. Editorial standards.
  T3 (tech / industry press):     TechCrunch, The Information, Axios, Wired.
  T4 (analyst / aggregator):      LinkedIn, Wikipedia, SimilarWeb, Glassdoor.
  T5 (blog / social / press):     Medium, Substack, Twitter, PR Newswire. Often PR-driven.

Research signals (from live web search):
{research}

Scoring rules:
  1. Score each dimension 0-10 based on the EVIDENCE in the signals.
  2. For risk, higher score = MORE risk (worse for the investor).
  3. Strongly PREFER citing T1-T2 sources over T4-T5 sources.
  4. The "source" field MUST be a URL that appears literally in the signals above.
  5. The "source_tier" field MUST match the tier shown next to that source.
  6. If a dimension has only T4-T5 sources, that's a soft signal — be more conservative.
  7. The risk dimension must include a 5-part breakdown (regulatory, competitive,
     execution, financial, ip_legal). Each sub-score 0-10 with its own reasoning.
  8. The recommendation field must include: action (PASS / MONITOR / INVEST /
     STRONG_INVEST), confidence (LOW / MEDIUM / HIGH), and 2-3 upgrade + downgrade
     conditions.
  9. Return ONLY valid JSON. No prose. No markdown fences.

JSON schema:
{{
  "scores": {{
    "team":     {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url>", "source_tier": <1-5>}},
    "market":   {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url>", "source_tier": <1-5>}},
    "traction": {{"score": <0-10>, "reasoning": "<1 sentence>", "source": "<url>", "source_tier": <1-5>}},
    "risk":     {{
        "score": <0-10>, "reasoning": "<1 sentence aggregate>",
        "source": "<url>", "source_tier": <1-5>,
        "breakdown": {{
            "regulatory":  {{"score": <0-10>, "reasoning": "<1 sentence>"}},
            "competitive": {{"score": <0-10>, "reasoning": "<1 sentence>"}},
            "execution":   {{"score": <0-10>, "reasoning": "<1 sentence>"}},
            "financial":   {{"score": <0-10>, "reasoning": "<1 sentence>"}},
            "ip_legal":    {{"score": <0-10>, "reasoning": "<1 sentence>"}}
        }}
    }}
  }},
  "verdict": "<one sharp investment verdict sentence>",
  "key_insight": "<the single most important finding>",
  "recommendation": {{
    "action": "<PASS | MONITOR | INVEST | STRONG_INVEST>",
    "confidence": "<LOW | MEDIUM | HIGH>",
    "rationale": "<one sentence why>",
    "upgrade_conditions": ["<bullet 1>", "<bullet 2>"],
    "downgrade_conditions": ["<bullet 1>", "<bullet 2>"]
  }}
}}
"""


def _fallback_scores() -> Dict[str, Any]:
    risk_blank = {
        "score": 5, "reasoning": "Data unavailable.", "source": "", "source_tier": 4,
        "breakdown": {
            "regulatory":  {"score": 5, "reasoning": "Insufficient data."},
            "competitive": {"score": 5, "reasoning": "Insufficient data."},
            "execution":   {"score": 5, "reasoning": "Insufficient data."},
            "financial":   {"score": 5, "reasoning": "Insufficient data."},
            "ip_legal":    {"score": 5, "reasoning": "Insufficient data."},
        },
    }
    return {
        "scores": {
            "team":     {"score": 5, "reasoning": "Data unavailable.", "source": "", "source_tier": 4},
            "market":   {"score": 5, "reasoning": "Data unavailable.", "source": "", "source_tier": 4},
            "traction": {"score": 5, "reasoning": "Data unavailable.", "source": "", "source_tier": 4},
            "risk":     risk_blank,
        },
        "verdict": "Insufficient data — manual review recommended.",
        "key_insight": "Scoring agent could not reach Groq API.",
        "recommendation": {
            "action": "PASS",
            "confidence": "LOW",
            "rationale": "Cannot make a recommendation without scoring data.",
            "upgrade_conditions": ["Retry analysis after API reconnects"],
            "downgrade_conditions": [],
        },
    }


def _research_for_prompt(research: Dict[str, Any], cap: int = 6) -> str:
    """Render research with tier annotations for Llama to see."""
    chunks = []
    for cat, items in research.items():
        chunks.append(f"## {cat}")
        for it in items[:cap]:
            t = it.get("title", "")
            s = it.get("snippet", "")
            u = it.get("url", "")
            tier = it.get("tier", credibility.classify_source(u))
            chunks.append(f"- [T{tier}] {t} | {s} | {u}")
    return "\n".join(chunks)


def _enrich_scores_with_triangulation(
    scores: Dict[str, Any],
    research: Dict[str, List[Dict]],
    pr_shine: float = 0.0,
) -> Dict[str, Any]:
    """Compute per-dimension truth_score from raw_score, triangulation, and PR shine.

    truth = 5 + (raw - 5) * final_discount
    final_discount = base_discount * (1 - pr_shine * 0.4)
    base_discount = truth_discount(triangulation) ∈ [0.4, 1.0]

    Mutates and returns the scores dict, adding fields:
      raw_score, truth_score, triangulation, tier_weight (per dimension)
    """
    dim_to_cat = {
        "team":     "founder_signals",
        "market":   "market_signals",
        "traction": "traction_signals",
        "risk":     "risk_signals",
    }
    for dim, cat in dim_to_cat.items():
        s = scores.get(dim, {})
        raw = float(s.get("score", 0) or 0)
        triang = credibility.triangulation_score(research.get(cat, []))
        base_discount = credibility.truth_discount(triang)
        final_discount = pr_detection.apply_pr_shine_to_truth_discount(base_discount, pr_shine)
        truth = round(5 + (raw - 5) * final_discount, 2)
        s["raw_score"] = raw
        s["truth_score"] = truth
        s["triangulation"] = triang
        s["tier_weight"] = final_discount
        scores[dim] = s
    return scores


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


async def scoring_agent(company: str, research: Dict[str, Any], cb: ProgressCb) -> Dict[str, Any]:
    """v3-1: Now runs a multi-agent committee — 4 specialists + synthesizer."""
    await _emit(cb, "scoring", f"Scoring Committee firing — 4 specialists + synthesizer ({config.GROQ_MODEL})...")
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

            # v3-6 streaming: pipe synthesizer tokens to the progress callback
            async def _stream_chunk(delta: str) -> None:
                await _emit(cb, "reasoning_chunk", delta)

            # v3-1: Multi-agent committee — 4 specialists in parallel + streaming synthesizer
            parsed = await specialists.run_committee(
                client=client,
                model=config.GROQ_MODEL,
                company=company,
                research=research,
                llm_span_factory=_llm,
                annotate=_annotate,
                synth_stream_cb=_stream_chunk if cb else None,
            )

            # Sanity + tier annotation + defensive coercion
            scores = parsed.get("scores", {}) or {}
            for dim in ("team", "market", "traction", "risk"):
                if dim not in scores or not isinstance(scores[dim], dict):
                    scores[dim] = {"score": 5, "reasoning": "Missing.", "source": "", "source_tier": 4}
                if "source_tier" not in scores[dim]:
                    src = scores[dim].get("source", "")
                    scores[dim]["source_tier"] = credibility.classify_source(src)
                # Coerce risk.breakdown sub-scores that came back as bare numbers
                if dim == "risk":
                    bd = scores[dim].get("breakdown") or {}
                    for k, v in list(bd.items()):
                        if not isinstance(v, dict):
                            bd[k] = {
                                "score": v if isinstance(v, (int, float)) else 5,
                                "reasoning": "",
                            }
                    scores[dim]["breakdown"] = bd

            # v2: detect PR shine, then enrich with triangulation + PR discount
            pr_signal = pr_detection.detect_pr_shine(research)
            scores = _enrich_scores_with_triangulation(
                scores, research, pr_shine=pr_signal["pr_shine_score"],
            )
            parsed["scores"] = scores
            parsed["pr_shine"] = pr_signal

            spec_count = len(parsed.get("_specialist_outputs", {}))
            await _emit(
                cb, "scoring",
                f"Scoring Committee complete — {spec_count} specialists + synthesizer, 4 dimensions scored.",
            )
            return parsed
        except Exception as e:
            _annotate(
                input_data="(scoring committee failed)",
                output_data=f"ERROR: {type(e).__name__}: {str(e)[:200]}",
                tags={"company": company, "agent": "scoring", "error": True},
            )
            await _emit(cb, "scoring", f"Scoring Agent complete — fallback scores ({type(e).__name__}: {str(e)[:80]}).")
            # Enrich even fallback scores with triangulation so the UI doesn't break
            fb = _fallback_scores()
            pr_signal = pr_detection.detect_pr_shine(research)
            fb["scores"] = _enrich_scores_with_triangulation(
                fb["scores"], research, pr_shine=pr_signal["pr_shine_score"],
            )
            fb["pr_shine"] = pr_signal
            return fb


COMPARABLES_PROMPT = """You are a VC analyst. Suggest 2 OTHER companies that are
the most structurally similar to {company} — same vertical, similar
stage, comparable business model.

Research signals (lightweight context):
{research_brief}

Return ONLY valid JSON:
{{
  "peers": [
    {{"name": "<company name>", "why_similar": "<one sentence>"}},
    {{"name": "<company name>", "why_similar": "<one sentence>"}}
  ]
}}
"""


async def comparables_agent(
    company: str, research: Dict[str, List[Dict]], cb: ProgressCb
) -> Dict[str, Any]:
    """v3-7 (reduced) — Llama suggests 2 peer companies + why-similar."""
    if not config.have_groq():
        return {"peers": []}
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=config.GROQ_API_KEY)
        with _llm("dealagent.comparables", model_name=config.GROQ_MODEL, model_provider="groq"):
            completion = await client.chat.completions.create(
                model=config.GROQ_MODEL,
                max_tokens=300,
                temperature=0.4,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": COMPARABLES_PROMPT.format(
                    company=company,
                    research_brief=_research_for_prompt(research, cap=2)[:1500],
                )}],
            )
            text = completion.choices[0].message.content or "{}"
            try:
                parsed = json.loads(_strip_json_fence(text))
            except Exception:
                parsed = {"peers": []}
            _annotate(
                output_data=str(parsed.get("peers", []))[:200],
                tags={"company": company, "agent": "comparables"},
            )
            return parsed
    except Exception as e:
        return {"peers": [], "error": f"{type(e).__name__}: {str(e)[:120]}"}


ADVERSARIAL_PROMPT = """You are a skeptical contrarian VC analyst. Your job is to STEELMAN the case AGAINST investing in {company}.

Below are the current DD scores and reasoning. For each dimension scored >=7,
write the strongest plausible counter-argument USING ONLY EVIDENCE THAT COULD
BE PULLED FROM THE RESEARCH SIGNALS. Don't invent facts.

Current scores:
{score_summary}

Research signals (for grounding your counter-arguments):
{research_brief}

Return JSON only:
{{
  "counter_arguments": [
    {{
      "dimension": "<team|market|traction|risk>",
      "current_score": <number>,
      "counter": "<one strong sentence arguing the score should be lower>",
      "suggested_adjustment": <-2.0 to 0.0>
    }}
  ]
}}

Only include dimensions you have a substantive counter for. If you can't find a real counter-argument, leave the array empty.
"""


async def adversarial_agent(
    company: str, scoring: Dict[str, Any], research: Dict[str, List[Dict]], cb: ProgressCb
) -> Dict[str, Any]:
    """v2.1 G — generate steelmanned counter-arguments for high scores.

    Returns a dict with applied adjustments and counter-arguments. The
    final scoring result is mutated in place (truth_score downgraded for
    dimensions with compelling counters).
    """
    await _emit(cb, "adversarial", "Adversarial Agent firing — generating steelman counter-arguments...")
    out: Dict[str, Any] = {"counter_arguments": [], "applied_adjustments": []}

    if not config.have_groq():
        await _emit(cb, "adversarial", "Adversarial Agent complete — no Groq key, skipped.")
        return out

    # Build score summary (only dims >= 7 are eligible)
    scores = scoring.get("scores", {})
    eligible = []
    for dim in ("team", "market", "traction"):
        s = scores.get(dim, {})
        score = float(s.get("score", 0) or 0)
        if score >= 7:
            eligible.append(f"  {dim}: {score}/10 — {s.get('reasoning','')[:120]}")
    if not eligible:
        await _emit(cb, "adversarial", "Adversarial Agent complete — no high scores to challenge.")
        return out

    score_summary = "\n".join(eligible)
    research_brief = _research_for_prompt(research, cap=4)

    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=config.GROQ_API_KEY)
        with _llm("dealagent.adversarial", model_name=config.GROQ_MODEL, model_provider="groq"):
            completion = await client.chat.completions.create(
                model=config.GROQ_MODEL,
                max_tokens=800,
                temperature=0.5,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": ADVERSARIAL_PROMPT.format(
                        company=company,
                        score_summary=score_summary,
                        research_brief=research_brief,
                    ),
                }],
            )
            text = completion.choices[0].message.content or "{}"
            try:
                parsed = json.loads(_strip_json_fence(text))
            except Exception:
                parsed = {}
            counters = parsed.get("counter_arguments", [])
            if isinstance(counters, list):
                out["counter_arguments"] = counters[:4]

            # Apply nudges to truth_score for dimensions with substantive counters
            for c in out["counter_arguments"]:
                dim = c.get("dimension")
                adj = float(c.get("suggested_adjustment", 0) or 0)
                counter_text = c.get("counter", "")
                if dim in scores and len(counter_text) > 40 and -2.0 <= adj <= 0:
                    truth = scores[dim].get("truth_score")
                    if isinstance(truth, (int, float)):
                        new_truth = round(max(0.0, truth + adj), 2)
                        scores[dim]["truth_score"] = new_truth
                        scores[dim]["adversarial_adjustment"] = adj
                        out["applied_adjustments"].append({
                            "dimension": dim,
                            "before": truth,
                            "after": new_truth,
                            "delta": adj,
                            "counter": counter_text[:160],
                        })
            _annotate(
                input_data=score_summary[:400],
                output_data=f"{len(out['counter_arguments'])} counter-args, {len(out['applied_adjustments'])} applied",
                tags={"company": company, "agent": "adversarial"},
            )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"

    await _emit(
        cb, "adversarial",
        f"Adversarial Agent complete — {len(out['counter_arguments'])} counter-args, {len(out['applied_adjustments'])} truth-score adjustments.",
    )
    return out


def _overall(scores: Dict[str, Any], use_truth: bool = False) -> float:
    """Compute overall score. use_truth=False → raw (back-compat). use_truth=True → truth."""
    key = "truth_score" if use_truth else "score"
    team     = float(scores.get("team",     {}).get(key, 0) or 0)
    market   = float(scores.get("market",   {}).get(key, 0) or 0)
    traction = float(scores.get("traction", {}).get(key, 0) or 0)
    risk     = float(scores.get("risk",     {}).get(key, 0) or 0)
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
            metadata={
                "prompt_id": out.get("prompt_id", ""),
                "publish_status": out.get("publish_status", ""),
                "publish_message": out.get("publish_message", ""),
            },
            tags={"agent": "publish", "tool": "senso", "fallback": out.get("fallback", False)},
        )
    if out.get("prompt_id") and out.get("fallback"):
        msg = f"Publisher Agent complete — Senso prompt {out['prompt_id'][:8]}… created ({out.get('publish_message','')})"
    else:
        msg = f"Publisher Agent complete — live at {out['cited_url']}"
    await _emit(cb, "publish", msg)
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

        # 1. Research (returns dict with _signals + _wikidata_facts)
        research_bundle = await _timed(
            "research",
            research_agent(company_name, progress_callback),
            timing,
        )
        research = research_bundle.get("_signals", {}) if isinstance(research_bundle, dict) else {}
        wikidata_facts = (
            research_bundle.get("_wikidata_facts", {}) if isinstance(research_bundle, dict) else {}
        )

        # 2. Scoring
        scoring = await _timed(
            "scoring",
            scoring_agent(company_name, research, progress_callback),
            timing,
        )
        scores = scoring.get("scores", {})

        # v3-3: Ground every score in its actual source snippet
        scoring = grounding.ground_scores(scoring, research)

        # v3-2: Critic review — flag weak evidence, downgrade confidence
        critic_review = critic.review(scoring, research)
        if scoring.get("recommendation"):
            scoring["recommendation"] = critic.maybe_downgrade_confidence(
                scoring["recommendation"], critic_review,
            )

        # v2.1 G: adversarial counter-arguments may downgrade truth scores
        adversarial = await _timed(
            "adversarial",
            adversarial_agent(company_name, scoring, research, progress_callback),
            timing,
        )

        # v3-7 (reduced): Llama-suggested peer companies
        comparables = await _timed(
            "comparables",
            comparables_agent(company_name, research, progress_callback),
            timing,
        )

        overall_raw = _overall(scores, use_truth=False)
        overall_truth = _overall(scores, use_truth=True)
        # Use truth score for benchmarking (it's the more honest number)
        overall_score = overall_truth

        # 3. Benchmark
        benchmark = await _timed(
            "benchmark",
            benchmark_agent(overall_score, progress_callback),
            timing,
        )

        # v2: compute summary stats for triangulation + tier distribution
        triangulation_per_dim = credibility.triangulation_per_dimension(research)
        tier_dist = credibility.tier_distribution(research)
        avg_triangulation = round(
            sum(triangulation_per_dim.values()) / max(1, len(triangulation_per_dim)), 3
        )

        # v2.1 B: extract numerical claims + detect contradictions
        num_claims, num_contras, num_summary = numerics.extract_all_numerics(research)
        # v2.1 C: time-burst detection
        time_burst = pr_detection.detect_time_burst(research)
        # v3-4: per-dimension temporal trajectory
        trajectory_per_dim = trajectory.analyze(research)
        # v3-5: semantic memory — find similar past deals
        similar_past = await asyncio.to_thread(
            memory.find_similar_past_reports,
            company_name,
            scoring.get("verdict", ""),
            scoring.get("key_insight", ""),
            3,
        )

        # Build the report so far (publisher needs it)
        report: Dict[str, Any] = {
            "report_id": report_id,
            "company_name": company_name,
            "scores": scores,
            "overall_score": overall_score,            # back-compat: == overall_truth
            "overall_raw_score": overall_raw,          # v2
            "overall_truth_score": overall_truth,      # v2
            "triangulation_per_dimension": triangulation_per_dim,
            "avg_triangulation": avg_triangulation,
            "tier_distribution": tier_dist,
            "pr_shine": scoring.get("pr_shine", {}),   # v2 phase 5
            "verdict": scoring.get("verdict", ""),
            "key_insight": scoring.get("key_insight", ""),
            "recommendation": scoring.get("recommendation", {}),  # v2.1 E
            "benchmark": benchmark,
            "research": research,
            # v2.1 surface fields
            "wikidata_facts": wikidata_facts,        # v2.1 A
            "numerical_claims": num_claims,          # v2.1 B
            "numerical_contradictions": num_contras, # v2.1 B
            "numerical_summary": num_summary,        # v2.1 B
            "time_burst": time_burst,                # v2.1 C
            "adversarial": adversarial,              # v2.1 G
            "critic_review": critic_review,          # v3-2
            "specialist_outputs": scoring.get("_specialist_outputs", {}),  # v3-1
            "trajectory": trajectory_per_dim,        # v3-4
            "provenance_summary": scoring.get("provenance_summary", {}),  # v3-3
            "similar_past_reports": similar_past,    # v3-5
            "comparables": comparables,              # v3-7
        }

        # 4. Publish
        pub = await _timed(
            "publish",
            publisher_agent(report, progress_callback),
            timing,
        )
        report["cited_url"] = pub.get("cited_url", "")
        report["publish_fallback"] = pub.get("fallback", False)
        report["senso_prompt_id"] = pub.get("prompt_id", None)
        report["senso_publish_status"] = pub.get("publish_status", "")
        report["senso_publish_message"] = pub.get("publish_message", "")

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
