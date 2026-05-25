"""Multi-agent decomposition for DealAgent v3.

Replaces the single 'scoring_agent' Llama call with FOUR specialist analysts
running in parallel, each producing only their assigned dimensions, plus a
Synthesizer agent that combines them into the final scores dict + verdict +
recommendation.

Specialists:
  - FinancialAnalyst  → traction + risk.financial + risk.regulatory
  - MarketAnalyst     → market + risk.competitive
  - TechnicalAnalyst  → team + risk.execution
  - LegalAnalyst      → risk.ip_legal + cross-validates risk.regulatory

Each gets the same research dict but is instructed to FOCUS ON its domain.
Each returns partial scoring JSON. The Synthesizer merges them and writes the
final verdict, key_insight, and investment recommendation.

This is more legible than one mega-prompt: each specialist's output is
inspectable in Datadog as its own LLM span, with focused tokens and lower
hallucination rate per domain.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


def _research_for_prompt(research: Dict[str, List[Dict]], cap: int = 6) -> str:
    chunks: List[str] = []
    for cat, items in research.items():
        if not items:
            continue
        chunks.append(f"## {cat}")
        for it in items[:cap]:
            t = it.get("title", "") or ""
            s = it.get("snippet", "") or ""
            u = it.get("url", "") or ""
            tier = it.get("tier", "?")
            chunks.append(f"- [T{tier}] {t} | {s} | {u}")
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# Specialist prompts
# ---------------------------------------------------------------------------

SPECIALIST_BASE = """You are a senior VC analyst with deep expertise in {expertise}.

Company: {company}

You will see ALL research signals tagged by tier. FOCUS only on signals
relevant to your specialty. Ignore the rest.

Tiers (prefer higher):
  T1 regulatory/first-party · T2 established press · T3 tech press
  T4 analyst/aggregator · T5 blog/social/PR

Research signals:
{research}

Score ONLY the dimensions assigned to you. For each, give:
  - score: 0-10 (for risk sub-dimensions, higher = MORE risk)
  - reasoning: ONE sentence
  - source: a URL literally present in the signals above
  - source_tier: 1-5 matching the source

Return ONLY valid JSON in this exact shape:
{schema}
"""


FINANCIAL_SCHEMA = """{{
  "traction":         {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}},
  "risk_financial":   {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}},
  "risk_regulatory":  {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}}
}}"""

MARKET_SCHEMA = """{{
  "market":           {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}},
  "risk_competitive": {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}}
}}"""

TECHNICAL_SCHEMA = """{{
  "team":             {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}},
  "risk_execution":   {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}}
}}"""

LEGAL_SCHEMA = """{{
  "risk_ip_legal":    {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}},
  "risk_regulatory_legal_view": {{"score": <0-10>, "reasoning": "<sentence>", "source": "<url>", "source_tier": <1-5>}}
}}"""


SPECIALISTS = [
    {
        "key": "financial",
        "expertise": "financial analysis, revenue, burn, funding, SEC filings",
        "schema": FINANCIAL_SCHEMA,
        "model_name": "dealagent.specialists.financial",
    },
    {
        "key": "market",
        "expertise": "market sizing, competitive dynamics, market share",
        "schema": MARKET_SCHEMA,
        "model_name": "dealagent.specialists.market",
    },
    {
        "key": "technical",
        "expertise": "engineering velocity, team quality, GitHub activity, founder backgrounds",
        "schema": TECHNICAL_SCHEMA,
        "model_name": "dealagent.specialists.technical",
    },
    {
        "key": "legal",
        "expertise": "IP, patents, court records, regulatory liability",
        "schema": LEGAL_SCHEMA,
        "model_name": "dealagent.specialists.legal",
    },
]


def _empty_specialist_output(spec_key: str) -> Dict[str, Any]:
    """Fallback shapes when a specialist call fails."""
    blank = {"score": 5, "reasoning": "Specialist unavailable.", "source": "", "source_tier": 4}
    if spec_key == "financial":
        return {"traction": blank, "risk_financial": blank, "risk_regulatory": blank}
    if spec_key == "market":
        return {"market": blank, "risk_competitive": blank}
    if spec_key == "technical":
        return {"team": blank, "risk_execution": blank}
    if spec_key == "legal":
        return {"risk_ip_legal": blank, "risk_regulatory_legal_view": blank}
    return {}


async def run_specialist(
    client: Any,
    model: str,
    spec: Dict[str, Any],
    company: str,
    research: Dict[str, List[Dict]],
    llm_span_factory: Callable,
    annotate: Callable,
) -> Dict[str, Any]:
    """Run a single specialist agent. Returns its partial scoring dict.

    llm_span_factory: callable(name, model_name, provider) -> ctx manager
    annotate: callable(**kw) -> None  (LLMObs annotate or no-op)
    """
    prompt = SPECIALIST_BASE.format(
        expertise=spec["expertise"],
        company=company,
        research=_research_for_prompt(research),
        schema=spec["schema"],
    )
    try:
        with llm_span_factory(spec["model_name"], model_name=model, model_provider="groq"):
            completion = await client.chat.completions.create(
                model=model,
                max_tokens=800,
                temperature=0.25,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or "{}"
            text_clean = _strip_fence(text)
            parsed = json.loads(text_clean)
            usage = getattr(completion, "usage", None)
            metrics = {}
            if usage is not None:
                metrics = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            annotate(
                input_data=[{"role": "user", "content": prompt}],
                output_data=[{"role": "assistant", "content": text}],
                metrics=metrics,
                tags={"company": company, "specialist": spec["key"]},
            )
            return parsed
    except Exception as e:
        annotate(
            output_data=f"ERROR: {type(e).__name__}: {str(e)[:160]}",
            tags={"company": company, "specialist": spec["key"], "error": True},
        )
        return _empty_specialist_output(spec["key"])


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

SYNTHESIZER_PROMPT = """You are the lead VC partner reviewing 4 specialist analyst reports on {company}.

Each specialist scored their domain independently. Your job is to:
  1. Combine their scores into the canonical 4-dimension structure (team, market, traction, risk).
  2. Write a unified verdict + key insight.
  3. Produce an investment recommendation matrix.

Risk is an aggregate. Compute it as the AVERAGE of the 5 risk sub-scores from the specialists
(financial, regulatory, competitive, execution, ip_legal). The breakdown comes directly from
the specialists.

FINANCIAL ANALYST said:
{financial_json}

MARKET ANALYST said:
{market_json}

TECHNICAL ANALYST said:
{technical_json}

LEGAL ANALYST said:
{legal_json}

Return ONLY valid JSON:
{{
  "scores": {{
    "team":     <copy from technical.team verbatim>,
    "market":   <copy from market.market verbatim>,
    "traction": <copy from financial.traction verbatim>,
    "risk": {{
      "score": <average of the 5 sub-scores, rounded to nearest int>,
      "reasoning": "<one sentence synthesizing risk across all 5 sub-dims>",
      "source": "<best supporting URL from the specialists' sources>",
      "source_tier": <tier of that source 1-5>,
      "breakdown": {{
        "regulatory":  <use financial.risk_regulatory>,
        "competitive": <use market.risk_competitive>,
        "execution":   <use technical.risk_execution>,
        "financial":   <use financial.risk_financial>,
        "ip_legal":    <use legal.risk_ip_legal>
      }}
    }}
  }},
  "verdict": "<one sharp sentence weighing all 4 dimensions>",
  "key_insight": "<the single most important finding from the 4 specialist reports>",
  "recommendation": {{
    "action": "<PASS | MONITOR | INVEST | STRONG_INVEST>",
    "confidence": "<LOW | MEDIUM | HIGH>",
    "rationale": "<one sentence why>",
    "upgrade_conditions": ["<bullet>", "<bullet>"],
    "downgrade_conditions": ["<bullet>", "<bullet>"]
  }}
}}
"""


async def run_synthesizer(
    client: Any,
    model: str,
    company: str,
    specialist_results: Dict[str, Dict[str, Any]],
    llm_span_factory: Callable,
    annotate: Callable,
) -> Dict[str, Any]:
    """Combine 4 specialist outputs into final scoring + recommendation."""
    prompt = SYNTHESIZER_PROMPT.format(
        company=company,
        financial_json=json.dumps(specialist_results.get("financial", {}), indent=2),
        market_json=json.dumps(specialist_results.get("market", {}), indent=2),
        technical_json=json.dumps(specialist_results.get("technical", {}), indent=2),
        legal_json=json.dumps(specialist_results.get("legal", {}), indent=2),
    )
    try:
        with llm_span_factory("dealagent.synthesizer", model_name=model, model_provider="groq"):
            completion = await client.chat.completions.create(
                model=model,
                max_tokens=1500,
                temperature=0.25,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            text = completion.choices[0].message.content or "{}"
            parsed = json.loads(_strip_fence(text))
            usage = getattr(completion, "usage", None)
            metrics = {}
            if usage is not None:
                metrics = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            annotate(
                input_data=[{"role": "user", "content": prompt}],
                output_data=[{"role": "assistant", "content": text}],
                metrics=metrics,
                tags={"company": company, "agent": "synthesizer"},
            )
            return parsed
    except Exception as e:
        annotate(
            output_data=f"ERROR: {type(e).__name__}: {str(e)[:160]}",
            tags={"company": company, "agent": "synthesizer", "error": True},
        )
        return {}


# ---------------------------------------------------------------------------
# Public API: orchestrate all 4 specialists + synthesizer
# ---------------------------------------------------------------------------

async def run_committee(
    client: Any,
    model: str,
    company: str,
    research: Dict[str, List[Dict]],
    llm_span_factory: Callable,
    annotate: Callable,
) -> Dict[str, Any]:
    """Run all 4 specialists in parallel, then synthesizer. Returns final scoring dict.

    Output shape:
      {
        "scores": {team, market, traction, risk: {score, breakdown: {...}}},
        "verdict": str,
        "key_insight": str,
        "recommendation": {action, confidence, rationale, upgrade_conditions, downgrade_conditions},
        "_specialist_outputs": {financial, market, technical, legal},  # for diagnostics
      }
    """
    # Fire all 4 specialists in parallel
    tasks = [
        run_specialist(client, model, spec, company, research, llm_span_factory, annotate)
        for spec in SPECIALISTS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    specialist_outputs = {
        SPECIALISTS[i]["key"]: results[i] for i in range(len(SPECIALISTS))
    }

    # Run synthesizer
    synth = await run_synthesizer(
        client, model, company, specialist_outputs, llm_span_factory, annotate,
    )

    # Fall back to manual merge if synthesizer failed
    if not synth or "scores" not in synth:
        synth = _manual_merge(specialist_outputs)

    synth["_specialist_outputs"] = specialist_outputs
    return synth


def _manual_merge(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback merge if the synthesizer LLM call fails."""
    fin = specs.get("financial", {})
    mkt = specs.get("market", {})
    tech = specs.get("technical", {})
    legal = specs.get("legal", {})

    def _score(d, k):
        return float((d.get(k) or {}).get("score", 5) or 5)

    risk_sub = {
        "regulatory":  fin.get("risk_regulatory")  or {"score": 5, "reasoning": ""},
        "competitive": mkt.get("risk_competitive") or {"score": 5, "reasoning": ""},
        "execution":   tech.get("risk_execution")  or {"score": 5, "reasoning": ""},
        "financial":   fin.get("risk_financial")   or {"score": 5, "reasoning": ""},
        "ip_legal":    legal.get("risk_ip_legal")  or {"score": 5, "reasoning": ""},
    }
    avg_risk = round(
        (_score(fin, "risk_regulatory") + _score(mkt, "risk_competitive")
         + _score(tech, "risk_execution") + _score(fin, "risk_financial")
         + _score(legal, "risk_ip_legal")) / 5.0
    )

    return {
        "scores": {
            "team":     tech.get("team", {"score": 5, "reasoning": "", "source": "", "source_tier": 4}),
            "market":   mkt.get("market", {"score": 5, "reasoning": "", "source": "", "source_tier": 4}),
            "traction": fin.get("traction", {"score": 5, "reasoning": "", "source": "", "source_tier": 4}),
            "risk": {
                "score": avg_risk,
                "reasoning": "Aggregate of 5 sub-dimension risks from specialist analysts.",
                "source": "",
                "source_tier": 4,
                "breakdown": risk_sub,
            },
        },
        "verdict": "Multi-agent committee output (synthesizer fallback).",
        "key_insight": "Synthesizer failed — manually merged from specialist outputs.",
        "recommendation": {
            "action": "MONITOR",
            "confidence": "LOW",
            "rationale": "Synthesizer agent unavailable; using mechanical merge.",
            "upgrade_conditions": [],
            "downgrade_conditions": [],
        },
    }
