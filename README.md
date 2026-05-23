# DealAgent

**Autonomous due diligence agent.** Type a company → 5 agents fire → cited report published to the web → agent pays agent via x402. 20 hours of VC due diligence compressed into 4 minutes.

Built at NYC AI Agents Hackathon 2026.

## The 5 Agents

| # | Agent | Tool | What it does |
|---|-------|------|--------------|
| 1 | Research  | **Nimble**      | 4 parallel live web searches (founder, traction, market, risk) |
| 2 | Scoring   | **Claude**      | Structured JSON scores w/ source citations |
| 3 | Benchmark | **ClickHouse**  | Compares to historical deals |
| 4 | Publisher | **Senso**       | Publishes cited report to `cited.md` |
| 5 | Payment   | **x402 / CDP**  | Pays 1.00 USDC on Base, agent-to-agent |

Full **Datadog LLM Observability** trace wraps every agent call.

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. configure
cp .env.example .env
# fill in API keys

# 3. run
python main.py
# open http://localhost:8000
```

## Architecture

```
User → FastAPI (SSE) → Orchestrator
                          ├── Research   (Nimble, 4× parallel)
                          ├── Scoring    (Claude — claude-sonnet-4-20250514)
                          ├── Benchmark  (ClickHouse Cloud)
                          ├── Publisher  (Senso → cited.md)
                          └── Payment    (x402 / Base / USDC)
                      ↑
                  Datadog span wraps the whole pipeline
                  and each individual agent
```

Every agent is **non-fatal**: if Nimble, Claude, ClickHouse, or Senso is unreachable, the pipeline degrades gracefully and the demo never breaks.

## API

```bash
# Synchronous report
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Anthropic"}'

# Live SSE progress stream
curl "http://localhost:8000/api/analyze/stream?company=Anthropic"
```

## Files

```
dealagent/
├── main.py
├── requirements.txt
├── .env.example
├── agents/pipeline.py        # all 5 agents + orchestrator
├── api/server.py             # FastAPI + SSE
├── core/
│   ├── config.py
│   ├── nimble_client.py      # 4× parallel via asyncio.gather
│   ├── clickhouse_client.py  # benchmark + payment log
│   ├── senso_client.py       # cited.md publisher
│   └── x402_client.py        # agent-to-agent USDC on Base
└── frontend/index.html       # single-file dark terminal UI
```

## Why this matters

> "DealAgent doesn't help analysts work faster. It works when analysts can't.
> $1. 4 minutes. Zero humans."
