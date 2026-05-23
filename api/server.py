"""FastAPI server — serves UI, runs pipeline, streams progress via SSE."""
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.pipeline import run_dealagent


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"

app = FastAPI(title="DealAgent", version="1.0.0")


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
