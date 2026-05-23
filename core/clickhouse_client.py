"""ClickHouse client — historical benchmarking + payment log."""
from typing import Dict, Any, Optional

from core import config

_client = None
_init_attempted = False


def _get_client():
    """Lazy-initialize ClickHouse connection. Returns None if unavailable."""
    global _client, _init_attempted
    if _client is not None:
        return _client
    if _init_attempted:
        return None
    _init_attempted = True

    if not config.have_clickhouse():
        return None

    try:
        import clickhouse_connect
        _client = clickhouse_connect.get_client(
            host=config.CLICKHOUSE_HOST,
            port=config.CLICKHOUSE_PORT,
            username=config.CLICKHOUSE_USER,
            password=config.CLICKHOUSE_PASSWORD,
            database=config.CLICKHOUSE_DATABASE,
            secure=True,
        )
        _ensure_tables(_client)
        return _client
    except Exception:
        _client = None
        return None


def _ensure_tables(client) -> None:
    """Create tables if they don't exist."""
    client.command("""
        CREATE TABLE IF NOT EXISTS dd_reports (
            company_name String,
            report_id String,
            team_score Float32,
            market_score Float32,
            traction_score Float32,
            risk_score Float32,
            overall_score Float32,
            verdict String,
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (created_at, company_name)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS payment_log (
            report_id String,
            amount_usd Float32,
            tx_hash String,
            payer String,
            paid_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY paid_at
    """)


def benchmark(overall_score: float) -> Dict[str, Any]:
    """Return historical comparison stats. Never raises."""
    client = _get_client()
    if client is None:
        return {"total_in_db": 0, "note": "DB unavailable"}

    try:
        row = client.query(
            "SELECT count(), avg(overall_score), countIf(overall_score >= 7.0) FROM dd_reports"
        ).result_rows
        if not row:
            return {"total_in_db": 0, "note": "first report in database"}
        count, avg, top = row[0]
        if count == 0:
            return {"total_in_db": 0, "note": "first report in database"}
        avg = float(avg or 0)
        return {
            "total_in_db": int(count),
            "avg_score_in_db": round(avg, 2),
            "top_decile_count": int(top),
            "this_company_vs_avg": "above" if overall_score >= avg else "below",
            "delta": round(overall_score - avg, 2),
        }
    except Exception as e:
        return {"total_in_db": 0, "note": f"DB error: {type(e).__name__}"}


def insert_report(report: Dict[str, Any]) -> bool:
    """Persist the report. Returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        scores = report.get("scores", {})
        client.insert(
            "dd_reports",
            [[
                report.get("company_name", ""),
                report.get("report_id", ""),
                float(scores.get("team", {}).get("score", 0)),
                float(scores.get("market", {}).get("score", 0)),
                float(scores.get("traction", {}).get("score", 0)),
                float(scores.get("risk", {}).get("score", 0)),
                float(report.get("overall_score", 0)),
                report.get("verdict", "")[:500],
            ]],
            column_names=[
                "company_name", "report_id",
                "team_score", "market_score", "traction_score", "risk_score",
                "overall_score", "verdict",
            ],
        )
        return True
    except Exception:
        return False


def insert_payment(report_id: str, amount_usd: float, tx_hash: str, payer: str) -> bool:
    """Log payment. Returns True on success."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.insert(
            "payment_log",
            [[report_id, float(amount_usd), tx_hash, payer]],
            column_names=["report_id", "amount_usd", "tx_hash", "payer"],
        )
        return True
    except Exception:
        return False
