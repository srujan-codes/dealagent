"""x402 / CDP simulator — agent-to-agent USDC micropayment on Base."""
import uuid
from typing import Dict, Any

from core import config


def _tx_hash() -> str:
    """0x + 64 hex chars."""
    raw = (uuid.uuid4().hex + uuid.uuid4().hex)[:64]
    return "0x" + raw


def pay(report_id: str, recipient: str = "0xDealAgentTreasury") -> Dict[str, Any]:
    """Simulate an x402 USDC payment. Always succeeds (returns simulated tx)."""
    tx_hash = _tx_hash()
    payer = config.AGENT_WALLET_ADDRESS or "0xAutonomousResearchAgent"
    return {
        "verified": True,
        "tx_hash": tx_hash,
        "amount_usd": 1.00,
        "network": "base",
        "asset": "USDC",
        "payer": payer,
        "recipient": recipient,
        "report_id": report_id,
        "message": f"Agent {payer[:10]}… paid 1.00 USDC on Base — no human in the loop.",
    }
