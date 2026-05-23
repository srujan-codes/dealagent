"""Centralized config loader. Reads .env once at import time."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Groq — fast open-source LLM inference (Llama 3.3 70B on LPU hardware)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Nimble
NIMBLE_API_KEY = os.getenv("NIMBLE_API_KEY", "")
NIMBLE_URL = "https://nimble-retriever.webit.live/search"

# ClickHouse
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8443"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "dealagent")

# Senso — base + endpoints are owned by core/senso_client.py
SENSO_API_KEY = os.getenv("SENSO_API_KEY", "")

# x402 / CDP
CDP_API_KEY_NAME = os.getenv("CDP_API_KEY_NAME", "")
CDP_API_KEY_PRIVATE_KEY = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")
AGENT_WALLET_ADDRESS = os.getenv("AGENT_WALLET_ADDRESS", "0xAutonomousResearchAgent")

# Datadog
DD_API_KEY = os.getenv("DD_API_KEY", "")
DD_SITE = os.getenv("DD_SITE", "datadoghq.com")
DD_SERVICE = os.getenv("DD_SERVICE", "dealagent")
DD_ENV = os.getenv("DD_ENV", "hackathon")

# Pipeline
HTTP_TIMEOUT = 30.0


def have_groq() -> bool:
    return bool(GROQ_API_KEY)


def have_nimble() -> bool:
    return bool(NIMBLE_API_KEY)


def have_clickhouse() -> bool:
    return bool(CLICKHOUSE_HOST and CLICKHOUSE_PASSWORD)


def have_senso() -> bool:
    return bool(SENSO_API_KEY)
