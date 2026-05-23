"""One-time setup: create CDP server wallets + fund via faucet.

Run once after CDP_API_KEY_NAME and CDP_API_KEY_PRIVATE_KEY are in .env.
Writes the dealagent wallet address + treasury address back into .env
so the pipeline can use them.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from cdp import CdpClient


NETWORK = "base-sepolia"


async def main() -> int:
    api_key_id = os.getenv("CDP_API_KEY_NAME", "")
    api_key_secret = os.getenv("CDP_API_KEY_PRIVATE_KEY", "")
    wallet_secret = os.getenv("CDP_WALLET_SECRET", "")
    if not api_key_id or not api_key_secret:
        print("✗ Missing CDP_API_KEY_NAME or CDP_API_KEY_PRIVATE_KEY in .env")
        return 1
    if not wallet_secret:
        print("✗ Missing CDP_WALLET_SECRET in .env. Generate one in CDP portal → Access → Wallet Secret")
        return 1

    async with CdpClient(
        api_key_id=api_key_id,
        api_key_secret=api_key_secret,
        wallet_secret=wallet_secret,
    ) as client:
        # 1. Create/get the agent's spending wallet
        sender = await client.evm.get_or_create_account(name="dealagent")
        print(f"✓ Sender wallet:  {sender.address}")

        # 2. Create/get the treasury (recipient) wallet
        treasury = await client.evm.get_or_create_account(name="dealagent-treasury")
        print(f"✓ Treasury wallet: {treasury.address}")

        # 3. Fund the sender via CDP's built-in faucet (Base Sepolia USDC)
        try:
            tx = await sender.request_faucet(network=NETWORK, token="usdc")
            print(f"✓ Faucet drop:    tx={tx} (waiting ~10s for confirm)")
            await asyncio.sleep(10)
        except Exception as e:
            print(f"⚠ Faucet: {type(e).__name__}: {e}")

        # 4. Sanity check the balance
        try:
            balances = await sender.list_token_balances(network=NETWORK)
            print(f"✓ Balances:       {balances}")
        except Exception as e:
            print(f"⚠ Balance check failed: {e}")

    # 5. Write addresses back into .env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    text = env_path.read_text()

    def upsert(key: str, value: str, t: str) -> str:
        pattern = rf"^{re.escape(key)}=.*$"
        if re.search(pattern, t, flags=re.M):
            return re.sub(pattern, f"{key}={value}", t, flags=re.M)
        return t.rstrip() + f"\n{key}={value}\n"

    text = upsert("AGENT_WALLET_ADDRESS", sender.address, text)
    text = upsert("TREASURY_WALLET_ADDRESS", treasury.address, text)
    text = upsert("CDP_NETWORK", NETWORK, text)
    env_path.write_text(text)
    print(f"\n✓ .env updated with AGENT_WALLET_ADDRESS + TREASURY_WALLET_ADDRESS")
    print(f"\nBasescan Sepolia: https://sepolia.basescan.org/address/{sender.address}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
