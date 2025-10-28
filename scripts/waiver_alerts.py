# scripts/waiver_alerts.py
from datetime import datetime
import requests
from zoneinfo import ZoneInfo
import config
from scripts.utils import get_next_transaction_deadline
TZ = ZoneInfo("America/New_York")

def main():
    if not config.DISCORD_WEBHOOK_URL:
        print('Missing Discord Webhook URL')
        return
    deadline_et, gw = get_next_transaction_deadline(config.TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF, config.CURRENT_GAMEWEEK)
    hours_left = (deadline_et - datetime.now(TZ)).total_seconds() / 3600

    # Fire at ~24h / 6h / 1h (tolerance ±20 min)
    for target in (24, 6, 1):
        if abs(hours_left - target) <= 20/60:
            ts = deadline_et.strftime("%a %b %d • %I:%M %p %Z")
            msg = f"🔔 FPL Draft transactions for **GW {gw}** are due in ~**{target}h** (deadline **{ts}**)."
            # Example mention (optional): "<@123456789012345678>" for a user or "<@&ROLE_ID>" for a role
            requests.post(config.DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
            break
        else:
            print("Outside target reminder windows")

if __name__ == "__main__":
    main()