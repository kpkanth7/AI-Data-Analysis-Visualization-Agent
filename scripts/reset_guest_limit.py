"""
Developer utility: reset guest daily limits in the guest_usage table.

Usage:
    # Reset ALL guest limits for today (fresh slate for all IPs)
    python scripts/reset_guest_limit.py --all

    # Reset a specific IP hash
    python scripts/reset_guest_limit.py --hash <ip_hash>

    # Show current usage table
    python scripts/reset_guest_limit.py --show
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.postgres import get_connection
from sqlalchemy import text


def show_usage():
    with get_connection() as conn:
        rows = conn.execute(text(
            "SELECT ip_hash, usage_date, queries, uploads, upload_bytes "
            "FROM guest_usage ORDER BY usage_date DESC, queries DESC"
        )).fetchall()
    if not rows:
        print("guest_usage table is empty.")
        return
    print(f"{'ip_hash':<34} {'date':<12} {'queries':>8} {'uploads':>8} {'bytes':>12}")
    print("-" * 78)
    for r in rows:
        print(f"{r[0]:<34} {str(r[1]):<12} {r[2]:>8} {r[3]:>8} {r[4]:>12,}")


def reset_all():
    with get_connection() as conn:
        conn.execute(text("DELETE FROM guest_usage WHERE usage_date = CURRENT_DATE"))
        conn.commit()
    print("Cleared all guest usage for today.")


def reset_hash(ip_hash: str):
    with get_connection() as conn:
        result = conn.execute(text(
            "DELETE FROM guest_usage WHERE ip_hash = :h AND usage_date = CURRENT_DATE"
        ), {"h": ip_hash})
        conn.commit()
        count = result.rowcount
    if count:
        print(f"Reset usage for ip_hash={ip_hash}")
    else:
        print(f"No record found for ip_hash={ip_hash} today.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--show":
        show_usage()
    elif args[0] == "--all":
        reset_all()
        show_usage()
    elif args[0] == "--hash" and len(args) == 2:
        reset_hash(args[1])
        show_usage()
    else:
        print(__doc__)
