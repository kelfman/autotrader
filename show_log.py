"""
show_log.py — pretty-print experiment_log.jsonl.

Usage:
    python show_log.py                  # show all experiments
    python show_log.py --last 20        # show the last 20
    python show_log.py --accepted-only  # show only accepted experiments
    python show_log.py --detail 5       # full detail for iteration 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOG_PATH = Path(__file__).parent / "experiment_log.jsonl"


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def print_table(entries: list[dict]) -> None:
    if not entries:
        print("No experiments found.")
        return

    header = (
        f"{'#':>4}  {'Time':>8}  {'Type':<12}  "
        f"{'Before':>7}  {'After':>7}  {'Δ':>7}  {'OK':>3}  Rationale"
    )
    sep = "─" * 100
    print(sep)
    print(header)
    print(sep)

    for e in entries:
        ts = str(e.get("timestamp", ""))[:19].replace("T", " ")[-8:]  # HH:MM:SS
        n = e.get("iteration", "?")
        ctype = str(e.get("change_type", "?"))[:12]
        before = e.get("fitness_before", 0.0)
        after = e.get("fitness_after", 0.0)
        delta = after - before
        ok = "✓" if e.get("accepted") else "✗"
        rationale = str(e.get("rationale", ""))[:55]
        print(
            f"{n:>4}  {ts:>8}  {ctype:<12}  "
            f"{before:>+7.4f}  {after:>+7.4f}  {delta:>+7.4f}  {ok:>3}  {rationale}"
        )

    print(sep)

    accepted = [e for e in entries if e.get("accepted")]
    best = max(entries, key=lambda e: e.get("fitness_after", float("-inf")), default=None)
    print(f"\nTotal: {len(entries)}  |  Accepted: {len(accepted)}  |  Rejected: {len(entries) - len(accepted)}")
    if best:
        print(f"Best fitness seen: {best.get('fitness_after', 0.0):+.4f}  (iteration {best.get('iteration')})")


def print_detail(entries: list[dict], iteration: int) -> None:
    match = [e for e in entries if e.get("iteration") == iteration]
    if not match:
        print(f"No entry found for iteration {iteration}.")
        return

    e = match[0]
    print(f"\n── Iteration {iteration} ─────────────────────────────────────────────")
    print(f"  Timestamp:    {e.get('timestamp', 'N/A')}")
    print(f"  Accepted:     {'yes' if e.get('accepted') else 'no'}")
    print(f"  Change type:  {e.get('change_type')}")
    print(f"  Fitness:      {e.get('fitness_before'):+.4f} → {e.get('fitness_after'):+.4f}  (Δ={e.get('delta'):+.4f})")
    print(f"  Has new code: {'yes' if e.get('has_new_code') else 'no'}")
    print(f"\n  Rationale:\n    {e.get('rationale', '')}")
    print(f"\n  Params: {json.dumps(e.get('params', {}), indent=4)}")

    fr = e.get("fitness_result", {})
    if fr:
        print(f"\n  Fitness result:")
        print(f"    mean_sharpe={fr.get('mean_sharpe', 0):+.4f}  std_sharpe={fr.get('std_sharpe', 0):.4f}")
        for w in fr.get("windows", []):
            ok = "✓" if w.get("is_valid") else "⚠"
            print(
                f"    {ok} [{w['window_start'][:10]} → {w['window_end'][:10]}]  "
                f"Sharpe={w['sharpe']:+.3f}  Return={w['total_return_pct']:+.1f}%  "
                f"MDD={w['max_drawdown_pct']:.1f}%  Trades={w['n_trades']}  "
                f"WinRate={w['win_rate']*100:.0f}%"
            )

    if e.get("note"):
        print(f"\n  Note: {e['note']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Show experiment log")
    p.add_argument("--last", "-n", type=int, default=None, help="Show last N entries")
    p.add_argument("--accepted-only", "-a", action="store_true", help="Only accepted experiments")
    p.add_argument("--detail", "-d", type=int, default=None, metavar="ITER", help="Full detail for one iteration")
    args = p.parse_args()

    entries = load_log()
    if not entries:
        print(f"Log is empty or not found at {LOG_PATH}")
        sys.exit(0)

    if args.detail is not None:
        print_detail(entries, args.detail)
        return

    if args.accepted_only:
        entries = [e for e in entries if e.get("accepted")]

    if args.last is not None:
        entries = entries[-args.last :]

    print_table(entries)


if __name__ == "__main__":
    main()
