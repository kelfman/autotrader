"""
show_log.py — pretty-print experiment logs.

Usage:
    python show_log.py                  # show all V1 experiments
    python show_log.py --last 20        # show the last 20
    python show_log.py --accepted-only  # show only accepted experiments
    python show_log.py --detail 5       # full detail for iteration 5

    python show_log.py --tracks         # per-track summary table (V2)
    python show_log.py --track D        # show iteration table for Track D
    python show_log.py --track D -d 19  # full detail for Track D iteration 19
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT    = Path(__file__).parent
LOG_PATH = _ROOT / "experiment_log.jsonl"
RUNS_DIR = _ROOT / "runs"

TA_BASELINE_FITNESS = 0.4714


# ── Loading ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def load_log() -> list[dict]:
    return _load_jsonl(LOG_PATH)


def _discover_track_logs() -> dict[str, dict]:
    """
    Scan runs/track_*/ for experiment logs and return a dict keyed by track_id
    with {path, signal_class, entries}.
    """
    tracks: dict[str, dict] = {}

    if not RUNS_DIR.exists():
        return tracks

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("track_"):
            continue
        log_path = run_dir / "experiment_log.jsonl"
        entries = _load_jsonl(log_path)

        parts = run_dir.name.split("_", 2)
        track_id = parts[1] if len(parts) >= 2 else run_dir.name
        signal_class = parts[2] if len(parts) >= 3 else "unknown"

        if entries:
            signal_class = entries[0].get("signal_class", signal_class)

        tracks[track_id] = {
            "path": log_path,
            "signal_class": signal_class,
            "entries": entries,
            "run_dir": run_dir,
        }

    v1_entries = load_log()
    if v1_entries:
        tracks["E"] = {
            "path": LOG_PATH,
            "signal_class": "ta_baseline",
            "entries": v1_entries,
            "run_dir": _ROOT,
        }

    return tracks


# ── Per-iteration table (existing) ───────────────────────────────────────────

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
        ts = str(e.get("timestamp", ""))[:19].replace("T", " ")[-8:]
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


# ── Per-track summary (V2) ───────────────────────────────────────────────────

def _fitness_trajectory(entries: list[dict]) -> str:
    """Build a compact string of accepted fitness values showing the discovery arc."""
    accepted = [e for e in entries if e.get("accepted")]
    if not accepted:
        return "—"
    values = [e.get("fitness_after", 0.0) for e in accepted]
    return " → ".join(f"{v:+.3f}" for v in values)


def print_tracks_summary() -> None:
    tracks = _discover_track_logs()

    if not tracks:
        print("No track logs found.")
        return

    sep = "═" * 110
    print()
    print(sep)
    print(f"  {'Track':>5}  {'Signal Class':<25}  {'Iters':>5}  "
          f"{'Accepted':>8}  {'Rate':>5}  {'Best':>8}  {'vs TA':>8}  Trajectory")
    print(sep)

    for tid in sorted(tracks.keys()):
        t = tracks[tid]
        entries = t["entries"]
        n_total = len(entries)
        accepted = [e for e in entries if e.get("accepted")]
        n_accepted = len(accepted)
        rate = n_accepted / n_total * 100 if n_total else 0

        best_entry = max(entries, key=lambda e: e.get("fitness_after", float("-inf")), default=None)
        best_fit = best_entry.get("fitness_after", 0.0) if best_entry else 0.0

        vs_ta = best_fit - TA_BASELINE_FITNESS
        vs_str = f"{vs_ta:+.4f}" if n_total else "—"

        traj = _fitness_trajectory(entries)
        if len(traj) > 40:
            parts = traj.split(" → ")
            if len(parts) > 5:
                traj = " → ".join(parts[:2]) + " → … → " + " → ".join(parts[-2:])

        marker = "★" if best_fit > TA_BASELINE_FITNESS else " "

        print(
            f"{marker} {tid:>4}  {t['signal_class']:<25}  {n_total:>5}  "
            f"{n_accepted:>8}  {rate:>4.0f}%  {best_fit:>+8.4f}  {vs_str:>8}  {traj}"
        )

    print(sep)
    print(f"\n  ★ = beats TA baseline ({TA_BASELINE_FITNESS:+.4f})")

    beat_baseline = [tid for tid, t in tracks.items()
                     if t["entries"] and max(e.get("fitness_after", float("-inf"))
                     for e in t["entries"]) > TA_BASELINE_FITNESS]
    print(f"  Tracks beating baseline: {len(beat_baseline)}/{len(tracks)}"
          f"  ({', '.join(sorted(beat_baseline)) if beat_baseline else 'none'})")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Show experiment logs (V1 + V2 tracks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--last", "-n", type=int, default=None, help="Show last N entries")
    p.add_argument("--accepted-only", "-a", action="store_true", help="Only accepted experiments")
    p.add_argument("--detail", "-d", type=int, default=None, metavar="ITER", help="Full detail for one iteration")
    p.add_argument("--tracks", action="store_true", help="Show per-track summary table (V2)")
    p.add_argument("--track", type=str, default=None, metavar="ID",
                   help="Show iteration table for a specific track (e.g. --track D)")
    args = p.parse_args()

    if args.tracks:
        print_tracks_summary()
        return

    if args.track is not None:
        tid = args.track.upper()
        all_tracks = _discover_track_logs()
        if tid not in all_tracks:
            available = ", ".join(sorted(all_tracks.keys())) if all_tracks else "none"
            print(f"Track '{tid}' not found. Available: {available}")
            sys.exit(1)

        entries = all_tracks[tid]["entries"]
        signal_class = all_tracks[tid]["signal_class"]
        print(f"\n── Track {tid} ({signal_class}) ──\n")

        if args.detail is not None:
            print_detail(entries, args.detail)
            return

        if args.accepted_only:
            entries = [e for e in entries if e.get("accepted")]
        if args.last is not None:
            entries = entries[-args.last:]
        print_table(entries)
        return

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
        entries = entries[-args.last:]

    print_table(entries)


if __name__ == "__main__":
    main()
