#!/usr/bin/env python3
# test_data.py
"""
Quick verifier for DWANode logs.
- Lists available runs under base logs dir
- Loads latest (or chosen) run
- Validates arrays and lengths
- Prints concise summary
- (Optional) exports a CSV summary

Usage:
  python3 test_data.py --base /root/omo_ws/src/omo_local_planner/logs
  python3 test_data.py --base ... --list
  python3 test_data.py --base ... --run 20250811_110423
  python3 test_data.py --base ... --export-csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


RE_RUN = re.compile(r"^\d{8}_\d{6}$")  # e.g., 20250811_110423


def find_runs(base_dir: Path):
    if not base_dir.exists():
        return []
    runs = [d for d in base_dir.iterdir() if d.is_dir() and RE_RUN.match(d.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def load_run(run_dir: Path) -> Tuple[dict, Optional[dict], np.lib.npyio.NpzFile]:
    params_path = run_dir / "params.json"
    meta_path = run_dir / "meta.json"
    data_path = run_dir / "run_data.npz"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    with open(params_path, "r") as f:
        params = json.load(f)

    meta = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

    data = np.load(data_path, allow_pickle=True)
    return params, meta, data


def validate(data) -> bool:
    required = [
        "timestamps", "v_min", "v_max",
        "vel_pairs",
        "goal_costs","path_costs", "align_costs",  "goal_front_costs", "obs_costs",
        "norm_costs",
        "best_pairs", "best_costs",
    ]
    ok = True

    # Keys check
    missing = [k for k in required if k not in data.files]
    if missing:
        print(f"[ERROR] Missing keys in NPZ: {missing}")
        ok = False

    if "timestamps" not in data.files:
        return False

    N = len(data["timestamps"])
    if N == 0:
        print("[WARN] No ticks recorded (timestamps empty).")
        return ok

    # Length consistency
    to_check_len = [
        "v_min", "v_max", "best_pairs", "best_costs", "norm_costs",
        "vel_pairs", "goal_costs","path_costs", "align_costs", "goal_front_costs", "obs_costs",
    ]
    for k in to_check_len:
        if k not in data.files:
            continue
        arr = data[k]
        try:
            length = len(arr)
        except Exception:
            print(f"[ERROR] Cannot get length of {k}")
            ok = False
            continue
        if length != N:
            print(f"[ERROR] Length mismatch: {k}={length}, timestamps={N}")
            ok = False

    # Shape checks (be flexible for norm_costs length >= 4 or variable)
    if "best_pairs" in data.files:
        bp = data["best_pairs"]
        if not (bp.ndim == 2 and bp.shape[1] == 2):
            print(f"[ERROR] best_pairs shape expected (?,2) but got {bp.shape}")
            ok = False

    if "norm_costs" in data.files:
        nc = data["norm_costs"]
        if nc.size > 0:
            if nc.ndim == 2:
                if nc.shape[1] < 1:
                    print(f"[ERROR] norm_costs has zero columns.")
                    ok = False
            elif nc.ndim == 1:
                # object array per tick: acceptable
                pass
            else:
                print(f"[ERROR] Unexpected norm_costs ndim={nc.ndim}")
                ok = False

    # Spot-check object arrays for per-tick candidate lengths
    for k in ["vel_pairs", "path_costs", "goal_costs", "goal_front_costs", "obs_costs"]:
        if k in data.files and len(data[k]) > 0:
            first = data[k][0]
            if not isinstance(first, np.ndarray):
                print(f"[WARN] {k}[0] is {type(first)}, expected ndarray (ok if intentional).")

    return ok


def print_summary(run_dir: Path, params: dict, meta: Optional[dict], data):
    N = len(data["timestamps"])
    print("=" * 70)
    print(f"Run directory : {run_dir}")
    if meta:
        print(f"Start time    : {meta.get('start_time')}")
        print(f"End time      : {meta.get('end_time')}")
    print(f"Ticks recorded: {N}")
    print("-" * 70)
    print("Key params:")
    for k in [
        "dt", "sim_time", "sim_period", "v_samples", "w_samples",
        "acc_lim_v", "acc_lim_w", "wc_obstacle", "wc_path", "wc_goal", "wc_goal_center",
    ]:
        if k in params:
            print(f"  {k:16s}: {params[k]}")

    bp = data["best_pairs"]
    bc = data["best_costs"]
    print("-" * 70)
    print("Best pair / cost stats:")
    if bp.size > 0:
        vmin, vmax = float(bp[:, 0].min()), float(bp[:, 0].max())
        wmin, wmax = float(bp[:, 1].min()), float(bp[:, 1].max())
        print(f"  best_v range: {vmin:.4f} ~ {vmax:.4f}")
        print(f"  best_w range: {wmin:.4f} ~ {wmax:.4f}")
    if bc.size > 0:
        print(f"  best_cost mean/min/max: {float(np.mean(bc)):.6f} / {float(np.min(bc)):.6f} / {float(np.max(bc)):.6f}")

    # Norm costs summary (supports variable length)
    if "norm_costs" in data.files and data["norm_costs"].size > 0:
        nc = data["norm_costs"]
        if nc.ndim == 2:
            means = np.mean(nc, axis=0)
            print(f"  norm_costs mean (len={nc.shape[1]}): {means}")
        else:
            # object array: try convert to 2D by padding to max length
            lengths = [len(np.atleast_1d(nc[i])) for i in range(len(nc))]
            M = max(lengths)
            padded = np.full((len(nc), M), np.nan, dtype=float)
            for i in range(len(nc)):
                row = np.asarray(nc[i], dtype=float).ravel()
                padded[i, :len(row)] = row[:M]
            means = np.nanmean(padded, axis=0)
            print(f"  norm_costs mean (len={M}): {means}")

    # Show first and last tick details
    def show_tick(i: int):
        print("-" * 70)
        print(f"Tick {i}:")
        vp = data["vel_pairs"][i]
        print(f"  candidates shape: {getattr(vp, 'shape', None)}")
        for name in ["obs_costs", "path_costs", "goal_costs", "goal_front_costs"]:
            arr = data[name][i]
            l = len(arr) if hasattr(arr, "__len__") else "N/A"
            print(f"  {name:16s}: len={l}")
        nc_i = data["norm_costs"][i]
        print(f"  norm_costs      : {np.asarray(nc_i).ravel()}")
        print(f"  best_pair       : {data['best_pairs'][i]}")
        print(f"  best_cost       : {float(data['best_costs'][i]):.6f}")

    if N > 0:
        show_tick(0)
        show_tick(N - 1)


def export_csv(run_dir: Path, data):
    import csv

    # Determine norm_costs column count flexibly
    norm_len = 0
    if "norm_costs" in data.files and data["norm_costs"].size > 0:
        nc = data["norm_costs"]
        if nc.ndim == 2:
            norm_len = int(nc.shape[1])
        else:
            # object array; infer from first row
            first = np.asarray(nc[0]).ravel()
            norm_len = int(len(first))
    norm_len_eff = max(4, norm_len)  # ensure at least 4 columns for compatibility

    # Build header dynamically
    base_header = [
        "tick", "time",
        "best_v", "best_w", "best_cost",
        "v_min", "v_max",
    ]
    named = ["obs_norm", "path_norm", "goal_norm", "goal_center_norm"]
    if norm_len_eff <= 4:
        norm_header = named[:norm_len_eff]
    else:
        extra = [f"norm_extra_{i}" for i in range(5, norm_len_eff + 1)]
        norm_header = named + extra

    header = base_header + norm_header

    out_csv = run_dir / "summary.csv"
    N = len(data["timestamps"])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(N):
            time_i = float(data["timestamps"][i])
            best_v, best_w = data["best_pairs"][i]
            best_cost = float(data["best_costs"][i])
            vmin = float(data["v_min"][i])
            vmax = float(data["v_max"][i])

            # collect norm values (pad/truncate to norm_len_eff)
            vals = []
            if "norm_costs" in data.files and data["norm_costs"].size > 0:
                row = np.asarray(data["norm_costs"][i]).ravel().astype(float)
                if len(row) >= norm_len_eff:
                    vals = row[:norm_len_eff].tolist()
                else:
                    vals = row.tolist() + [float("nan")] * (norm_len_eff - len(row))
            else:
                vals = [float("nan")] * norm_len_eff

            w.writerow([i, time_i, best_v, best_w, best_cost, vmin, vmax] + vals)

    print(f"[OK] CSV written -> {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="DWANode log verifier")
    ap.add_argument("--base", type=str, required=True, help="Base logs directory (e.g., /root/omo_ws/src/omo_local_planner/logs)")
    ap.add_argument("--run", type=str, help="Specific run folder name (e.g., 20250811_110423)")
    ap.add_argument("--list", action="store_true", help="List available runs and exit")
    ap.add_argument("--export-csv", action="store_true", help="Export summary.csv for the selected run")
    args = ap.parse_args()

    base = Path(args.base)
    runs = find_runs(base)

    if args.list:
        if not runs:
            print(f"No runs under: {base}")
            return
        print("Available runs (newest first):")
        for r in runs:
            print("  ", r.name)
        return

    if args.run:
        run_dir = base / args.run
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    else:
        if not runs:
            raise FileNotFoundError(f"No runs under: {base}")
        run_dir = runs[0]

    params, meta, data = load_run(run_dir)
    ok = validate(data)
    print_summary(run_dir, params, meta, data)

    if args.export_csv:
        export_csv(run_dir, data)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
