"""External checkpoint-watcher dashboard for the Cand C (or any) training run.

Polls a training run's checkpoints/ dir; for each new ckpt_iter_*.pt that hasn't
been evaluated yet, runs three lightweight head-to-head evals:

  1. Profile-pool lift   — challenger vs N Shanky scripted profiles, ICM-adjusted.
  2. Self-improvement    — challenger vs the immediately-prior checkpoint of the
                            same run (ckpt_iter_{iter - checkpoint_step}).
  3. Anchor improvement  — challenger vs the earliest checkpoint of the run
                            (anchor; e.g. ckpt_iter_0200).

All three lifts plus per-row sigmas are appended to
<run_dir>/profile_dashboard.jsonl. A simple self-contained
<run_dir>/profile_dashboard.html is (re)rendered after every new row, with two
inline-SVG charts (absolute strength vs profile pool, self-improvement vs
prev/anchor) and a small data table. No external server, no server-side libs.

Coupling: ZERO. The dashboard reads checkpoint files and writes its own log
files. It never touches the training process, never restarts anything, never
modifies any training-owned file.

CPU footprint: small. We nice ourselves on startup and skip an eval if one is
already in progress (single-threaded loop, so this is naturally serial). Modest
default --hands keeps each eval round under ~1 min on this 12-vCPU box. Modest
default --profiles caps the pool round to a few scripted opponents.

Resumable: on startup we parse the existing jsonl and skip any iter already
recorded — killing/restarting the dashboard does not re-eval.

Example:

    python scripts/profile_dashboard.py \\
        --run-dir runs/six_max_20260530_030500_phase4f_dcfr_candC_k200 \\
        --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \\
        --structure configs/ignition_double_up_6max_turbo.yaml \\
        --shanky-dir data/shanky_profiles \\
        --profiles 5 --hands 400 --poll-seconds 120

Then open <run_dir>/profile_dashboard.html in a browser.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("profile_dashboard")


# ============================================================
# Helpers
# ============================================================

_CKPT_RE = re.compile(r"^ckpt_iter_(\d+)\.pt$")


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return [(iter_n, path), ...] sorted by iter_n ascending."""
    if not ckpt_dir.is_dir():
        return []
    out = []
    for p in ckpt_dir.iterdir():
        m = _CKPT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    return out


def load_done_iters(jsonl_path: Path) -> set[int]:
    """Read the dashboard jsonl and return iters already evaluated.

    Tolerates a half-written final line (truncated jsonl from a kill mid-write).
    """
    if not jsonl_path.is_file():
        return set()
    done: set[int] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Truncated final line — ignore; the next write will overwrite it.
                continue
            it = row.get("iter")
            if isinstance(it, int):
                done.add(it)
    return done


def append_row(jsonl_path: Path, row: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ============================================================
# Eval primitives (delegated to scripts.eval_pool / eval_shanky_vs_dcfr)
# ============================================================

def _load_eval_globals(abstraction_path: str, structure_path: str):
    """Load abstraction + structure ONCE; reused for every checkpoint eval."""
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    import pickle
    with open(abstraction_path, "rb") as f:
        abstraction = pickle.load(f)
    structure = TournamentStructure.from_yaml(structure_path)
    return abstraction, structure


def _make_checkpoint_policy(name: str, ckpt_path: str, abstraction, structure):
    from scripts.eval_pool import CheckpointPolicy
    return CheckpointPolicy(
        name=name, ckpt_path=ckpt_path, abstraction=abstraction, structure=structure
    )


def _eval_vs_shanky_pool(
    challenger, structure, shanky_dir: str, n_profiles: int, hands: int,
    seed: int, big_blind_chips: int,
) -> dict:
    """Evaluate `challenger` vs the first `n_profiles` loadable Shanky scripted
    bots. Returns aggregate {pooled_diff, pooled_sigma, n_hands_total,
    per_profile=[{name, diff, stderr, sigma, n_hands}, ...]}.

    Pooling is unweighted across profiles: each profile contributes equally so
    a single weak/strong profile doesn't dominate. Pooled-sigma is computed as
    weighted-mean over independent matchups.
    """
    from scripts.eval_shanky_vs_dcfr import _build_shanky_policies
    from scripts.eval_pool import evaluate_matchup
    # Load ALL available profiles; trim to n_profiles. We don't use --only
    # because the name-normalization is fiddly (see eval_shanky_vs_dcfr).
    all_profiles = _build_shanky_policies(
        shanky_dir=shanky_dir, only=None, big_blind_chips=big_blind_chips,
    )
    if not all_profiles:
        return {"pooled_diff": float("nan"), "pooled_sigma": float("nan"),
                "n_hands_total": 0, "per_profile": []}
    pool = all_profiles[:n_profiles]
    per = []
    diffs = []
    stderrs = []
    n_hands_total = 0
    for i, opp in enumerate(pool):
        # Per-matchup seed offset so each is independent.
        r = evaluate_matchup(
            challenger=challenger, opponent=opp, structure=structure,
            hands=hands, seed=seed + 991 * (i + 1),
            mode="sample", log_every=max(hands, 1),
        )
        per.append({
            "name": opp.name, "diff": r["diff"], "stderr": r["stderr"],
            "sigma": r["sigma"], "n_hands": r["n_hands"],
            "n_capped": r["n_capped"],
        })
        diffs.append(r["diff"])
        stderrs.append(r["stderr"])
        n_hands_total += r["n_hands"]
    # Pool: mean of diffs; sigma from sqrt(sum(stderr^2))/len for independent
    # matchups, then pooled sigma = |pooled_diff| / pooled_stderr.
    pooled_diff = sum(diffs) / len(diffs)
    pooled_stderr = math.sqrt(sum(s * s for s in stderrs)) / len(stderrs)
    pooled_sigma = (abs(pooled_diff) / pooled_stderr) if pooled_stderr > 0 else float("nan")
    return {
        "pooled_diff": pooled_diff,
        "pooled_sigma": pooled_sigma,
        "n_hands_total": n_hands_total,
        "per_profile": per,
    }


def _eval_head_to_head(
    challenger, opponent, structure, hands: int, seed: int,
) -> dict:
    from scripts.eval_pool import evaluate_matchup
    return evaluate_matchup(
        challenger=challenger, opponent=opponent, structure=structure,
        hands=hands, seed=seed,
        mode="sample", log_every=max(hands, 1),
    )


# ============================================================
# Rendering
# ============================================================

def _svg_line_chart(
    title: str, x_label: str, y_label: str,
    series: list[tuple[str, list[tuple[float, float]], str]],
    width: int = 800, height: int = 320,
) -> str:
    """Render a simple multi-line chart as inline SVG.

    `series`: list of (label, [(x, y), ...], color).
    Caller passes only finite (x, y) — NaN points are filtered upstream.
    """
    pad_l, pad_r, pad_t, pad_b = 60, 130, 30, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    all_pts = [pt for _, pts, _ in series for pt in pts]
    if not all_pts:
        return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">' \
               f'<text x="{width//2}" y="{height//2}" text-anchor="middle" font-family="monospace" font-size="14">no data</text></svg>'
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if x_hi == x_lo:
        x_hi = x_lo + 1
    y_pad = (y_hi - y_lo) * 0.10 if y_hi > y_lo else max(abs(y_hi), 0.01)
    y_lo -= y_pad
    y_hi += y_pad

    def px(x):
        return pad_l + (x - x_lo) / (x_hi - x_lo) * plot_w

    def py(y):
        return pad_t + (1 - (y - y_lo) / (y_hi - y_lo)) * plot_h

    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
           f'style="background:#fafafa">']
    svg.append(f'<text x="{width//2}" y="20" text-anchor="middle" font-family="monospace" font-size="14" font-weight="bold">{title}</text>')
    # Axes
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1"/>')
    # Y-axis ticks (5)
    for i in range(5):
        yv = y_lo + (y_hi - y_lo) * i / 4
        yp = py(yv)
        svg.append(f'<line x1="{pad_l-4}" y1="{yp}" x2="{pad_l}" y2="{yp}" stroke="#333" stroke-width="1"/>')
        svg.append(f'<text x="{pad_l-8}" y="{yp+4}" text-anchor="end" font-family="monospace" font-size="10">{yv:+.3f}</text>')
    # Zero line if zero is in range
    if y_lo <= 0 <= y_hi:
        y0 = py(0)
        svg.append(f'<line x1="{pad_l}" y1="{y0}" x2="{pad_l+plot_w}" y2="{y0}" stroke="#999" stroke-width="1" stroke-dasharray="4,4"/>')
    # X-axis ticks (every 200 in the iter range usually)
    for tick_x in sorted(set(int(p[0]) for p in all_pts)):
        xp = px(tick_x)
        svg.append(f'<line x1="{xp}" y1="{pad_t+plot_h}" x2="{xp}" y2="{pad_t+plot_h+4}" stroke="#333" stroke-width="1"/>')
        svg.append(f'<text x="{xp}" y="{pad_t+plot_h+15}" text-anchor="middle" font-family="monospace" font-size="10">{tick_x}</text>')
    # Axis labels
    svg.append(f'<text x="{pad_l+plot_w//2}" y="{height-5}" text-anchor="middle" font-family="monospace" font-size="11">{x_label}</text>')
    svg.append(f'<text x="15" y="{pad_t+plot_h//2}" text-anchor="middle" font-family="monospace" font-size="11" transform="rotate(-90 15 {pad_t+plot_h//2})">{y_label}</text>')
    # Series
    for s_idx, (label, pts, color) in enumerate(series):
        if not pts:
            continue
        # Line
        d = " ".join(f'{"M" if i == 0 else "L"} {px(x):.1f} {py(y):.1f}' for i, (x, y) in enumerate(pts))
        svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        # Dots
        for x, y in pts:
            svg.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{color}"/>')
        # Legend
        ly = pad_t + 20 + s_idx * 18
        svg.append(f'<line x1="{pad_l+plot_w+10}" y1="{ly}" x2="{pad_l+plot_w+30}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{pad_l+plot_w+33}" y="{ly+4}" font-family="monospace" font-size="11">{label}</text>')
    svg.append('</svg>')
    return "".join(svg)


def render_dashboard(html_path: Path, jsonl_path: Path, run_dir: Path) -> None:
    rows = []
    if jsonl_path.is_file():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda r: r.get("iter", 0))

    # Series for chart 1: absolute strength (vs profile pool).
    pool_pts = [(r["iter"], r["pool_pooled_diff"])
                for r in rows if isinstance(r.get("pool_pooled_diff"), (int, float))
                and math.isfinite(r["pool_pooled_diff"])]
    chart1 = _svg_line_chart(
        title="Absolute strength: ICM lift vs Shanky profile pool",
        x_label="iter", y_label="pooled ICM lift",
        series=[("pool diff", pool_pts, "#1f77b4")],
    )

    # Series for chart 2: self-improvement (vs prev, vs anchor).
    prev_pts = [(r["iter"], r["self_vs_prev_diff"])
                for r in rows if isinstance(r.get("self_vs_prev_diff"), (int, float))
                and math.isfinite(r["self_vs_prev_diff"])]
    anchor_pts = [(r["iter"], r["self_vs_anchor_diff"])
                  for r in rows if isinstance(r.get("self_vs_anchor_diff"), (int, float))
                  and math.isfinite(r["self_vs_anchor_diff"])]
    chart2 = _svg_line_chart(
        title="Self-improvement: ICM lift vs prev ckpt (blue) + vs anchor ckpt (orange)",
        x_label="iter", y_label="ICM lift",
        series=[
            ("vs prev",   prev_pts,   "#1f77b4"),
            ("vs anchor", anchor_pts, "#ff7f0e"),
        ],
    )

    # Table.
    def _fmt_diff(v):
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "-"
        return f"{v:+.4f}"

    def _fmt_sigma(v):
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            return "-"
        return f"{v:.2f}"

    table_rows = []
    for r in rows[-30:]:  # show only last 30 to keep page small
        table_rows.append(
            "<tr>"
            f"<td>{r.get('iter','')}</td>"
            f"<td>{_fmt_diff(r.get('pool_pooled_diff'))}</td>"
            f"<td>{_fmt_sigma(r.get('pool_pooled_sigma'))}</td>"
            f"<td>{_fmt_diff(r.get('self_vs_prev_diff'))}</td>"
            f"<td>{_fmt_sigma(r.get('self_vs_prev_sigma'))}</td>"
            f"<td>{_fmt_diff(r.get('self_vs_anchor_diff'))}</td>"
            f"<td>{_fmt_sigma(r.get('self_vs_anchor_sigma'))}</td>"
            f"<td>{r.get('hands_per_eval','')}</td>"
            f"<td>{r.get('elapsed_s', 0):.1f}s</td>"
            f"<td>{r.get('timestamp','')[:19]}</td>"
            "</tr>"
        )
    table_html = "\n".join(table_rows)

    now = datetime.datetime.utcnow().isoformat()
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>profile dashboard — {run_dir.name}</title>
<meta http-equiv="refresh" content="60">
<style>
body {{ font-family: monospace; padding: 16px; background: #f0f0f0; max-width: 900px; }}
h1 {{ font-size: 18px; }}
h2 {{ font-size: 14px; margin-top: 24px; }}
table {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
td, th {{ border: 1px solid #aaa; padding: 4px 8px; text-align: right; }}
th {{ background: #ddd; }}
.note {{ color: #666; font-size: 11px; }}
</style></head>
<body>
<h1>profile-pool dashboard</h1>
<div class="note">run: <code>{run_dir}</code> · rendered (UTC) {now} · auto-refresh 60s · {len(rows)} rows</div>

<h2>chart 1 — vs profile pool (absolute strength)</h2>
{chart1}

<h2>chart 2 — vs prev/anchor checkpoint (self-improvement)</h2>
{chart2}

<h2>last 30 rows</h2>
<table>
<thead><tr><th>iter</th><th>pool diff</th><th>pool σ</th><th>vs prev diff</th><th>vs prev σ</th><th>vs anchor diff</th><th>vs anchor σ</th><th>hands/eval</th><th>elapsed</th><th>timestamp (UTC)</th></tr></thead>
<tbody>
{table_html}
</tbody>
</table>
</body></html>
"""
    html_path.write_text(html)


# ============================================================
# Main loop
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="training run directory (containing checkpoints/)")
    ap.add_argument("--abstraction", required=True,
                    help="abstraction.pkl path (must match the training config)")
    ap.add_argument("--structure", required=True,
                    help="tournament structure YAML path (must match training config)")
    ap.add_argument("--shanky-dir", default="data/shanky_profiles",
                    help="dir with .txt Shanky profiles")
    ap.add_argument("--profiles", type=int, default=5,
                    help="number of Shanky profiles to evaluate against per round")
    ap.add_argument("--hands", type=int, default=400,
                    help="hands per matchup (each of the 3 eval rounds uses this)")
    ap.add_argument("--poll-seconds", type=int, default=120,
                    help="how long to sleep between polls for new checkpoints")
    ap.add_argument("--big-blind-chips", type=int, default=100,
                    help="BB chip value passed to ShankyProfilePolicy")
    ap.add_argument("--seed", type=int, default=2026,
                    help="base seed for the eval rng stream")
    ap.add_argument("--nice", type=int, default=15,
                    help="os.nice() delta on startup; 0 to skip")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    jsonl_path = run_dir / "profile_dashboard.jsonl"
    html_path = run_dir / "profile_dashboard.html"

    if not run_dir.is_dir():
        sys.exit(f"--run-dir does not exist: {run_dir}")
    if not Path(args.abstraction).is_file():
        sys.exit(f"--abstraction not found: {args.abstraction}")
    if not Path(args.structure).is_file():
        sys.exit(f"--structure not found: {args.structure}")

    if args.nice > 0:
        try:
            os.nice(args.nice)
            log.info(f"niced self by +{args.nice}")
        except OSError as e:
            log.warning(f"could not nice: {e}")

    log.info("NOTE: this dashboard's eval rounds compete with the training run for CPU. "
             "Modest --hands ({}) and --profiles ({}) keep impact small.".format(
                 args.hands, args.profiles))
    log.info(f"loading abstraction + structure (once)")
    abstraction, structure = _load_eval_globals(args.abstraction, args.structure)

    log.info(f"polling: {ckpt_dir} every {args.poll_seconds}s")
    log.info(f"jsonl:   {jsonl_path}")
    log.info(f"html:    {html_path}")

    while True:
        done = load_done_iters(jsonl_path)
        all_ckpts = list_checkpoints(ckpt_dir)
        anchor = all_ckpts[0] if all_ckpts else None
        pending = [(it, p) for it, p in all_ckpts if it not in done]
        log.info(f"poll: {len(all_ckpts)} ckpts on disk, {len(done)} done, "
                 f"{len(pending)} pending, anchor=iter_{anchor[0] if anchor else '-'}")
        for iter_n, ckpt_path in pending:
            t0 = time.time()
            log.info(f"--- evaluating ckpt_iter_{iter_n} ---")
            try:
                challenger = _make_checkpoint_policy(
                    name=f"candC-{iter_n}", ckpt_path=str(ckpt_path),
                    abstraction=abstraction, structure=structure,
                )
            except Exception as e:
                log.exception(f"failed to load ckpt iter {iter_n}: {e}")
                continue

            # 1) profile pool
            try:
                pool_res = _eval_vs_shanky_pool(
                    challenger=challenger, structure=structure,
                    shanky_dir=args.shanky_dir, n_profiles=args.profiles,
                    hands=args.hands, seed=args.seed,
                    big_blind_chips=args.big_blind_chips,
                )
            except Exception as e:
                log.exception(f"pool eval failed at iter {iter_n}: {e}")
                pool_res = {"pooled_diff": float("nan"),
                            "pooled_sigma": float("nan"),
                            "n_hands_total": 0, "per_profile": []}

            # 2) vs immediately-previous ckpt of the same run
            prev_it, prev_path = None, None
            for it, pp in all_ckpts:
                if it < iter_n:
                    prev_it, prev_path = it, pp
            vs_prev = None
            if prev_path is not None:
                try:
                    prev_pol = _make_checkpoint_policy(
                        name=f"candC-{prev_it}", ckpt_path=str(prev_path),
                        abstraction=abstraction, structure=structure,
                    )
                    vs_prev = _eval_head_to_head(
                        challenger=challenger, opponent=prev_pol,
                        structure=structure, hands=args.hands,
                        seed=args.seed + 7919 * iter_n,
                    )
                except Exception as e:
                    log.exception(f"vs prev eval failed at iter {iter_n}: {e}")

            # 3) vs anchor of the same run (earliest ckpt)
            vs_anchor = None
            if anchor is not None and anchor[0] != iter_n:
                anchor_it, anchor_path = anchor
                try:
                    anchor_pol = _make_checkpoint_policy(
                        name=f"candC-{anchor_it}", ckpt_path=str(anchor_path),
                        abstraction=abstraction, structure=structure,
                    )
                    vs_anchor = _eval_head_to_head(
                        challenger=challenger, opponent=anchor_pol,
                        structure=structure, hands=args.hands,
                        seed=args.seed + 6491 * iter_n,
                    )
                except Exception as e:
                    log.exception(f"vs anchor eval failed at iter {iter_n}: {e}")

            elapsed = time.time() - t0
            row = {
                "iter": iter_n,
                "ckpt_path": str(ckpt_path),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "hands_per_eval": args.hands,
                "elapsed_s": elapsed,
                # profile pool
                "pool_pooled_diff": pool_res.get("pooled_diff"),
                "pool_pooled_sigma": pool_res.get("pooled_sigma"),
                "pool_n_hands_total": pool_res.get("n_hands_total"),
                "pool_per_profile": pool_res.get("per_profile"),
                # vs prev
                "self_vs_prev_iter": prev_it,
                "self_vs_prev_diff": (vs_prev or {}).get("diff"),
                "self_vs_prev_sigma": (vs_prev or {}).get("sigma"),
                "self_vs_prev_n_hands": (vs_prev or {}).get("n_hands"),
                # vs anchor
                "self_vs_anchor_iter": anchor[0] if anchor else None,
                "self_vs_anchor_diff": (vs_anchor or {}).get("diff"),
                "self_vs_anchor_sigma": (vs_anchor or {}).get("sigma"),
                "self_vs_anchor_n_hands": (vs_anchor or {}).get("n_hands"),
            }
            append_row(jsonl_path, row)
            log.info(
                f"iter {iter_n}  elapsed {elapsed:.1f}s  "
                f"pool diff {pool_res.get('pooled_diff'):+.4f} (σ {pool_res.get('pooled_sigma'):.2f})  "
                f"vs prev " + (
                    f"{vs_prev['diff']:+.4f} (σ {vs_prev['sigma']:.2f})" if vs_prev else "n/a"
                ) + f"  vs anchor " + (
                    f"{vs_anchor['diff']:+.4f} (σ {vs_anchor['sigma']:.2f})" if vs_anchor else "n/a"
                )
            )
            render_dashboard(html_path, jsonl_path, run_dir)
            # Drop the challenger / opponent solver objects so memory doesn't bloat.
            del challenger
            if prev_path is not None:
                del prev_pol  # noqa: F821 — only defined if prev_path branch taken
            if anchor is not None and anchor[0] != iter_n:
                del anchor_pol  # noqa: F821

        # Idle render even if nothing new — keeps the page fresh after kills.
        render_dashboard(html_path, jsonl_path, run_dir)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
