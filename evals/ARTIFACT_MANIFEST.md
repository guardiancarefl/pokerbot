# Uncommitted artifact manifest

Generated 2026-06-09 (B0 evidence-preservation pass). These directories/files
stay OUT of git (policy: anything >5MB stays out; see commit message). This
manifest records what exists on the Contabo working tree at cc8a9ae so their absence in a fresh clone is documented, not silent.

## Artifact directories

| path | files | size | representative file | sha256 |
|---|---|---|---|---|
| `evals/short_stack_floor_ab/` | 12 | 440.7 MB | `evals/short_stack_floor_ab/shard_A/games.jsonl` | `fd39d817d3d0c44b234f927073c07775f39c1fc7586e23fde461dd3d7f71c88e` |
| `evals/short_stack_floor_ab_hpl5/` | 12 | 537.3 MB | `evals/short_stack_floor_ab_hpl5/shard_A/games.jsonl` | `d7bcff8cb9cde023eaf656081351ef04211f2167c354f2355a4e37631d49f2fb` |
| `evals/resolver_shards/` | 63 | 590.2 KB | `evals/resolver_shards/baseline_blueprint.json` | `25c4c2585e0949075b7814a56e0a2cb485aaa6ccd348d2f7d94169111c321d8e` |
| `l4_corpus/` | 120 | 45.2 MB | `l4_corpus/l4_corpus_heldout/manifest.json` | `55825d44126b6243c8edad816cc2107d6bdcaeadb4f713b9d6c799897ecc94f0` |
| `.claude/` | 1 | 117.0 B | — | — |
| `evals/sng_baseline_20260610/` (`w*/games_*.jsonl`) | 24 | 7.0 MB | `evals/sng_baseline_20260610/w0/games_6pack.jsonl` | `b265eae849345364394057a45c7ff55fa4e33fa84be3ee33f6cf3e941d71483c` |
| `evals/sng_field_v2_20260610/` (`w*/games_*.jsonl` + `calib_w*/games.jsonl`) | 15 | 2.4 MB | `evals/sng_field_v2_20260610/w0/games_ticketmaster3.jsonl` | `4b6e1330993e002dc8baa5b767b2b448a568f013bda2dd79a220630085d0552f` |

- `evals/short_stack_floor_ab/` — Paired A/B decision+game logs, V0 vs V1 short-stack floor, hpl=3 stress arm (4 shards x 6,000 games)
- `evals/short_stack_floor_ab_hpl5/` — Paired A/B decision+game logs, hpl=5 live-matched arm (4 shards x 6,000 games) - source of the +0.0100 ICM/game headline
- `evals/resolver_shards/` — Resolver-era match-level eval shards (X0-X5 arms + bubble slice); resolver path closed, kept for the record
- `l4_corpus/` — Layer-4 adaptive-training corpus shards (train + heldout pkl.gz + manifests)
- `.claude/` — Local Claude Code session settings - machine-local tooling, not a project artifact
- `evals/sng_baseline_20260610/w*/games_*.jsonl` — SNG baseline v1 raw per-game logs (24 files, 24 profiles x 2000 games = 48,000 records, 7.0 MB total). Regenerable (deterministic, master seed 2026, `scripts/sng_baseline.py`). Committed artifacts: `summary_merged.json` + `REPORT.txt` + per-worker `summary.json`/`.log`.
- `evals/sng_field_v2_20260610/` raw per-game logs — v2 field expansion (7 profiles x 2000 games) + self-play calibration row (8 shards x 250 games). Regenerable (deterministic, master seed 2026, `scripts/sng_baseline.py` / `scripts/sng_selfplay_calibration.py`). Committed artifacts: `summary_merged.json` + `REPORT.txt` + `calibration_merged.json` + `rider1_hero_observed_probe.json` + per-worker `summary.json`/`.log`.
- `logs/` — **CANONICAL LIVE-PLAY RECORD** (gitignored, host-only). The `live_*`/`live_dryrun_*` JSONL families are the sole and permanent record of live play — no official hand-history source exists for this site/format. Never delete; gzip sessions >30 days old. See `logs/README.md` (committed via `git add -f`). Other files here (`*_train.log`, `gate_*.log`, `diag_*.log`) are reproducible run artifacts, not live records.

## Loose uncommitted eval files (small; left untracked pending triage)

| file | size | sha256 |
|---|---|---|
| `evals/X0_resolver_vs_self.json` | 3.5 KB | `8ca6adcfad03d248a9c6d7293d8f00de66a5a25af96f057084525c3f05e227d2` |
| `evals/diag_leak_tight.json` | 1.4 MB | `10dfb408243e9fb3b6066bfcdfff8f8dc63d1bfd6aea69507aadc0c00aa8eda3` |
| `evals/exploit_tier1_candC2000.json` | 19.4 KB | `b5ea6521f31bda50a64103abc01f3ff7fc9e7a37e6b4381d9396c7dd2d2d771d` |
| `evals/k1000_gate_iter100.json` | 462.6 KB | `0e00cbd33d31d0c35342bfa58d3cfd6255df231f32f15869c2c1e2880728d5f9` |
| `evals/k1000_gate_iter100_comp.json` | 75.8 KB | `993714c7eae5a86ff68fca3db394a7fce4b212eb2037a71abb568aaa98889479` |
| `evals/k1000_gate_iter200.json` | 462.6 KB | `a7e3e3c81bcff44946e10a10d6a4fe831e98a80069f0118ade36e2b1ff6647fd` |
| `evals/k1000_gate_iter200_comp.json` | 75.8 KB | `4441b2d22d8c867cb3fb425e262a4cc2dfc281f6bf518df6dff039371bc19052` |
| `evals/k1000_gate_iter300.json` | 462.5 KB | `22cfbb4f8feb847123746759187df39cf3c027d3b3dab7e296cb1cd0f676647f` |
| `evals/k1000_gate_iter300_comp.json` | 75.8 KB | `f978f3753faf6db5c8d8a0c2437d5d2f8478e613cce765c5eeca4fddcc0e7160` |
| `evals/k1000_gate_iter400.json` | 462.6 KB | `cba413eb9e337bee5a74de67c88d8ed0c8e745b914b5dc9e148949fd36fd7c5c` |
| `evals/k1000_gate_iter400_comp.json` | 75.8 KB | `2bba8cb557068e9640d2e1af064ce4c846d6aab9dd02efee98fb891df4dce085` |
| `evals/k1000_gate_iter500.json` | 462.6 KB | `84f9c7117079a968492ad9cefbfd73a0faf08760e0fa4c08730406c66b09986b` |
| `evals/k1000_gate_iter500_comp.json` | 75.8 KB | `dcd8a957dd526bff8847fa54071b5228d67b915a17d22ebd6e9938811d3870e4` |
| `evals/resolver_smoke.jsonl` | 1.9 KB | `09e7d69729087fd7660855bb5f750532b824f24e7d478c61a035fadfbb5cd527` |
| `evals/tournament_baseline_N500.json` | 1.2 MB | `357e1be3a13ea53fc703c56942dcde2a17199aa8d525f684a93a82ac28f13a04` |
| `evals/tournament_first_pass.json` | 681.5 KB | `17a00728647a667c50b4a1497b37e5196c4376b977709e35457cc8b40bbf3288` |
| `evals/tournament_pftighten.json` | 681.5 KB | `146c6eeb307c961a58e1e6d41b1f11b30c97f0d9cd94cb933be0861a0f4596ae` |
| `evals/tournament_pftighten_N500.json` | 1.2 MB | `c9a89427c42c9fa0ff6664371fcc2575182cc0f164268a37df916b49efddea07` |

## Other untracked items (out of B0 scope, listed for transparency)

- `scripts/continue_k200_real_ante.py` (3.3 KB)
- `scripts/monitor_k200_convergence.py` (16.8 KB)
- `scripts/render_learning_curves.py` (10.1 KB)
- `scripts/slope_h2h_k200.py` (4.8 KB)
- `scripts/throwaway_continue.py` (3.2 KB)
- `scripts/train_k200_real_ante.py` (5.4 KB)
- `scripts/verify_live_session_b.py` (18.2 KB)
