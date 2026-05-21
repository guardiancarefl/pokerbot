# Training run directory format

Each subdirectory in `runs/` is one training run, named
`<game>_<YYYYMMDD_HHMMSS>[_<run_name>]/`. Examples:runs/leduc_20260521_200251_smoke/         # smoke test run
runs/leduc_20260521_202346_phase1_default/  # real Phase 1 run

The timestamp is local time at run start. The optional `run_name` suffix
comes from `--run-name` on the CLI (or `run_name:` in the YAML config).

## Files in a run directoryruns/leduc_<ts>[_<name>]/
├── config.json              # effective hyperparameters
├── metrics.json             # final metrics summary
├── training.log             # stdout-mirrored log
├── advantage_losses.csv     # per-iteration, per-player advantage loss
└── checkpoints/
├── final.pt             # trained policy network (+ config + metrics)
├── config.json          # companion copy of effective config
└── metrics.json         # companion copy of final metrics

### `config.json`
The exact `TrainConfig` used for this run, as a flat JSON object. Includes
every field of the dataclass plus a few injected at runtime (`torch_version`,
`device`). To reproduce a run: copy this file's contents into a new YAML
config and rerun. CLI override flags applied to the original run are
already baked into this file's values.

### `metrics.json`
Final summary metrics. Always includes:
- `iterations`: number of Deep CFR iterations completed
- `num_traversals`: traversals per player per iteration
- `final_policy_loss`: loss of the last policy network training pass
- `train_seconds`: wall-clock time of solve()
- `exploitability_mbb_per_game`: Nash exploitability in milli-big-blinds
  per game. Lower is better. Null if `--skip-exploitability` was set.
- `exploitability_seconds`: time spent on the exploitability eval.

### `training.log`
Same content that streamed to stdout, with timestamps. Use this to debug
runs you didn't watch live, or to read back a long-finished run.

### `advantage_losses.csv`
Three columns: `iteration`, `player`, `advantage_loss`. One row per
(iteration, player) pair. Useful for plotting loss curves after the fact.
Empty cells in `advantage_loss` mean OpenSpiel returned `None` for that
iteration's training (insufficient samples in the buffer to form a batch);
these get written as empty strings in CSV and NaN in numpy/pandas.

### `checkpoints/final.pt`
PyTorch checkpoint dict containing:
- `policy_network_state_dict`: weights of the trained policy network
- `config`: same data as `config.json` (embedded for self-contained loading)
- `metrics`: same data as `metrics.json` (embedded)

To load: use `src.leduc.checkpoint.load_checkpoint(path)`. The architecture
of the policy network is determined by `config['policy_network_layers']`;
reconstruct the matching `torch.nn.Sequential` (or use the same `TrainConfig`
to spin up a new `DeepCFRSolver` and assign weights) before calling
`load_state_dict()`.

## What gets committed to git

Nothing. `runs/` is gitignored. Training runs produce a lot of files and
most of them are derivative — the config that produced them lives in
`configs/` (committed), and the metrics are easy to reproduce. If you need
to share a specific run's results, share the relevant files directly
(scp, gist, or attach to an issue).

## Cleaning up old runs

Old run directories are safe to delete. Nothing else in the project
references them. A reasonable retention policy:
- Keep the most recent N runs (e.g., last 5)
- Keep any run referenced in `SESSION_LOG.md` or a published result
- Delete the rest periodically

`du -sh runs/*` shows per-run disk usage if you need to free space.
