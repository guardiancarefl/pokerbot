# ReBeL Gate-2 — tonight's bootstrap findings + tomorrow's real-verdict plan

Date: 2026-06-03 (overnight bootstrap generation running, PID 20397).

## Tonight's result (rough-net training on ~558k bootstrap samples)

Trained the rough PBS value net (`scripts/rebel_train_value_net.py`) on the
cross-box merged bootstrap (`/workspace/rebel_samples` +
`/workspace/contabo_rebel_samples`, 557,824 samples, both hosts, dim-verified).

- Net: MLP 236→128→64→1, **38,657 params** (~13 samples/param), 10% held out.
- Input: the persisted 236-dim encoding (public block + acting-seat bucket one-hot).
- Target: `root_value` — scalar, **acting-player-relative** solved root value.
- **Held-out: MAE 0.208, RMSE 0.266, R² 0.237** (vs predict-mean baseline
  MAE 0.269 / RMSE 0.305). RMSE/std = 0.87. Trained in 18.6 s on the GPU.

**Read:** the net beats the predict-mean baseline — there IS real PBS→value
signal — but it plateaus at R²≈0.24 from **epoch 1** (epochs 2–60 barely move).
The ceiling is not undertraining; it is two structural caps, both diagnosed
below. This was the make-or-break diagnostic value of the peek.

### Why the cash-rate Gate-2 was DEFERRED (not run tonight)

A cash-rate number tonight would have been misleading noise, for two concrete
reasons surfaced during integration:

1. **Scalar net can't value non-hero leaves.** The net is acting-player-relative
   (trained at hero-decision PBSs where actor = hero). A depth-limited search
   leaf is usually a *different* seat's turn, but the solver needs *root-hero's*
   value there (`subgame_solver` reads only `leaf_value[hero_seat]`). The net
   only predicts the *actor's* value → using it at non-hero leaves is a
   perspective approximation the net never trained on → silent search corruption.
2. **No tight benchmark on this box.** `data/shanky_profiles/` (killphilmtt,
   timidtom, …) is gitignored and was not transferred; only built-in NIT/TAG
   archetypes are present. The real tight-matchup verdict needs the Shanky bots.

## Tomorrow's requirements for the REAL Gate 2

(a) **Per-seat 6-vector value targets** instead of the scalar `root_value`, so
    the net can value *non-hero* leaves correctly (docs/REBEL_PBS_6MAX.md §4
    head A — per-seat ICM-equity-delta 6-vector). This is the blocker that makes
    net-leaf search valid.
(b) **Belief block in the PBS encoding.** Currently only the 236-dim public +
    acting-seat bucket is captured; the soft 6×k per-seat belief block
    (PBS doc §3) is missing — a likely contributor to the R²≈0.24 ceiling.
(c) **M>1 leaves** (PROFILE_SAMPLE n_samples>1, or BEST_RESPONSE) for cleaner
    labels — part of the R² ceiling is irreducible label noise from M=1
    single-sample leaf rollouts, which more data alone will NOT fix.
(d) **scp the Shanky profiles** (killphil/timidtom/etc.) to `data/shanky_profiles/`
    so Gate 2 tests against the real tight benchmarks via
    `scripts/eval_resolver_vs_shanky.py`, not just built-in NIT/TAG.

### Tomorrow's flow

Build per-seat targets + belief block in `sample_io` → regenerate or augment the
sample format accordingly (the ~2.1M overnight samples are scalar/236-dim; decide
augment-in-place vs regenerate) → train properly (sized to ~2.1M, held-out
guided) → run the REAL cash-rate Gate 2: search + net vs bare k=200, tight
matchups (Shanky killphil/timidtom + NIT/TAG) + mixed table. Clearly beating
k=200 → scale to paper-grade per the bootstrap-then-accelerate plan; weak after
these fixes → reconsider whether search is the lever for the fixed-archetype case.

Generation continues overnight (PID 20397) → ~2.1M by morning for the real run.
