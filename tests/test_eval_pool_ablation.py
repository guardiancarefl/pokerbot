"""Stage 6-A tests for scripts.eval_pool_ablation (B1c sub-step 6 parallel harness).

Pure tests (seeding, aggregation) run anywhere; play/determinism tests need open_spiel
+ the structure yaml; the real-policy smoke needs solver artifacts and skips off-host.
"""
from __future__ import annotations

import glob
import math
import os
import random
import unittest


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()
_STRUCT = "configs/ignition_double_up_6max_turbo.yaml"


def _find_artifacts():
    abstr = sorted(glob.glob("runs/abstraction_*/abstraction.pkl"))
    ckpts = sorted(glob.glob("runs/six_max_*/checkpoints/ckpt_iter_*.pt"))
    if not abstr or not ckpts or not os.path.exists(_STRUCT):
        return None
    return abstr[0], ckpts[0], _STRUCT


# ============================================================
# Pure: seeding + aggregation
# ============================================================

class TestHandSeed(unittest.TestCase):
    def test_deterministic_and_index_derived(self):
        from scripts.eval_pool_ablation import hand_seed
        self.assertEqual(hand_seed(2026, 0, 0), hand_seed(2026, 0, 0))
        self.assertNotEqual(hand_seed(2026, 0, 0), hand_seed(2026, 0, 1))
        self.assertNotEqual(hand_seed(2026, 0, 0), hand_seed(2026, 1, 0))
        self.assertTrue(0 <= hand_seed(2026, 4, 4999) < 2 ** 31)

    def test_collision_rate_under_5pct(self):
        from scripts.eval_pool_ablation import hand_seed
        seeds = [hand_seed(2026, o, h) for o in range(5) for h in range(5000)]  # 25000
        n, u = len(seeds), len(set(seeds))
        collisions = n - u
        rate = collisions / n
        self.assertLess(rate, 0.05, f"collision rate {rate:.4f} ({collisions}/{n})")


class TestAggregate(unittest.TestCase):
    def test_diff_stderr_sigma_and_paired_lift(self):
        import numpy as np
        from scripts.eval_pool_ablation import aggregate
        # 1 opponent, 4 hands; BR contributions vs blueprint contributions
        br = [0.10, 0.20, 0.30, 0.40]
        bp = [0.05, 0.15, 0.20, 0.30]
        records = {0: {h: {"BR": br[h], "BP": bp[h]} for h in range(4)}}
        agg = aggregate(records, ["BR", "BP"], ["oppX"], lift_pairs=[("BR", "BP")])
        # per-matchup BR diff = mean(br)
        mr = agg["per_matchup"]["BR|oppX"]
        self.assertAlmostEqual(mr["diff"], float(np.mean(br)))
        self.assertAlmostEqual(mr["stderr"], float(np.std(br, ddof=1) / math.sqrt(4)))
        self.assertEqual(mr["n_hands"], 4)
        # paired lift = mean(br - bp)
        lift = agg["lifts"]["BR_minus_BP"]
        paired = [b - p for b, p in zip(br, bp)]
        self.assertAlmostEqual(lift["lift"], float(np.mean(paired)))
        self.assertAlmostEqual(lift["stderr"], float(np.std(paired, ddof=1) / math.sqrt(4)))
        self.assertEqual(lift["n_opponents_positive"], 1)

    def test_capped_hands_excluded(self):
        from scripts.eval_pool_ablation import aggregate
        records = {0: {0: {"BR": 0.1}, 1: {"BR": None}, 2: {"BR": 0.3}}}
        agg = aggregate(records, ["BR"], ["o"], lift_pairs=[])
        self.assertEqual(agg["per_matchup"]["BR|o"]["n_hands"], 2)  # None excluded


# ============================================================
# Play / determinism (need open_spiel + structure)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL and os.path.exists(_STRUCT),
                     "needs open_spiel + structure yaml")
class TestCRNAndDeterminism(unittest.TestCase):
    def _structure(self):
        from src.nlhe.game_strings import TournamentStructure
        return TournamentStructure.from_yaml(_STRUCT)

    def test_crn_same_seed_identical_start_across_policies(self):
        # Different challenger policies + the SAME hand seed -> identical seat
        # assignment + deal (chance_rng is consumed before any policy acts).
        from scripts.eval_pool_ablation import _play_one_hand
        from scripts.eval_pool import UniformRandomPolicy
        s = self._structure()
        a = UniformRandomPolicy("A")
        b = UniformRandomPolicy("B")
        opp = UniformRandomPolicy("opp")
        r1 = _play_one_hand(a, opp, s, seed=12345, mode="sample")
        r2 = _play_one_hand(b, opp, s, seed=12345, mode="sample")
        self.assertEqual(r1["seat_assignment"], r2["seat_assignment"])

    def test_same_seed_same_policy_full_determinism(self):
        from scripts.eval_pool_ablation import _play_one_hand
        from scripts.eval_pool import UniformRandomPolicy
        s = self._structure()
        c, o = UniformRandomPolicy("c"), UniformRandomPolicy("o")
        r1 = _play_one_hand(c, o, s, seed=777, mode="sample")
        r2 = _play_one_hand(c, o, s, seed=777, mode="sample")
        self.assertEqual(r1["seat_assignment"], r2["seat_assignment"])
        self.assertEqual(r1["equity"], r2["equity"])
        self.assertEqual(r1["exceeded_cap"], r2["exceeded_cap"])

    def test_fail_loud_worker_exception_propagates(self):
        # A worker exception must bubble up (no silent shard drop). workers=1 runs the
        # worker in-process; patch _play_one_hand to raise.
        from unittest import mock
        from scripts.eval_pool_ablation import run_ablation, PolicySpec
        cspec = [PolicySpec("c", "random")]
        ospec = [PolicySpec("o", "random")]
        with mock.patch("scripts.eval_pool_ablation._play_one_hand",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                run_ablation(cspec, ospec, abstraction_path=None, structure_path=_STRUCT,
                             hands=5, base_seed=1, workers=1)

    def test_determinism_across_worker_counts(self):
        # Cheap policies (random; no checkpoint, no solving) so this is fast. The
        # aggregate must be bit-identical for workers in {1, 4, 8}.
        from scripts.eval_pool_ablation import run_ablation, PolicySpec
        cspec = [PolicySpec("c", "random")]
        ospec = [PolicySpec("o", "random")]
        runs = {}
        for w in (1, 4, 8):
            agg = run_ablation(cspec, ospec, abstraction_path=None,
                               structure_path=_STRUCT, hands=80, base_seed=2026,
                               workers=w, mode="sample")
            runs[w] = agg["per_matchup"]["c|o"]
        self.assertEqual(runs[1], runs[4])
        self.assertEqual(runs[4], runs[8])


# ============================================================
# Real-policy single-matchup smoke (needs artifacts)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "needs open_spiel")
class TestAblationSmoke(unittest.TestCase):
    def test_three_challenger_single_matchup_smoke(self):
        art = _find_artifacts()
        if art is None:
            self.skipTest("solver artifacts not present")
        abstr_path, ckpt_path, struct_path = art
        from scripts.eval_pool_ablation import run_ablation, PolicySpec
        small = dict(n_samples=2, max_action_depth=2, n_iterations=10)
        cspec = [
            PolicySpec("blueprint", "checkpoint", ckpt=ckpt_path),
            PolicySpec("sg-profile", "subgame_profile", ckpt=ckpt_path, solve_kw=small),
            PolicySpec("sg-br", "subgame_br", ckpt=ckpt_path, solve_kw=small),
        ]
        ospec = [PolicySpec("opp", "checkpoint", ckpt=ckpt_path)]
        # workers=1 (in-process): the real-torch-policy smoke validates the full
        # pipeline without a process pool; the multiprocessing fan-out is covered by
        # the cheap-policy determinism test (combining real torch models + a process
        # pool makes the in-suite test contention-fragile).
        agg = run_ablation(cspec, ospec, abstr_path, struct_path, hands=60,
                           base_seed=7, workers=1, mode="sample",
                           lift_pairs=[("sg-br", "blueprint"), ("sg-br", "sg-profile")])
        self.assertEqual(len(agg["per_matchup"]), 3)
        for v in agg["per_matchup"].values():
            self.assertTrue(math.isfinite(v["diff"]))
            self.assertGreater(v["n_hands"], 0)
        self.assertIn("sg-br_minus_blueprint", agg["lifts"])
        # the subgame challengers recorded gate decisions
        self.assertGreater(agg["stats"]["sg-br"]["n_decisions_total"], 0)


# ============================================================
# Stage 6-B: verdict branches (pure) + orchestration/JSON smoke
# ============================================================

class TestVerdict(unittest.TestCase):
    def _v(self, **kw):
        from scripts.eval_pool_ablation import compute_verdict
        base = dict(L=0.006, sigma_L=2.5, ordering_ok=True, n_opp_positive=5,
                    n_opp=5, L_brvsp=0.003, sigma_brvsp=2.0)
        base.update(kw)
        return compute_verdict(**base)["status"]

    def test_pass_strict(self):
        self.assertEqual(self._v(), "PASS")

    def test_substantive_pass(self):
        self.assertEqual(self._v(L=0.003, sigma_L=1.7, n_opp_positive=4,
                                 L_brvsp=0.002, sigma_brvsp=1.6), "SUBSTANTIVE_PASS")

    def test_br_equivalent_to_profile_when_indistinguishable(self):
        # meets strict BR-vs-blueprint, but BR not distinguishable from PROFILE
        self.assertEqual(self._v(L_brvsp=0.0004, sigma_brvsp=0.8),
                         "PASS_BR_EQUIVALENT_TO_PROFILE")

    def test_br_equivalent_when_profile_significantly_better(self):
        from scripts.eval_pool_ablation import compute_verdict
        r = compute_verdict(L=0.006, sigma_L=2.5, ordering_ok=True, n_opp_positive=5,
                            n_opp=5, L_brvsp=-0.004, sigma_brvsp=2.2)
        self.assertEqual(r["status"], "PASS_BR_EQUIVALENT_TO_PROFILE")
        self.assertIn("PROFILE significantly", r["recommendation"])

    def test_ambiguous(self):
        self.assertEqual(self._v(L=0.001, sigma_L=1.0), "AMBIGUOUS")     # <0.002
        self.assertEqual(self._v(L=0.003, sigma_L=1.2, n_opp_positive=3),
                         "AMBIGUOUS")                                     # sigma<1.5

    def test_fail(self):
        self.assertEqual(self._v(L=-0.002, sigma_L=2.5), "FAIL")
        self.assertEqual(self._v(L=0.0), "FAIL")

    def test_substantive_needs_4of5_opponents(self):
        # L/sigma meet substantive but only 3/5 opponents positive -> not substantive
        self.assertEqual(self._v(L=0.003, sigma_L=1.7, n_opp_positive=3,
                                 L_brvsp=0.002, sigma_brvsp=1.6), "AMBIGUOUS")


@unittest.skipUnless(_HAS_OPEN_SPIEL, "needs open_spiel")
class TestThreeWayOrchestration(unittest.TestCase):
    def test_three_way_smoke_and_json_schema(self):
        art = _find_artifacts()
        if art is None:
            self.skipTest("solver artifacts not present")
        import json
        from scripts.eval_pool_ablation import run_three_way, _build_output, \
            BLUEPRINT, SG_PROFILE, SG_BR, PolicySpec
        abstr_path, ckpt_path, struct_path = art
        opp = [PolicySpec("opp", "checkpoint", ckpt=ckpt_path)]
        small = dict(n_samples=2, max_action_depth=2, n_iterations=10)
        agg = run_three_way(ckpt_path, opp, abstr_path, struct_path, hands=50,
                            base_seed=7, workers=1, mode="sample", solve_kw=small)
        out = _build_output(agg, ckpt_path, abstr_path, struct_path, wall_clock_s=1.0)
        # schema: required top-level keys
        for k in ("blueprint", "challengers", "config", "git_rev", "per_matchup",
                  "pooled_diff", "lifts", "stats", "verdict", "wall_clock_s"):
            self.assertIn(k, out)
        self.assertEqual(out["challengers"], [BLUEPRINT, SG_PROFILE, SG_BR])
        self.assertEqual(len(out["per_matchup"]), 3)            # 3 challengers x 1 opp
        self.assertIn("sg-br_minus_blueprint", out["lifts"])
        self.assertIn(out["verdict"]["status"],
                      {"PASS", "SUBSTANTIVE_PASS", "PASS_BR_EQUIVALENT_TO_PROFILE",
                       "AMBIGUOUS", "FAIL"})
        self.assertGreater(out["stats"][SG_BR]["n_decisions_total"], 0)
        json.dumps(out, default=float)                          # serializable


if __name__ == "__main__":
    unittest.main()
