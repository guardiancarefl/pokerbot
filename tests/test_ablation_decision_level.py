"""Stage G (Q11 Level 2) decision-level ablation — G-F test plan.

Five tests from docs/STAGE_G_DESIGN.md G-F. The pure stub-solver invariants run
sandbox-free; the integration tests (reproducibility, filter, cost) load the real
blueprint and skip gracefully off-host.
"""
from __future__ import annotations

import glob
import os
import random
import time
import unittest

import numpy as np


def _find_artifacts():
    abstr = sorted(glob.glob("runs/abstraction_*/abstraction.pkl"))
    ckpts = sorted(glob.glob("runs/six_max_*/checkpoints/ckpt_iter_*.pt"))
    struct = "configs/ignition_double_up_6max_turbo.yaml"
    if not abstr or not ckpts or not os.path.exists(struct):
        return None
    return abstr[0], ckpts[0], struct


# ============================================================
# Pure stub-solver invariants (no solver / no game needed)
# ============================================================

class TestStubSolverInvariants(unittest.TestCase):
    def setUp(self):
        from scripts.ablation_decision_level import stub_root_policy
        self.stub = stub_root_policy
        # 5 legal actions (0..4), 2 illegal (5,6)
        self.mask = np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.float64)
        self.adv = np.array([0.4, 0.1, -0.2, 0.3, 0.05, 9.9, 9.9], dtype=np.float64)

    def test_no_signal_cross_mode_identity_and_blueprint(self):
        """Bias-inactive subgame ⇒ q_BR ≡ q_PROFILE ⇒ σ_BR ≡ σ_PROFILE (cross-mode
        identity); and flat q ⇒ r=0 ⇒ σ_M == σ0 (reduces to blueprint)."""
        q_flat = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 0, 0], dtype=np.float64)
        sig_m, sig0 = self.stub(self.adv, q_flat, self.mask)
        # flat q across actions ⇒ regret ≈ 0 ⇒ policy == blueprint (to float32 eps;
        # the residual ~1e-7 comes from σ0 summing to 1 only to float32 precision).
        np.testing.assert_allclose(sig_m, sig0, atol=1e-5)
        # cross-mode identity: identical q vectors ⇒ identical policies
        q_same = np.array([1.5, 0.2, -0.7, 2.0, 0.3, 0, 0], dtype=np.float64)
        sig_br, _ = self.stub(self.adv, q_same, self.mask)
        sig_pr, _ = self.stub(self.adv, q_same.copy(), self.mask)
        np.testing.assert_allclose(sig_br, sig_pr, atol=1e-12)
        self.assertAlmostEqual(float(np.abs(sig_br - sig_pr).sum()), 0.0, places=12)

    def test_synthetic_boost_pulls_toward_action(self):
        """Boosting one legal action's value pulls hero's policy toward it vs blueprint,
        and that action becomes the argmax of the refined policy."""
        astar = 2  # an action the blueprint actually disfavors (adv=-0.2 ⇒ σ0=0)
        q = np.array([0.1, 0.1, 5.0, 0.1, 0.1, 0, 0], dtype=np.float64)
        sig_m, sig0 = self.stub(self.adv, q, self.mask)
        self.assertGreater(sig_m[astar], sig0[astar])
        self.assertEqual(int(np.argmax(sig_m)), astar)
        # illegal actions stay at zero
        self.assertEqual(sig_m[5], 0.0)
        self.assertEqual(sig_m[6], 0.0)
        # policy is a distribution over legal actions (float32 from RM+)
        self.assertAlmostEqual(float(sig_m.sum()), 1.0, places=5)


# ============================================================
# Integration tests (load the real blueprint)
# ============================================================

@unittest.skipIf(_find_artifacts() is None, "solver artifacts absent (off-host)")
class TestStageGIntegration(unittest.TestCase):
    _shared = {}

    @classmethod
    def setUpClass(cls):
        from src.nlhe.abstraction import Abstraction
        from src.nlhe.biased_policy import BiasedBlueprint
        from src.nlhe.game_strings import TournamentStructure
        from scripts.eval_6max_self_play import _load_solver
        abstr_p, ckpt_p, struct_p = _find_artifacts()
        cls._shared["structure"] = TournamentStructure.from_yaml(struct_p)
        cls._shared["solver"] = _load_solver(
            ckpt_p, Abstraction.load(abstr_p), cls._shared["structure"])
        cls._shared["biased"] = BiasedBlueprint()

    def _sample_one_root(self, seed):
        import scripts.ablation_decision_level as adl
        rng = random.Random(seed)
        battery, _ = adl.build_root_battery(self._shared["structure"], rng, 1,
                                            min_late_roots=0)
        return battery[0] if battery else None

    def test_reproducibility_same_seed(self):
        """Same root + same seed ⇒ identical refined policies across two runs."""
        import scripts.ablation_decision_level as adl
        entry = self._sample_one_root(seed=101)
        self.assertIsNotNone(entry, "could not sample a valid root")
        r1 = adl.evaluate_root(entry, self._shared["solver"], self._shared["biased"],
                               samples=6, seed=777)
        r2 = adl.evaluate_root(entry, self._shared["solver"], self._shared["biased"],
                               samples=6, seed=777)
        np.testing.assert_allclose(r1["sigma_br"], r2["sigma_br"], atol=1e-12)
        np.testing.assert_allclose(r1["sigma_profile"], r2["sigma_profile"], atol=1e-12)

    def test_filter_excludes_fold_shove(self):
        """The >=3-action filter is real: every sampled root has >=3 legal actions,
        and the filter rate is tracked (Addition 1)."""
        import scripts.ablation_decision_level as adl
        battery, stats = adl.build_root_battery(
            self._shared["structure"], random.Random(202), 8, min_late_roots=0)
        for e in battery:
            self.assertGreaterEqual(e["n_actions"], adl.MIN_ACTIONS)
            self.assertLessEqual(e["n_actions"], 7)
        # filter-rate bookkeeping is populated and in [0,1]
        self.assertIn("filter_rate", stats)
        self.assertTrue(0.0 <= stats["filter_rate"] <= 1.0)
        self.assertEqual(stats["roots_passing_other_filters"],
                         stats["kept_ge3_actions"] + stats["excluded_few_actions"])

    def test_single_root_cost_sane(self):
        """A single root through all three modes completes within a sane wall-clock
        (cost-estimate sanity, G-F #5). Generous bound to avoid CI flakiness."""
        import scripts.ablation_decision_level as adl
        entry = self._sample_one_root(seed=303)
        self.assertIsNotNone(entry)
        t0 = time.perf_counter()
        rec = adl.evaluate_root(entry, self._shared["solver"], self._shared["biased"],
                                samples=8, seed=42)
        dt = time.perf_counter() - t0
        # design G-D: ~32s/root at M=8; allow 5x headroom for an oversubscribed box.
        self.assertLess(dt, 160.0, f"single root took {dt:.1f}s (expected ~tens of s)")
        # produced a valid policy
        self.assertAlmostEqual(sum(rec["sigma_br"]), 1.0, places=6)
        self.assertEqual(len(rec["legal_actions"]), int(sum(rec["_mask"])))


if __name__ == "__main__":
    unittest.main()
