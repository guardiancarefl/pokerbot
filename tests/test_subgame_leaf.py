"""Structural tests for src.nlhe.subgame_leaf (B1c sub-step 2, Stage B scaffold).

Stage B ships only the Q9 interface contracts (no evaluation behavior). These
tests pin the scaffold: the context constructs with its documented defaults, the
BlueprintProvider protocol accepts a real solver, SubgameNode has the new
leaf_value field defaulting None, and the entry points accept the right argument
types (raising NotImplementedError until Stages C/D/E land).
"""
from __future__ import annotations

import glob
import os
import unittest


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()


def _find_solver_artifacts():
    """Locate an abstraction, a 6-max checkpoint, and the tournament structure.

    Returns (abstraction_path, checkpoint_path, structure_path) or None if any
    is absent (so the real-solver test skips gracefully off-host)."""
    abstr = sorted(glob.glob("runs/abstraction_*/abstraction.pkl"))
    ckpts = sorted(glob.glob("runs/six_max_*/checkpoints/ckpt_iter_*.pt"))
    struct = "configs/ignition_double_up_6max_turbo.yaml"
    if not abstr or not ckpts or not os.path.exists(struct):
        return None
    return abstr[0], ckpts[0], struct


def _make_minimal_context(blueprint=None):
    """A LeafEvalContext with only the required fields supplied."""
    from src.nlhe.subgame_leaf import LeafEvalContext
    from src.nlhe.biased_policy import BiasedBlueprint
    from src.nlhe.icm import sng_payouts_6max_double_up
    return LeafEvalContext(
        blueprint=blueprint if blueprint is not None else object(),
        biased_blueprint=BiasedBlueprint(),
        starting_stacks=[10000] * 6,
        payouts=sng_payouts_6max_double_up(),
        hero_seat=0,
    )


# ============================================================
# Mode + protocol (sandbox-runnable)
# ============================================================

class TestLeafEvalMode(unittest.TestCase):
    def test_two_modes_distinct(self):
        from src.nlhe.subgame_leaf import LeafEvalMode
        self.assertNotEqual(LeafEvalMode.BEST_RESPONSE, LeafEvalMode.PROFILE_SAMPLE)
        self.assertEqual(LeafEvalMode.BEST_RESPONSE.value, "best_response")
        self.assertEqual(LeafEvalMode.PROFILE_SAMPLE.value, "profile_sample")


class TestBlueprintProviderProtocol(unittest.TestCase):
    def test_runtime_checkable_stub(self):
        from src.nlhe.subgame_leaf import BlueprintProvider

        class _Stub:
            encoder = object()
            policy_nets = object()

        class _Missing:
            encoder = object()
            # no policy_nets

        self.assertIsInstance(_Stub(), BlueprintProvider)
        self.assertNotIsInstance(_Missing(), BlueprintProvider)
        self.assertNotIsInstance(object(), BlueprintProvider)


class TestLeafEvalContextDefaults(unittest.TestCase):
    def test_constructs_with_defaults(self):
        from src.nlhe.subgame_leaf import LeafEvalMode
        ctx = _make_minimal_context()
        # documented defaults
        self.assertEqual(ctx.mode, LeafEvalMode.BEST_RESPONSE)
        self.assertEqual(ctx.n_samples, 8)            # provisional design M
        self.assertIsNone(ctx.opponent_prior)
        self.assertIsNone(ctx.rng)
        self.assertTrue(ctx.icm_short_circuit)
        self.assertIsNone(ctx.time_budget_s)
        self.assertEqual(ctx.num_paid, 3)
        # required fields preserved
        self.assertEqual(ctx.hero_seat, 0)
        self.assertEqual(list(ctx.starting_stacks), [10000] * 6)


# ============================================================
# SubgameNode.leaf_value field
# ============================================================

class TestSubgameNodeLeafValueField(unittest.TestCase):
    def test_leaf_value_defaults_none_and_settable(self):
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.LEAF, state=None, depth=4)
        self.assertIsNone(n.leaf_value)
        n.leaf_value = [0.1, -0.2, 0.0, 0.1, 0.0, 0.0]
        self.assertEqual(len(n.leaf_value), 6)

    def test_existing_node_construction_unbroken(self):
        # The field is appended with a default, so positional/keyword construction
        # used elsewhere keeps working.
        from src.nlhe.subgame import SubgameNode, NodeKind
        n = SubgameNode(kind=NodeKind.DECISION, state=None, depth=0,
                        current_player=2)
        self.assertIsNone(n.leaf_value)
        self.assertEqual(n.current_player, 2)


# ============================================================
# Entry-point signatures (scaffold raises NotImplementedError)
# ============================================================

class TestEntryPointSignatures(unittest.TestCase):
    def test_evaluate_leaf_accepts_node_and_ctx(self):
        from src.nlhe.subgame_leaf import evaluate_leaf
        from src.nlhe.subgame import SubgameNode, NodeKind
        node = SubgameNode(kind=NodeKind.LEAF, state=None, depth=4)
        ctx = _make_minimal_context()
        with self.assertRaises(NotImplementedError):
            evaluate_leaf(node, ctx)

    def test_evaluate_leaves_accepts_tree_and_ctx(self):
        from src.nlhe.subgame_leaf import evaluate_leaves
        from src.nlhe.subgame import SubgameNode, SubgameTree, NodeKind
        root = SubgameNode(kind=NodeKind.LEAF, state=None, depth=0)
        tree = SubgameTree(root=root, all_nodes=[root], n_leaf_nodes=1)
        ctx = _make_minimal_context()
        with self.assertRaises(NotImplementedError):
            evaluate_leaves(tree, ctx)


# ============================================================
# Protocol accepts a REAL solver (requires artifacts on host)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestBlueprintProviderRealSolver(unittest.TestCase):
    def test_real_solver_satisfies_protocol(self):
        artifacts = _find_solver_artifacts()
        if artifacts is None:
            self.skipTest("solver artifacts (abstraction/checkpoint/structure) not present")
        abstr_path, ckpt_path, struct_path = artifacts
        from src.nlhe.abstraction import Abstraction
        from src.nlhe.game_strings import TournamentStructure
        from scripts.eval_6max_self_play import _load_solver
        from src.nlhe.subgame_leaf import BlueprintProvider

        abstr = Abstraction.load(abstr_path)
        structure = TournamentStructure.from_yaml(struct_path)
        solver = _load_solver(ckpt_path, abstr, structure)

        self.assertIsInstance(solver, BlueprintProvider)
        # the attributes the leaf evaluator will actually use exist:
        self.assertTrue(hasattr(solver.encoder, "encode_from_parsed"))
        self.assertTrue(hasattr(solver.policy_nets, "predict_advantages"))

        # And it slots into a context as the blueprint.
        ctx = _make_minimal_context(blueprint=solver)
        self.assertIs(ctx.blueprint, solver)


# ============================================================
# PROFILE_SAMPLE mode behavior (Stage C) — requires a real solver
# ============================================================

def _walk_to_decision(game, seed, postflop=False):
    """Return a decision state from the production game (optionally post-flop)."""
    import random
    rng = random.Random(seed)
    s = game.new_initial_state()
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    if not postflop:
        return s
    # everyone calls/checks until the round completes into the flop deal, then
    # advance past the flop chance to the first post-flop decision.
    guard = 0
    while not s.child(1).is_chance_node():
        s = s.child(1); guard += 1; assert guard < 20
    s = s.child(1)
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    return s


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestProfileSampleMode(unittest.TestCase):
    """PROFILE_SAMPLE leaf evaluation against a real loaded blueprint."""

    @classmethod
    def setUpClass(cls):
        artifacts = _find_solver_artifacts()
        if artifacts is None:
            raise unittest.SkipTest("solver artifacts not present")
        import pyspiel
        from src.nlhe.abstraction import Abstraction
        from src.nlhe.game_strings import TournamentStructure, six_max_sng
        from scripts.eval_6max_self_play import _load_solver
        from src.nlhe.biased_policy import BiasedBlueprint
        from src.nlhe.subgame import SubgameNode, NodeKind

        abstr_path, ckpt_path, struct_path = artifacts
        cls.solver = _load_solver(ckpt_path, Abstraction.load(abstr_path),
                                  TournamentStructure.from_yaml(struct_path))
        cls.stack = int(cls.solver.encoder.starting_stack)
        cls.game = pyspiel.load_game(six_max_sng(starting_stack=cls.stack))
        cls.biased = BiasedBlueprint()
        st = _walk_to_decision(cls.game, seed=7)
        cls.leaf = SubgameNode(kind=NodeKind.LEAF, state=st, depth=4,
                               current_player=st.current_player())

    def _ctx(self, leaf=None, **kw):
        from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode
        from src.nlhe.icm import sng_payouts_6max_double_up
        leaf = leaf or self.leaf
        defaults = dict(
            blueprint=self.solver,
            biased_blueprint=self.biased,
            starting_stacks=[self.stack] * 6,
            payouts=list(sng_payouts_6max_double_up()),
            hero_seat=leaf.current_player,
            mode=LeafEvalMode.PROFILE_SAMPLE,
        )
        defaults.update(kw)
        return LeafEvalContext(**defaults)

    # --- 1. reproducibility (same seed -> bit-identical) ---
    def test_reproducible_same_seed(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        r1 = evaluate_leaf(self.leaf, self._ctx(n_samples=8, rng=random.Random(42)))
        r2 = evaluate_leaf(self.leaf, self._ctx(n_samples=8, rng=random.Random(42)))
        self.assertEqual(r1.value, r2.value)
        self.assertFalse(r1.degraded)
        self.assertEqual(len(r1.value), 6)

    # --- 2. conservation (sum ~ 0) ---
    def test_conservation_sum_zero(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        r = evaluate_leaf(self.leaf, self._ctx(n_samples=16, rng=random.Random(3)))
        self.assertLess(abs(sum(r.value)), 1e-6)

    # --- 3. ITM short-circuit equivalence (on vs off) ---
    def test_itm_short_circuit_equivalence(self):
        # Consistent ITM on a fresh 6-alive game requires num_paid >= alive (=6).
        # With equal payouts of length 6, icm_equity is [payout]*6 for ANY positive
        # stacks, so BOTH the Option-A short-circuit and the full rollout yield the
        # zero-vector exactly. (Realistic num_paid=3 ITM needs busted-seat games,
        # exercised elsewhere; this test pins the short-circuit MECHANISM + equivalence.)
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        kw = dict(n_samples=16, num_paid=6, payouts=[2.0] * 6)
        r_on = evaluate_leaf(self.leaf, self._ctx(icm_short_circuit=True,
                                                  rng=random.Random(1), **kw))
        r_off = evaluate_leaf(self.leaf, self._ctx(icm_short_circuit=False,
                                                   rng=random.Random(1), **kw))
        self.assertTrue(r_on.short_circuited)
        self.assertFalse(r_off.short_circuited)
        for i in range(6):
            self.assertLess(abs(r_on.value[i]), 1e-6)
            self.assertLess(abs(r_off.value[i]), 1e-6)

    # --- 4. point-mass-on-blueprint == blueprint-only (within 3x stderr) ---
    def test_pointmass_blueprint_matches_unbiased(self):
        import math
        import random
        from src.nlhe.subgame_leaf import _rollout_once
        M = 64
        hero = self.leaf.current_player
        ctx = self._ctx()  # only blueprint/stacks/payouts/hero used by _rollout_once

        def collect(use_bias):
            vals = []
            for i in range(M):
                r = random.Random(2000 + i)         # identical seed per paired sample
                self.solver.encoder.reset_cache()   # identical cache state per sample
                if use_bias:
                    # point-mass on bias 0 == apply the blueprint bias config
                    fn = lambda probs, mask, cp: self.biased.action_probs(probs, mask, 0)
                else:
                    fn = lambda probs, mask, cp: probs  # raw RM+ blueprint, no bias path
                vec = _rollout_once(self.leaf.state.clone(), ctx, fn, r)
                if vec is not None:
                    vals.append(vec[hero])
            return vals

        a = collect(use_bias=True)    # bias-0 path
        b = collect(use_bias=False)   # unbiased reference
        self.assertGreater(min(len(a), len(b)), 30)
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = sum((x - mean_a) ** 2 for x in a) / max(1, len(a) - 1)
        var_b = sum((x - mean_b) ** 2 for x in b) / max(1, len(b) - 1)
        stderr = math.sqrt(var_a / len(a) + var_b / len(b))
        # If bias-0 != identity (bias-plumbing bug) the means diverge beyond noise.
        self.assertLessEqual(abs(mean_a - mean_b), 3.0 * stderr + 1e-9)

    # --- 5. symmetric-prior consistency (permutation-invariance proxy) ---
    def test_symmetric_prior_seed_consistency(self):
        # Exact opponent-seat permutation symmetry is infeasible in NLHE (distinct
        # hole cards). The realizable core of "symmetric prior -> value determined
        # by the leaf, not the sampling order" is seed-independence in expectation:
        # under a uniform prior, two seeds agree within 3x stderr elementwise.
        import math
        import random
        from src.nlhe.subgame_leaf import _rollout_once
        M = 64
        ctx = self._ctx()  # uniform prior (opponent_prior=None)

        def collect(seed0):
            vecs = []
            for i in range(M):
                r = random.Random(seed0 + i)
                self.solver.encoder.reset_cache()
                opp = {s: 0 for s in range(6)}  # placeholder; replaced below
                # uniform draws over k biases per opponent seat
                k = self.biased.k
                opp = {s: r.choices(range(k), weights=[1.0 / k] * k, k=1)[0]
                       for s in range(6) if s != ctx.hero_seat}
                fn = lambda probs, mask, cp, _o=opp: self.biased.action_probs(
                    probs, mask, 0 if cp == ctx.hero_seat else _o.get(cp, 0))
                vec = _rollout_once(self.leaf.state.clone(), ctx, fn, r)
                if vec is not None:
                    vecs.append(vec)
            return vecs

        va = collect(5000)
        vb = collect(9000)
        self.assertGreater(min(len(va), len(vb)), 30)
        for seat in range(6):
            xa = [v[seat] for v in va]; xb = [v[seat] for v in vb]
            ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
            sa = sum((x - ma) ** 2 for x in xa) / max(1, len(xa) - 1)
            sb = sum((x - mb) ** 2 for x in xb) / max(1, len(xb) - 1)
            stderr = math.sqrt(sa / len(xa) + sb / len(xb))
            self.assertLessEqual(abs(ma - mb), 3.0 * stderr + 1e-9,
                                 f"seat {seat} seed-inconsistent beyond 3 sigma")

    # --- 6. budget guard: degraded flag + wall-clock respected ---
    def test_budget_guard_degrades_and_respects_clock(self):
        import time
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        ctx = self._ctx(n_samples=300, time_budget_s=0.001, rng=random.Random(7))
        t0 = time.perf_counter()
        r = evaluate_leaf(self.leaf, ctx)
        elapsed = time.perf_counter() - t0
        # (a) returns degraded
        self.assertTrue(r.degraded)
        # (b) respects the budget within slack. A correct guard stops after ~1
        # in-flight rollout; a broken guard would run all 300 (~seconds). Slack
        # widened to 0.4 s for the oversubscribed host (CLAUDE.md ~10x variance)
        # while still catching a seconds-long runaway.
        self.assertLessEqual(elapsed, 0.001 + 0.4)

    # --- 7. failure degradation: NaN blueprint -> degraded, no exception ---
    def test_nan_blueprint_degrades_no_raise(self):
        import math
        import random
        import types
        import numpy as np
        from src.nlhe.subgame_leaf import evaluate_leaf, BlueprintProvider

        nan_blueprint = types.SimpleNamespace(
            encoder=self.solver.encoder,  # real encoder (encode_from_parsed/reset_cache)
            policy_nets=types.SimpleNamespace(
                predict_advantages=lambda seat, features: np.full(7, np.nan, np.float32)),
        )
        self.assertIsInstance(nan_blueprint, BlueprintProvider)
        ctx = self._ctx(blueprint=nan_blueprint, n_samples=8, rng=random.Random(0))
        r = evaluate_leaf(self.leaf, ctx)  # must not raise
        self.assertTrue(r.degraded)
        self.assertEqual(r.n_completed, 0)
        self.assertEqual(len(r.value), 6)
        self.assertTrue(all(math.isfinite(x) for x in r.value))


if __name__ == "__main__":
    unittest.main()
