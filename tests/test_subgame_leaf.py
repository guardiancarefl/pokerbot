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
        self.assertEqual(ctx.n_samples, 8)            # design M (restored 5->8 in session-16 Q13)
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
    def test_evaluate_leaf_signature(self):
        # Both modes are implemented now (Stages C/D), so evaluate_leaf no longer
        # raises NotImplementedError; assert the (node, ctx) signature structurally.
        import inspect
        from src.nlhe.subgame_leaf import evaluate_leaf
        params = list(inspect.signature(evaluate_leaf).parameters)
        self.assertEqual(params[:2], ["node", "ctx"])

    def test_evaluate_leaves_signature(self):
        # evaluate_leaves is implemented (Stage E); assert (tree, ctx) signature.
        import inspect
        from src.nlhe.subgame_leaf import evaluate_leaves
        params = list(inspect.signature(evaluate_leaves).parameters)
        self.assertEqual(params[:2], ["tree", "ctx"])

    def test_evaluate_leaves_handles_zero_leaves(self):
        # A tree with no LEAF nodes (e.g. a terminal root) is a no-op: returns a
        # zero-count summary, mutates nothing, never raises, never touches the
        # blueprint. Sandbox-runnable (no solver needed).
        from src.nlhe.subgame_leaf import evaluate_leaves, LeafBatchResult
        from src.nlhe.subgame import SubgameNode, SubgameTree, NodeKind
        root = SubgameNode(kind=NodeKind.TERMINAL, state=None, depth=0,
                           terminal_returns=[0.0] * 6)
        tree = SubgameTree(root=root, all_nodes=[root], n_terminal_nodes=1)
        res = evaluate_leaves(tree, _make_minimal_context())
        self.assertIsInstance(res, LeafBatchResult)
        self.assertEqual(res.n_leaves, 0)
        self.assertEqual(res.n_evaluated, 0)
        self.assertFalse(res.partial_eval_degraded)


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
# Leaf evaluation behavior (Stages C+D) — requires a real solver
# ============================================================

_SHARED = {}


def _shared_solver():
    """Load the blueprint solver once, cached across test classes."""
    if "solver" not in _SHARED:
        artifacts = _find_solver_artifacts()
        if artifacts is None:
            _SHARED["solver"] = None
            return None
        from src.nlhe.abstraction import Abstraction
        from src.nlhe.game_strings import TournamentStructure
        from scripts.eval_6max_self_play import _load_solver
        abstr_path, ckpt_path, struct_path = artifacts
        _SHARED["structure"] = TournamentStructure.from_yaml(struct_path)
        _SHARED["solver"] = _load_solver(ckpt_path, Abstraction.load(abstr_path),
                                         _SHARED["structure"])
    return _SHARED["solver"]


def _walk_to_decision(game, seed, postflop=False):
    """Return a decision state from the production game (optionally post-flop)."""
    import random
    rng = random.Random(seed)
    s = game.new_initial_state()
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    if not postflop:
        return s
    guard = 0
    while not s.child(1).is_chance_node():
        s = s.child(1); guard += 1; assert guard < 20
    s = s.child(1)
    while s.is_chance_node():
        a, p = zip(*s.chance_outcomes()); s = s.child(int(rng.choices(a, weights=p, k=1)[0]))
    return s


def _busted_seat_decision(structure, stacks, dealer_seat, seed):
    """A decision state from a CONSISTENT busted-seat game (some stacks == 0).

    universal_poker can't 'sit out' a player, so busted seats are given a 1-chip
    placeholder by to_inner_game_string_for_state (they fold on first action).
    This yields a real, consistent game where only the nonzero-stack seats play —
    the only way to get a true sub-6-alive table for ITM / heads-up fixtures.
    """
    import pyspiel
    import random
    bl = structure.level(1)
    gs = structure.to_inner_game_string_for_state(blind_level=bl, stacks=stacks,
                                                  dealer_seat=dealer_seat)
    game = pyspiel.load_game(gs)
    return _walk_to_decision(game, seed)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestProfileSampleMode(unittest.TestCase):
    """Leaf evaluation against a real loaded blueprint. The core invariants are
    parametrized over BOTH modes; mode-specific tests live in their own classes."""

    @classmethod
    def setUpClass(cls):
        solver = _shared_solver()
        if solver is None:
            raise unittest.SkipTest("solver artifacts not present")
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.biased_policy import BiasedBlueprint
        from src.nlhe.subgame import SubgameNode, NodeKind
        cls.solver = solver
        cls.structure = _SHARED["structure"]
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

    def _both_modes(self):
        from src.nlhe.subgame_leaf import LeafEvalMode
        return (LeafEvalMode.PROFILE_SAMPLE, LeafEvalMode.BEST_RESPONSE)

    # ---- parametrized over both modes (Addition 4: rerun via subTest) ----

    def test_reproducible_same_seed(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        for mode in self._both_modes():
            with self.subTest(mode=mode):
                r1 = evaluate_leaf(self.leaf, self._ctx(mode=mode, n_samples=6,
                                                        rng=random.Random(42)))
                r2 = evaluate_leaf(self.leaf, self._ctx(mode=mode, n_samples=6,
                                                        rng=random.Random(42)))
                self.assertEqual(r1.value, r2.value)
                self.assertEqual(len(r1.value), 6)

    def test_conservation_sum_zero(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        for mode in self._both_modes():
            with self.subTest(mode=mode):
                r = evaluate_leaf(self.leaf, self._ctx(mode=mode, n_samples=10,
                                                       rng=random.Random(3)))
                self.assertLess(abs(sum(r.value)), 1e-6)

    def test_itm_short_circuit_equivalence(self):
        # Consistent ITM on a fresh 6-alive game requires num_paid >= alive (=6).
        # With equal payouts of length 6, icm_equity is [payout]*6 for ANY positive
        # stacks, so BOTH the Option-A short-circuit and the full rollout yield the
        # zero-vector exactly. (Realistic num_paid=3 busted-seat ITM is exercised by
        # test_itm_short_circuit_realistic, which documents an icm.py limitation.)
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        for mode in self._both_modes():
            with self.subTest(mode=mode):
                kw = dict(mode=mode, n_samples=10, num_paid=6, payouts=[2.0] * 6)
                r_on = evaluate_leaf(self.leaf, self._ctx(icm_short_circuit=True,
                                                          rng=random.Random(1), **kw))
                r_off = evaluate_leaf(self.leaf, self._ctx(icm_short_circuit=False,
                                                           rng=random.Random(1), **kw))
                self.assertTrue(r_on.short_circuited)
                self.assertFalse(r_off.short_circuited)
                for i in range(6):
                    self.assertLess(abs(r_on.value[i]), 1e-6)
                    self.assertLess(abs(r_off.value[i]), 1e-6)

    def test_budget_guard_degrades_and_respects_clock(self):
        import time
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        for mode in self._both_modes():
            with self.subTest(mode=mode):
                ctx = self._ctx(mode=mode, n_samples=300, time_budget_s=0.001,
                                rng=random.Random(7))
                t0 = time.perf_counter()
                r = evaluate_leaf(self.leaf, ctx)
                elapsed = time.perf_counter() - t0
                self.assertTrue(r.degraded)
                # correct guard stops after ~1 in-flight rollout; broken guard runs
                # all 300 (~seconds). Slack widened to 0.4 s for the oversubscribed
                # host (CLAUDE.md ~10x variance), still catches a seconds-long runaway.
                self.assertLessEqual(elapsed, 0.001 + 0.4)

    def test_nan_blueprint_degrades_no_raise(self):
        import math
        import random
        import types
        import numpy as np
        from src.nlhe.subgame_leaf import evaluate_leaf, BlueprintProvider
        nan_blueprint = types.SimpleNamespace(
            encoder=self.solver.encoder,
            policy_nets=types.SimpleNamespace(
                predict_advantages=lambda seat, features: np.full(7, np.nan, np.float32)),
        )
        self.assertIsInstance(nan_blueprint, BlueprintProvider)
        for mode in self._both_modes():
            with self.subTest(mode=mode):
                ctx = self._ctx(mode=mode, blueprint=nan_blueprint, n_samples=6,
                                rng=random.Random(0))
                r = evaluate_leaf(self.leaf, ctx)  # must not raise
                self.assertTrue(r.degraded)
                self.assertEqual(r.n_completed, 0)
                self.assertTrue(all(math.isfinite(x) for x in r.value))

    # ---- PROFILE_SAMPLE-specific ----

    def test_pointmass_blueprint_matches_unbiased(self):
        import math
        import random
        from src.nlhe.subgame_leaf import _rollout_once
        M = 64
        hero = self.leaf.current_player
        ctx = self._ctx()

        def collect(use_bias):
            vals = []
            for i in range(M):
                r = random.Random(2000 + i)
                self.solver.encoder.reset_cache()
                if use_bias:
                    fn = lambda probs, mask, cp: self.biased.action_probs(probs, mask, 0)
                else:
                    fn = lambda probs, mask, cp: probs
                vec = _rollout_once(self.leaf.state.clone(), ctx, fn, r)
                if vec is not None:
                    vals.append(vec[hero])
            return vals

        a = collect(True); b = collect(False)
        self.assertGreater(min(len(a), len(b)), 30)
        ma = sum(a) / len(a); mb = sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) / max(1, len(a) - 1)
        vb = sum((x - mb) ** 2 for x in b) / max(1, len(b) - 1)
        stderr = math.sqrt(va / len(a) + vb / len(b))
        self.assertLessEqual(abs(ma - mb), 3.0 * stderr + 1e-9)

    def test_symmetric_prior_seed_consistency(self):
        # permutation-invariance proxy: exact seat-permutation symmetry is
        # infeasible in NLHE (distinct hole cards), so we test the realizable core
        # — under a uniform prior, two seeds agree within 3x stderr elementwise.
        import math
        import random
        from src.nlhe.subgame_leaf import _rollout_once, _bias_dist_fn
        M = 64
        ctx = self._ctx()
        k = self.biased.k

        def collect(seed0):
            vecs = []
            for i in range(M):
                r = random.Random(seed0 + i)
                self.solver.encoder.reset_cache()
                opp = {s: r.choices(range(k), weights=[1.0 / k] * k, k=1)[0]
                       for s in range(6) if s != ctx.hero_seat}
                vec = _rollout_once(self.leaf.state.clone(), ctx,
                                    _bias_dist_fn(ctx, opp), r)
                if vec is not None:
                    vecs.append(vec)
            return vecs

        va = collect(5000); vb = collect(9000)
        self.assertGreater(min(len(va), len(vb)), 30)
        for seat in range(6):
            xa = [v[seat] for v in va]; xb = [v[seat] for v in vb]
            ma = sum(xa) / len(xa); mb = sum(xb) / len(xb)
            sa = sum((x - ma) ** 2 for x in xa) / max(1, len(xa) - 1)
            sb = sum((x - mb) ** 2 for x in xb) / max(1, len(xb) - 1)
            stderr = math.sqrt(sa / len(xa) + sb / len(xb))
            self.assertLessEqual(abs(ma - mb), 3.0 * stderr + 1e-9,
                                 f"seat {seat} seed-inconsistent beyond 3 sigma")

    # ---- Stage E: evaluate_leaves batch path (shared cache) ----

    def _small_tree(self, depth=2):
        import random
        from src.nlhe.subgame import build_subgame_tree
        st = _walk_to_decision(self.game, seed=11, postflop=True)
        return build_subgame_tree(st, max_action_depth=depth,
                                  chance_samples_per_node=2, rng=random.Random(11))

    def _tree_ctx(self, tree, **kw):
        from src.nlhe.subgame_leaf import LeafEvalContext, LeafEvalMode
        from src.nlhe.icm import sng_payouts_6max_double_up
        defaults = dict(
            blueprint=self.solver, biased_blueprint=self.biased,
            starting_stacks=[self.stack] * 6,
            payouts=list(sng_payouts_6max_double_up()),
            hero_seat=tree.root.current_player, mode=LeafEvalMode.PROFILE_SAMPLE)
        defaults.update(kw)
        return LeafEvalContext(**defaults)

    def test_evaluate_leaves_mutates_in_place(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaves
        from src.nlhe.subgame import iter_leaf_nodes
        tree = self._small_tree()
        leaves = list(iter_leaf_nodes(tree))
        self.assertGreater(len(leaves), 1)
        res = evaluate_leaves(tree, self._tree_ctx(tree, n_samples=6, rng=random.Random(5)))
        self.assertEqual(res.n_leaves, len(leaves))
        self.assertEqual(res.n_evaluated, len(leaves))
        self.assertFalse(res.partial_eval_degraded)
        for lf in leaves:
            self.assertIsNotNone(lf.leaf_value)
            self.assertEqual(len(lf.leaf_value), 6)

    def test_evaluate_leaves_budget_respected(self):
        import random
        from src.nlhe.subgame_leaf import evaluate_leaves
        from src.nlhe.subgame import iter_leaf_nodes
        tree = self._small_tree()
        leaves = list(iter_leaf_nodes(tree))
        self.assertGreater(len(leaves), 1)
        # time_budget_s=0.0 -> the between-leaves deadline trips before any leaf,
        # so the batch is partial. Must not raise.
        res = evaluate_leaves(tree, self._tree_ctx(tree, n_samples=6,
                                                   time_budget_s=0.0, rng=random.Random(5)))
        self.assertTrue(res.partial_eval_degraded)
        self.assertLess(res.n_evaluated, res.n_leaves)
        # consistency: exactly the evaluated leaves are populated; the rest are None.
        n_populated = sum(1 for lf in leaves if lf.leaf_value is not None)
        n_none = sum(1 for lf in leaves if lf.leaf_value is None)
        self.assertEqual(n_populated, res.n_evaluated)
        self.assertEqual(n_none, res.n_leaves - res.n_evaluated)

    def test_evaluate_leaves_matches_evaluate_leaf_sequentially(self):
        """Batching does not STRUCTURALLY change the per-leaf result. Asserts:
          (a) reproducibility — same seed -> bit-identical leaf values;
          (b) ICM conservation — each leaf value 6-vector sums to ~0;
          (c) agreement with standalone evaluate_leaf — MEAN |diff| is small (most
              leaves match closely), per-element bounded.

        SURFACED FINDING: cache-sharing (the Stage E speedup) FREEZES each
        (hero, board) bucket-MC draw for the whole batch, whereas standalone
        evaluate_leaf re-draws it per call. Because the abstraction's bucket MC is
        noisy (bucket_runouts=20), a few leaves can differ from per-leaf-fresh
        evaluation by ~0.5 (a bucket-assignment flip), which exceeds pure rollout
        stderr. This is NOT a bug — both are valid given the bucket noise, and the
        batch's values are self-consistent across leaves (the property sub-step 3's
        CFR actually needs). The mean-|diff| bound below catches a real structural
        break (which would shift ~every element) while tolerating these flips."""
        import random
        from src.nlhe.subgame_leaf import evaluate_leaves, evaluate_leaf
        from src.nlhe.subgame import iter_leaf_nodes
        M = 24
        tree = self._small_tree()
        leaves = list(iter_leaf_nodes(tree))
        evaluate_leaves(tree, self._tree_ctx(tree, n_samples=M, rng=random.Random(101)))
        batch_vals = [list(lf.leaf_value) for lf in leaves]
        # (a) reproducibility: re-run with the same seed -> identical.
        for lf in leaves:
            lf.leaf_value = None
        evaluate_leaves(tree, self._tree_ctx(tree, n_samples=M, rng=random.Random(101)))
        for idx, lf in enumerate(leaves):
            self.assertEqual(list(lf.leaf_value), batch_vals[idx],
                             f"evaluate_leaves not reproducible at leaf {idx}")
        # (b) conservation.
        for idx, b in enumerate(batch_vals):
            self.assertLess(abs(sum(b)), 1e-6, f"leaf {idx} not conserved: sum={sum(b)}")
        # (c) mostly-agrees with standalone; bounded outliers.
        diffs = []
        for idx, lf in enumerate(leaves):
            seq = evaluate_leaf(lf, self._tree_ctx(tree, n_samples=M,
                                                   rng=random.Random(7777))).value
            for seat in range(6):
                diffs.append(abs(batch_vals[idx][seat] - seq[seat]))
        mean_diff = sum(diffs) / len(diffs)
        self.assertLessEqual(mean_diff, 0.2,
                             f"mean |batch-seq|={mean_diff:.3f} too large (structural divergence?)")
        self.assertLessEqual(max(diffs), 1.5,
                             f"max |batch-seq|={max(diffs):.3f} too large")

    # ---- Addition 2: biases actually bias the rollout ----

    def test_biases_produce_different_rollouts(self):
        """Addition 2 — biases must actually bias (not silently default to bias 0).

        DEVIATION (surfaced in the Stage D report): the spec sketched a HERO-value
        directional MC signal (fold-biased opponents -> higher hero EV by >3 sigma).
        Measured, that effect is real but tiny (~0.05) and washes out against the
        per-hand ICM variance (~2.0) even at M>=120 (a 15bb preflop hero often folds
        immediately, so opponent style barely moves hero's value at a single leaf) —
        a 3-sigma MC gate there would be flaky. The SAME failure mode (biases 1-3
        collapsing to bias 0 / broken bias application) is caught DETERMINISTICALLY
        and hero-hand-independently here: with real blueprint probs, the biased
        action distributions the rollout samples from MUST differ across biases, in
        the right direction (fold-bias raises P(FOLD); raise-bias lowers it).
        """
        import random
        import numpy as np
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.infoset6 import parse_state_6max
        from src.nlhe.fast_view import fast_view_and_discretize
        from src.nlhe.subgame_leaf import _blueprint_probs, _legal_mask
        ctx = self._ctx()
        parsed = parse_state_6max(self.leaf.state)
        _view, disc, _legal = fast_view_and_discretize(self.leaf.state, parsed)
        cp = self.leaf.current_player
        probs = _blueprint_probs(ctx, parsed, cp, disc, random.Random(1))
        self.assertIsNotNone(probs)
        mask = _legal_mask(disc)
        d0 = self.biased.action_probs(probs, mask, 0)  # blueprint
        d1 = self.biased.action_probs(probs, mask, 1)  # fold-biased
        d3 = self.biased.action_probs(probs, mask, 3)  # raise-biased
        # biases must NOT collapse to the blueprint or to each other
        self.assertFalse(np.allclose(d1, d0), "fold-bias == blueprint (not applied)")
        self.assertFalse(np.allclose(d3, d0), "raise-bias == blueprint (not applied)")
        self.assertGreater(float(np.max(np.abs(d1 - d3))), 0.05,
                           "fold- and raise-biased distributions barely differ")
        # directional sanity on the distribution (when FOLD is a non-trivial option)
        F = int(DiscreteAction.FOLD)
        if mask[F] > 0 and 0.0 < probs[F] < 1.0:
            self.assertGreater(d1[F], d0[F])  # fold-biased folds MORE
            self.assertLess(d3[F], d0[F])     # raise-biased folds LESS

    # ---- Addition 3: realistic busted-seat ITM (documents an icm.py limitation) ----

    def test_itm_short_circuit_realistic(self):
        """Realistic ITM (num_paid=3, 3 PRE-busted + 3 alive) — the regime that fires
        in production. After the icm.py pre-busted fix (eligible-set in
        icm_adjust_returns), Option-A (short-circuit) and Option-B (full rollout)
        AGREE: both are ~0. The 3 alive are locked at the equal payout, a mid-rollout
        bust keeps the newly-busted seat at its in-money finish (delta 0), and
        pre-busted seats are excluded rather than spuriously enriched. (Pre-fix this
        test documented a large discrepancy; the fix removes it — see test_icm /
        test_icm_returns for the unit-level coverage.)"""
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf, LeafEvalMode, LeafEvalContext
        from src.nlhe.icm import is_itm, sng_payouts_6max_double_up
        from src.nlhe.subgame import SubgameNode, NodeKind
        stacks_itm = [4000, 4000, 4000, 0, 0, 0]
        self.assertTrue(is_itm(stacks_itm, 3))  # fixture genuinely fires
        st = _busted_seat_decision(self.structure, stacks_itm, dealer_seat=0, seed=3)
        leaf = SubgameNode(kind=NodeKind.LEAF, state=st, depth=4,
                           current_player=st.current_player())

        def ev(sc, seed):
            return evaluate_leaf(leaf, LeafEvalContext(
                blueprint=self.solver, biased_blueprint=self.biased,
                starting_stacks=stacks_itm, payouts=list(sng_payouts_6max_double_up()),
                hero_seat=leaf.current_player, mode=LeafEvalMode.PROFILE_SAMPLE,
                num_paid=3, icm_short_circuit=sc, n_samples=40, rng=random.Random(seed)))

        r_on = ev(True, 1)
        r_off = ev(False, 1)
        self.assertTrue(r_on.short_circuited)
        self.assertGreater(r_off.n_completed, 0)
        # Both ~0 (locked ITM), and they AGREE — the Q5 short-circuit accuracy claim
        # now holds in the realistic busted-seat regime after the icm.py fix.
        self.assertLess(max(abs(x) for x in r_on.value), 1e-6)
        self.assertLess(max(abs(x) for x in r_off.value), 1e-6)
        max_gap = max(abs(r_on.value[i] - r_off.value[i]) for i in range(6))
        self.assertLess(max_gap, 1e-6)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestBestResponseMode(unittest.TestCase):
    """BEST_RESPONSE-specific mechanism checks (Q10 #9, #10, #11)."""

    @classmethod
    def setUpClass(cls):
        solver = _shared_solver()
        if solver is None:
            raise unittest.SkipTest("solver artifacts not present")
        import pyspiel
        from src.nlhe.game_strings import six_max_sng
        from src.nlhe.biased_policy import BiasedBlueprint
        from src.nlhe.subgame import SubgameNode, NodeKind
        cls.solver = solver
        cls.structure = _SHARED["structure"]
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
            blueprint=self.solver, biased_blueprint=self.biased,
            starting_stacks=[self.stack] * 6,
            payouts=list(sng_payouts_6max_double_up()),
            hero_seat=leaf.current_player, mode=LeafEvalMode.BEST_RESPONSE)
        defaults.update(kw)
        return LeafEvalContext(**defaults)

    # ---- chance-node-leaf safety (sub-step-5 integration fix) ----

    def _chance_leaf(self):
        """A LEAF whose state is a CHANCE node: a board-deal pending because the
        betting round closes at the depth limit (the leaf shape the depth-K solver
        produces but Stages F/G never exercised)."""
        from src.nlhe.subgame import SubgameNode, NodeKind
        s = _walk_to_decision(self.game, seed=7)
        guard = 0
        while not s.child(1).is_chance_node():   # call until the round closes -> chance
            s = s.child(1); guard += 1
            if guard > 25:
                self.skipTest("no round-closing chance state reached")
        cs = s.child(1)
        self.assertTrue(cs.is_chance_node())
        return SubgameNode(kind=NodeKind.LEAF, state=cs, depth=4, current_player=None)

    def test_chance_leaf_evaluates_finite_no_crash(self):
        # Regression: before the fix, parsing a chance-node leaf crashed
        # (observation_string(current_player()==-1)). Must now evaluate cleanly.
        import math
        import random
        from src.nlhe.subgame_leaf import evaluate_leaf
        leaf = self._chance_leaf()
        res = evaluate_leaf(leaf, self._ctx(leaf=leaf, hero_seat=0, n_samples=6,
                                            rng=random.Random(1)))
        v = res.value
        self.assertEqual(len(v), 6)
        self.assertTrue(all(math.isfinite(x) for x in v))           # finite
        self.assertLess(max(abs(x) for x in v), 3.0)                # ICM-delta magnitude
        self.assertLess(abs(sum(v)), 0.05)                          # ~prize-pool-conserved

    def test_chance_leaf_bias_values_computed(self):
        # The chance-safe parse -> live-opp detection -> per-bias rollouts chain
        # produces finite values for all k biases (the BR mechanism reaches the
        # post-board-deal decisions via the rollout). Whether the k values DIFFER
        # (bias-active) or are equal (bias-inactive) is leaf-dependent and reported
        # from the aggregate measurement (predominantly inactive in this turbo regime).
        import math
        import random
        from src.nlhe.subgame_leaf import (
            _opponent_bias_values, _menu_biases, _parse_leaf_state)
        leaf = self._chance_leaf()
        money = _parse_leaf_state(leaf.state)["money"]
        live = [o for o in range(6) if o != 0 and money[o] > 0]
        self.assertTrue(live, "expected live opponents at a flop-deal chance leaf")
        o = live[0]
        menu = _menu_biases(None, o, self.biased.k)
        ctx = self._ctx(leaf=leaf, hero_seat=0, n_samples=6)
        vals, hit = _opponent_bias_values(leaf, ctx, o, menu, random.Random(3))
        self.assertFalse(hit)
        means = {b: (sum(v) / len(v) if v else float("nan")) for b, v in vals.items()}
        self.assertEqual(len(means), self.biased.k)
        self.assertTrue(all(math.isfinite(m) for m in means.values()))

    def test_chance_leaf_uses_blueprint_only_skips_br(self):
        # Stage-5-B cost optimization: _best_response_biases returns an EMPTY br_dict
        # at a chance-node leaf (skip the v×k BR menu -> phase 2 plays all opponents at
        # blueprint). Justified by 22/25 chance leaves being bias-inactive.
        import random
        from src.nlhe.subgame_leaf import _best_response_biases, evaluate_leaf
        leaf = self._chance_leaf()
        br, hit = _best_response_biases(leaf, self._ctx(leaf=leaf, hero_seat=0,
                                                        n_samples=4), random.Random(1))
        self.assertEqual(br, {})        # blueprint-only: no BR bias selection
        self.assertFalse(hit)
        # the leaf still evaluates to a finite ICM-magnitude value (via blueprint roll)
        res = evaluate_leaf(leaf, self._ctx(leaf=leaf, hero_seat=0, n_samples=4,
                                            rng=random.Random(1)))
        self.assertTrue(all(abs(x) < 3.0 for x in res.value))

    # ---- Q10 #9: maximization fires (max >= mean) + argmax selection ----

    def test_br_mechanism_opponent_value_ge_uniform(self):
        import random
        from src.nlhe.subgame_leaf import (
            _opponent_bias_values, _best_response_biases, _menu_biases)
        from src.nlhe.infoset6 import parse_state_6max
        money = parse_state_6max(self.leaf.state)["money"]
        hero = self.leaf.current_player
        live = [o for o in range(6) if o != hero and money[o] > 0]
        self.assertTrue(live)
        o = live[0]
        k = self.biased.k
        menu = _menu_biases(None, o, k)
        ctx = self._ctx(n_samples=30)
        vals, _ = _opponent_bias_values(self.leaf, ctx, o, menu, random.Random(999))
        means = {b: sum(v) / len(v) for b, v in vals.items() if v}
        max_v = max(means.values())
        mean_v = sum(means.values()) / len(means)
        # BR (max over biases) >= uniform (mean over biases): the max>=mean identity.
        self.assertGreaterEqual(max_v, mean_v - 1e-9)
        # the selection picks the argmax with lowest-index tie-break.
        br, _ = _best_response_biases(self.leaf, ctx, random.Random(999))
        expected = min(b for b in menu if means[b] >= max_v - 1e-9)
        self.assertEqual(br[o], expected)

    # ---- Q10 #10: tie-break determinism ----

    def test_br_tie_break_deterministic(self):
        # A CALL-dominant synthetic blueprint puts all RM+ mass on CALL; every bias
        # config leaves an all-CALL distribution unchanged, so all biases produce
        # IDENTICAL opponent values (exact tie under CRN) -> lowest index (0) wins.
        import random
        import types
        import numpy as np
        from src.nlhe.subgame_leaf import _best_response_biases

        def call_dominant(seat, features):
            a = np.full(7, -1.0, dtype=np.float32)
            a[1] = 10.0  # CALL
            return a
        stub = types.SimpleNamespace(
            encoder=self.solver.encoder,
            policy_nets=types.SimpleNamespace(predict_advantages=call_dominant))
        brs = []
        for seed in (1, 2, 3):
            ctx = self._ctx(blueprint=stub, n_samples=6, rng=random.Random(seed))
            br, _ = _best_response_biases(self.leaf, ctx, random.Random(seed))
            brs.append(br)
        for br in brs:
            for o, b in br.items():
                self.assertEqual(b, 0, "tie should resolve to lowest bias index")
        self.assertEqual(brs[0], brs[1])
        self.assertEqual(brs[1], brs[2])

    # ---- Q10 #11 (optional): hero value BR <= uniform in heads-up zero-sum ----

    def test_br_hero_value_le_uniform_2p(self):
        # Heads-up (one live opponent) via a 2-alive busted-seat game, winner-take-all
        # (num_paid=1) so ICM is ~zero-sum and is_itm does NOT fire. An opponent who
        # best-responds can only hurt hero vs a uniform-prior opponent. Optional /
        # non-blocking: uses generous tolerance; the load-bearing mechanism check is #9.
        import math
        import random
        from src.nlhe.icm import is_itm
        from src.nlhe.subgame import SubgameNode, NodeKind
        from src.nlhe.subgame_leaf import (
            LeafEvalContext, LeafEvalMode, _rollout_once, _bias_dist_fn,
            _best_response_biases)
        stacks_hu = [3000, 3000, 0, 0, 0, 0]   # 2 alive -> heads-up
        self.assertFalse(is_itm(stacks_hu, 1))  # num_paid=1 -> short-circuit won't fire
        st = _busted_seat_decision(self.structure, stacks_hu, dealer_seat=0, seed=5)
        leaf = SubgameNode(kind=NodeKind.LEAF, state=st, depth=4,
                           current_player=st.current_player())
        hero = leaf.current_player
        payouts = [2.0]  # winner-take-all
        base = dict(blueprint=self.solver, biased_blueprint=self.biased,
                    starting_stacks=stacks_hu, payouts=payouts, hero_seat=hero,
                    num_paid=1, icm_short_circuit=False)
        opp = [o for o in range(6) if o != hero and stacks_hu[o] > 0]
        if not opp:
            self.skipTest("heads-up fixture has no live opponent")
        M = 40

        def hero_value(biases_fn, seed0):
            vals = []
            ctx = LeafEvalContext(mode=LeafEvalMode.BEST_RESPONSE, n_samples=M,
                                  rng=random.Random(seed0), **base)
            for i in range(M):
                r = random.Random(seed0 + i)
                self.solver.encoder.reset_cache()
                vec = _rollout_once(leaf.state.clone(), ctx, biases_fn(r), r)
                if vec is not None:
                    vals.append(vec[hero])
            return vals, ctx

        # BR: opponent plays its best-response bias.
        ctx_br = LeafEvalContext(mode=LeafEvalMode.BEST_RESPONSE, n_samples=M,
                                 rng=random.Random(123), **base)
        br, _ = _best_response_biases(leaf, ctx_br, random.Random(123))
        br_vals, _ = hero_value(lambda r: _bias_dist_fn(ctx_br, br), 4242)
        # Uniform: opponent draws bias uniformly each rollout.
        k = self.biased.k
        uni_vals, _ = hero_value(
            lambda r: _bias_dist_fn(ctx_br, {o: r.choices(range(k),
                                    weights=[1.0 / k] * k, k=1)[0] for o in opp}), 4242)
        self.assertGreater(min(len(br_vals), len(uni_vals)), 20)
        m_br = sum(br_vals) / len(br_vals); m_uni = sum(uni_vals) / len(uni_vals)
        vbr = sum((x - m_br) ** 2 for x in br_vals) / max(1, len(br_vals) - 1)
        vun = sum((x - m_uni) ** 2 for x in uni_vals) / max(1, len(uni_vals) - 1)
        stderr = math.sqrt(vbr / len(br_vals) + vun / len(uni_vals))
        # hero no better under BR opponent than under uniform opponent (1-sided, 3 sigma).
        self.assertLessEqual(m_br, m_uni + 3.0 * stderr)


if __name__ == "__main__":
    unittest.main()
