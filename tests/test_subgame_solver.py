"""Stage 3-A tests for src.nlhe.subgame_solver (B1c sub-step 3 scaffold).

Stage 3-A ships the solver module, warm-up caching, and the K=0 blueprint-
passthrough path. These tests pin: the dataclasses construct/validate, K=0 returns
the blueprint's masked root policy BIT-IDENTICALLY, the result is reproducible and a
valid distribution, and the K>=1 boundary is an explicit NotImplementedError.

The K=0 / reproducibility / sanity tests build a REAL production decision state via
six_max_sng (no checkpoint needed) and drive a MOCK blueprint that replaces only the
network — the game, the tree, and the masks are real.
"""
from __future__ import annotations

import random
import unittest

import numpy as np


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()


# ============================================================
# Mock blueprint (replaces only the network; game/tree/masks are real)
# ============================================================

class _MockEncoder:
    """Encoder stub: returns a fixed feature vector (the mock net ignores it) and
    a no-op cache reset. Shape is irrelevant — predict_advantages does not read it."""

    def encode_from_parsed(self, parsed, rng=None):
        return np.zeros(8, dtype=np.float32)

    def reset_cache(self):
        pass


class _MockNets:
    """Advantage net stub: deterministic 7-vector per seat (some negative, so RM+
    zeros them and the resulting distribution is non-uniform — a meaningful sanity
    check). Identical inputs → identical outputs, so the solve is reproducible."""

    _BASE = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05], dtype=np.float32)

    def predict_advantages(self, seat, features):
        return (self._BASE + np.float32(0.01) * np.float32(seat)).astype(np.float32)


class _MockBlueprint:
    def __init__(self):
        self.encoder = _MockEncoder()
        self.policy_nets = _MockNets()


def _first_decision_state(starting_stack: int = 10000, seed: int = 42):
    """Real six_max_sng decision state, past hole-card chance (mirrors
    tests/test_subgame.py)."""
    import pyspiel
    from src.nlhe.game_strings import six_max_sng
    state = pyspiel.load_game(six_max_sng(starting_stack=starting_stack)).new_initial_state()
    r = random.Random(seed)
    while state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state = state.child(int(r.choices(actions, weights=probs, k=1)[0]))
    return state


def _make_ctx(tree, n_iterations=0, rng=None, hero_seat=None):
    from src.nlhe.icm import sng_payouts_6max_double_up
    from src.nlhe.subgame_solver import SubgameSolveContext
    hero = tree.root.current_player if hero_seat is None else hero_seat
    return SubgameSolveContext(
        blueprint=_MockBlueprint(),
        starting_stacks=[10000] * 6,
        payouts=sng_payouts_6max_double_up(),
        hero_seat=hero,
        n_iterations=n_iterations,
        rng=rng,
    )


# ============================================================
# Sandbox-runnable: dataclass construction / validation
# ============================================================

class TestContextValidation(unittest.TestCase):
    def _ctx(self, **over):
        from src.nlhe.icm import sng_payouts_6max_double_up
        from src.nlhe.subgame_solver import SubgameSolveContext
        kw = dict(blueprint=object(), starting_stacks=[10000] * 6,
                  payouts=sng_payouts_6max_double_up(), hero_seat=0)
        kw.update(over)
        return SubgameSolveContext(**kw)

    def test_defaults(self):
        ctx = self._ctx()
        self.assertEqual(ctx.n_iterations, 1000)
        self.assertEqual(ctx.num_paid, 3)
        self.assertEqual(ctx.average_weighting, "linear")
        self.assertIsNone(ctx.rng)

    def test_rejects_bad_stack_len(self):
        with self.assertRaises(ValueError):
            self._ctx(starting_stacks=[10000] * 5)

    def test_rejects_empty_payouts(self):
        with self.assertRaises(ValueError):
            self._ctx(payouts=[])

    def test_rejects_bad_hero_seat(self):
        with self.assertRaises(ValueError):
            self._ctx(hero_seat=6)

    def test_rejects_negative_iterations(self):
        with self.assertRaises(ValueError):
            self._ctx(n_iterations=-1)

    def test_rejects_bad_weighting(self):
        with self.assertRaises(ValueError):
            self._ctx(average_weighting="quadratic")


class TestResultFields(unittest.TestCase):
    def test_result_constructs(self):
        from src.nlhe.subgame_solver import SubgameSolveResult
        res = SubgameSolveResult(
            root_policy=np.zeros(7, dtype=np.float32),
            root_blueprint=np.zeros(7, dtype=np.float32),
            legal_mask=np.zeros(7, dtype=np.float32),
            hero_seat=0, n_iterations=0, n_decision_nodes_cached=1,
        )
        self.assertFalse(res.degraded)
        self.assertTrue(np.isnan(res.converged_l1_tail))


# ============================================================
# K=0 path (real game + mock blueprint)
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestK0BlueprintPassthrough(unittest.TestCase):
    def _tree(self, depth=1, seed=42):
        from src.nlhe.subgame import build_subgame_tree
        state = _first_decision_state(seed=seed)
        return build_subgame_tree(state, max_action_depth=depth,
                                  rng=random.Random(seed + 1))

    def test_k0_matches_blueprint_masked_policy_bit_identical(self):
        from src.nlhe.solver import _strategy_from_advantages
        from src.nlhe.subgame_solver import solve_subgame, _mask_from_children
        tree = self._tree(depth=1)
        ctx = _make_ctx(tree, n_iterations=0)
        res = solve_subgame(tree, ctx)

        # Independently recompute the blueprint's masked policy at the root the
        # exact same way warm-up does, and require BIT-identity.
        hero = tree.root.current_player
        adv = _MockNets().predict_advantages(hero, None)
        mask = _mask_from_children(tree.root)
        expected = _strategy_from_advantages(adv.astype(np.float32), mask)

        self.assertTrue(np.array_equal(res.root_policy, expected),
                        f"{res.root_policy} != {expected}")
        # K=0: refined == blueprint, and nothing degraded (no leaves read).
        self.assertTrue(np.array_equal(res.root_policy, res.root_blueprint))
        self.assertEqual(res.n_iterations, 0)
        self.assertFalse(res.degraded)
        self.assertGreaterEqual(res.n_decision_nodes_cached, 1)

    def test_warmup_covers_interior_decision_nodes(self):
        # A depth-2 tree has interior decision nodes beyond the root; warm-up must
        # cache all of them (count matches the tree's decision-node count).
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=2)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        self.assertEqual(res.n_decision_nodes_cached, tree.n_decision_nodes)

    def test_root_policy_is_valid_distribution(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        p, mask = res.root_policy, res.legal_mask
        self.assertEqual(p.shape, (7,))
        self.assertTrue(np.all(p >= 0.0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        # mass only on legal actions
        self.assertEqual(float(p[mask == 0].sum()), 0.0)
        self.assertGreaterEqual(float(mask.sum()), 1.0)

    def test_reproducible_same_seed(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1, seed=7)
        r1 = solve_subgame(tree, _make_ctx(tree, n_iterations=0, rng=random.Random(123)))
        tree2 = self._tree(depth=1, seed=7)
        r2 = solve_subgame(tree2, _make_ctx(tree2, n_iterations=0, rng=random.Random(123)))
        self.assertTrue(np.array_equal(r1.root_policy, r2.root_policy))

    def test_root_not_hero_raises(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        wrong = (tree.root.current_player + 1) % 6
        with self.assertRaises(ValueError):
            solve_subgame(tree, _make_ctx(tree, n_iterations=0, hero_seat=wrong))

    def test_k2_runs_via_general_loop(self):
        # K>1 is now implemented (Stage 3-C) — it no longer raises; it returns a
        # valid refined policy via the multi-iteration loop.
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=1)
        for c in tree.root.children:
            if c.is_leaf:
                c.leaf_value = [0.0] * 6  # populate so nothing is degraded
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=2))
        self.assertEqual(res.n_iterations, 2)
        self.assertFalse(res.degraded)
        self.assertAlmostEqual(float(res.root_policy.sum()), 1.0, places=5)


# ============================================================
# Stage 3-B: the K=1 path (Stage-G-stub bit-identity gate)
# ============================================================

def _depth1_tree(seed=42):
    """Real six_max_sng depth-1 subgame around hero's first decision."""
    from src.nlhe.subgame import build_subgame_tree
    return build_subgame_tree(_first_decision_state(seed=seed),
                              max_action_depth=1, rng=random.Random(seed + 1))


def _set_leaf_values(tree, hero_entry_fn, seed=99):
    """Hand-populate every LEAF child's leaf_value deterministically (bypasses the
    rollout evaluator so the bit-identity gate isolates the regret-update wiring).
    hero_entry_fn(action_int) -> hero's value at that action; other seats get
    deterministic noise (irrelevant to hero-value q)."""
    rng = np.random.default_rng(seed)
    hero = tree.root.current_player
    for child in tree.root.children:
        if child.is_leaf:
            v = rng.normal(0.0, 0.2, size=6)
            v[hero] = hero_entry_fn(int(child.action_from_parent))
            child.leaf_value = v.tolist()


def _independent_q(tree, ctx):
    """Recompute hero action values q[a] from the root children WITHOUT calling the
    solver's helper — the reference the bit-identity gate feeds to stub_root_policy."""
    from src.nlhe.icm_returns import icm_adjust_returns
    q = np.zeros(7, dtype=np.float64)
    hero = tree.root.current_player
    for child in tree.root.children:
        a = int(child.action_from_parent)
        if child.is_terminal:
            icm = icm_adjust_returns(list(child.terminal_returns),
                                     list(ctx.starting_stacks), list(ctx.payouts))
            q[a] = float(icm[hero])
        else:
            q[a] = float(child.leaf_value[hero])
    return q


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestK1RegretUpdate(unittest.TestCase):
    def test_k1_bit_identical_to_stub(self):
        """THE GATE: K=1 solver root_policy == Stage-G stub_root_policy(adv, q, mask),
        bit-identical, on a real ≥3-action production decision."""
        from scripts.ablation_decision_level import stub_root_policy
        from src.nlhe.subgame_solver import solve_subgame, _mask_from_children
        tree = _depth1_tree(seed=42)
        self.assertGreaterEqual(len(tree.root.children), 3,
                                "need a root with >=3 legal actions for a non-trivial gate")
        # deterministic, clearly non-flat leaf values (so the update does real work)
        _set_leaf_values(tree, hero_entry_fn=lambda a: 0.05 * a - 0.1)
        ctx = _make_ctx(tree, n_iterations=1)
        res = solve_subgame(tree, ctx)

        hero = tree.root.current_player
        adv = _MockNets().predict_advantages(hero, None)
        mask = _mask_from_children(tree.root)
        q = _independent_q(tree, ctx)
        sigma_stub, _sigma0 = stub_root_policy(adv, q, mask)

        self.assertTrue(
            np.array_equal(res.root_policy, sigma_stub),
            f"K=1 solver {res.root_policy} != stub {sigma_stub}")
        self.assertEqual(res.n_iterations, 1)
        self.assertFalse(res.degraded)

    def test_flat_q_reduces_to_blueprint_passthrough(self):
        """Bias-inactive root (flat q across legal actions) ⇒ r=0 ⇒ K=1 == K=0."""
        from src.nlhe.subgame_solver import solve_subgame
        tree = _depth1_tree(seed=42)
        self.assertTrue(all(c.is_leaf for c in tree.root.children),
                        "this fixture assumes no terminal children at the root")
        _set_leaf_values(tree, hero_entry_fn=lambda a: 0.5)  # constant hero value
        k1 = solve_subgame(tree, _make_ctx(tree, n_iterations=1))
        k0 = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        np.testing.assert_allclose(k1.root_policy, k0.root_policy, atol=1e-6)

    def test_mask_preservation_pure(self):
        """Masked actions stay 0 regardless of how large their adv/q are."""
        from src.nlhe.subgame_solver import _regret_matched_policy
        adv = np.array([0.4, 0.1, -0.2, 0.3, 0.05, 9.9, 9.9], dtype=np.float64)
        mask = np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.float64)  # 5,6 illegal
        q = np.array([1.0, 2.0, 3.0, 0.5, 0.1, 100.0, 100.0], dtype=np.float64)
        sigma_m, sigma0 = _regret_matched_policy(adv, q, mask)
        self.assertEqual(float(sigma_m[5]), 0.0)
        self.assertEqual(float(sigma_m[6]), 0.0)
        self.assertEqual(float(sigma0[5]), 0.0)
        self.assertAlmostEqual(float(sigma_m.sum()), 1.0, places=5)

    def test_k0_vs_k1_differ_on_bias_active_root(self):
        """K=1 actually does work: a strongly bias-active root moves the policy off
        the blueprint passthrough by L1 > 0.001."""
        from src.nlhe.subgame_solver import solve_subgame
        tree = _depth1_tree(seed=42)
        # one action far more valuable than the rest -> large regret -> mass shifts
        acts = sorted(int(c.action_from_parent) for c in tree.root.children)
        boosted = acts[-1]
        _set_leaf_values(tree, hero_entry_fn=lambda a: (2.0 if a == boosted else -0.5))
        k0 = solve_subgame(tree, _make_ctx(tree, n_iterations=0))
        k1 = solve_subgame(tree, _make_ctx(tree, n_iterations=1))
        l1 = float(np.abs(k1.root_policy - k0.root_policy).sum())
        self.assertGreater(l1, 0.001, f"K=1 did not move the policy (L1={l1})")

    def test_k1_reproducible_same_seed(self):
        t1 = _depth1_tree(seed=7)
        _set_leaf_values(t1, hero_entry_fn=lambda a: 0.05 * a, seed=5)
        t2 = _depth1_tree(seed=7)
        _set_leaf_values(t2, hero_entry_fn=lambda a: 0.05 * a, seed=5)
        from src.nlhe.subgame_solver import solve_subgame
        r1 = solve_subgame(t1, _make_ctx(t1, n_iterations=1, rng=random.Random(123)))
        r2 = solve_subgame(t2, _make_ctx(t2, n_iterations=1, rng=random.Random(123)))
        self.assertTrue(np.array_equal(r1.root_policy, r2.root_policy))


# ============================================================
# Stage 3-C: the K>1 multi-iteration vanilla weighted CFR loop
# ============================================================

def _set_all_leaf_values(tree, seed=99):
    """Hand-populate every LEAF in the tree with a deterministic 6-vector."""
    from src.nlhe.subgame import iter_leaf_nodes
    rng = np.random.default_rng(seed)
    for leaf in iter_leaf_nodes(tree):
        leaf.leaf_value = rng.normal(0.0, 0.3, size=6).tolist()


def _synthetic_chance_tree():
    """Analytical anchor (hero root, no real game state). Tests chance weighting:

        root (hero, actions {0,1})
          action 0 -> CHANCE {p=0.9 -> leaf hero=1.0 ; p=0.1 -> leaf hero=0.0}  => q[0]=0.9
          action 1 -> leaf hero=0.7                                              => q[1]=0.7

    Correct chance_prob weighting gives q[0]=0.9 > q[1]=0.7 ⇒ the BR is pure on
    action 0. A uniform mis-weighting would give q[0]=0.5 < 0.7 ⇒ pure on action 1,
    so the assertion (pure on 0) catches a weighting bug. Returns (tree, cache, hero).
    """
    from src.nlhe.subgame import SubgameNode, SubgameTree, NodeKind
    from src.nlhe.subgame_solver import _WarmupCache
    from src.nlhe.solver import _strategy_from_advantages
    hero = 0

    def leaf(v, prob, afp):
        n = SubgameNode(kind=NodeKind.LEAF, state=None, depth=2,
                        action_from_parent=afp, chance_prob=prob)
        lv = [0.0] * 6
        lv[hero] = v
        n.leaf_value = lv
        return n

    l_hi = leaf(1.0, 0.9, 0)
    l_lo = leaf(0.0, 0.1, 1)
    chance = SubgameNode(kind=NodeKind.CHANCE, state=None, depth=1,
                         action_from_parent=0, chance_prob=1.0)
    chance.children = [l_hi, l_lo]
    chance.action_at_child = [0, 1]
    l_b = leaf(0.7, 1.0, 1)

    root = SubgameNode(kind=NodeKind.DECISION, state=None, depth=0,
                       current_player=hero)
    root.children = [chance, l_b]
    root.action_at_child = [0, 1]  # DiscreteAction 0 and 1

    tree = SubgameTree(root=root, all_nodes=[root, chance, l_hi, l_lo, l_b],
                       n_decision_nodes=1, n_chance_nodes=1, n_leaf_nodes=3)

    cache = _WarmupCache()
    nid = id(root)
    mask = np.zeros(7, dtype=np.float32)
    mask[0] = 1.0
    mask[1] = 1.0
    cache.adv[nid] = np.zeros(7, dtype=np.float32)  # uniform start
    cache.mask[nid] = mask
    cache.sigma[nid] = _strategy_from_advantages(cache.adv[nid], mask)
    return tree, cache, hero


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestKMultiIteration(unittest.TestCase):
    def _tree(self, depth=2, seed=42):
        from src.nlhe.subgame import build_subgame_tree
        t = build_subgame_tree(_first_decision_state(seed=seed),
                               max_action_depth=depth, rng=random.Random(seed + 1))
        _set_all_leaf_values(t, seed=seed + 50)
        return t

    def test_k100_valid_distribution(self):
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=2)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=100))
        p, mask = res.root_policy, res.legal_mask
        self.assertEqual(p.shape, (7,))
        self.assertTrue(np.all(p >= 0.0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertEqual(float(p[mask == 0].sum()), 0.0)
        self.assertEqual(res.n_iterations, 100)

    def test_k1_general_loop_matches_special_branch(self):
        """The general K>1 loop at N=1 reproduces the Stage-3-B special branch
        (which solve_subgame routes K==1 to), within tiny tolerance."""
        import random as _r
        from src.nlhe.subgame_solver import solve_subgame, _build_warmup, _run_cfr
        tree = _depth1_tree(seed=42)
        _set_leaf_values(tree, hero_entry_fn=lambda a: 0.05 * a - 0.1)
        ctx1 = _make_ctx(tree, n_iterations=1)
        special = solve_subgame(tree, ctx1).root_policy  # K==1 special branch
        cache = _build_warmup(tree, ctx1, _r.Random(0))
        general_k1 = _run_cfr(tree, ctx1, cache)["root_policy"]
        np.testing.assert_allclose(general_k1, special, atol=1e-6)

    def test_convergence_l1_tail_decreases(self):
        from src.nlhe.subgame_solver import solve_subgame
        tail = {}
        for K in (10, 100, 1000):
            tree = self._tree(depth=2)  # identical tree+leaves each K
            res = solve_subgame(tree, _make_ctx(tree, n_iterations=K))
            tail[K] = res.converged_l1_tail
        # average-policy movement shrinks with more iterations (~1/K)
        self.assertGreater(tail[100], tail[1000])
        self.assertGreater(tail[10], tail[100])

    def test_k100_reproducible(self):
        from src.nlhe.subgame_solver import solve_subgame
        t1 = self._tree(depth=2, seed=11)
        t2 = self._tree(depth=2, seed=11)
        r1 = solve_subgame(t1, _make_ctx(t1, n_iterations=100, rng=random.Random(1)))
        r2 = solve_subgame(t2, _make_ctx(t2, n_iterations=100, rng=random.Random(1)))
        self.assertTrue(np.array_equal(r1.root_policy, r2.root_policy))

    def test_cost_gate_k1000_under_5s(self):
        import time
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=3, seed=42)
        t0 = time.perf_counter()
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=1000))
        dt = time.perf_counter() - t0
        self.assertLess(dt, 5.0, f"K=1000 took {dt:.2f}s on a {len(tree.all_nodes)}-node tree")
        self.assertEqual(res.n_iterations, 1000)


class TestVanillaWeightingAnalytical(unittest.TestCase):
    """Math anchor: no real game state, pure loop verification."""

    def test_converges_to_analytical_best_response(self):
        from src.nlhe.icm import sng_payouts_6max_double_up
        from src.nlhe.subgame_solver import SubgameSolveContext, _run_cfr
        tree, cache, hero = _synthetic_chance_tree()
        ctx = SubgameSolveContext(
            blueprint=_MockBlueprint(), starting_stacks=[10000] * 6,
            payouts=sng_payouts_6max_double_up(), hero_seat=hero, n_iterations=1000)
        out = _run_cfr(tree, ctx, cache)
        policy, degraded = out["root_policy"], out["degraded"]
        # q[0]=0.9 (=0.9*1.0+0.1*0.0, correct chance weighting) > q[1]=0.7 ⇒ pure on 0
        self.assertGreater(policy[0], 0.99, f"policy={policy}")
        self.assertLess(policy[1], 0.01, f"policy={policy}")
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        self.assertFalse(degraded)


# ============================================================
# Stage 3-D: diagnostic enrichment
# ============================================================

@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestDiagnosticEnrichment(unittest.TestCase):
    def _tree(self, depth=2, tseed=42, lseed=92):
        from src.nlhe.subgame import build_subgame_tree
        t = build_subgame_tree(_first_decision_state(seed=tseed),
                               max_action_depth=depth, rng=random.Random(tseed + 1))
        _set_all_leaf_values(t, seed=lseed)
        return t

    def test_diagnostic_fields_populated_and_serializable(self):
        import json
        from src.nlhe.subgame_solver import solve_subgame, summarize_solve_result
        res = solve_subgame(self._tree(), _make_ctx(self._tree(), n_iterations=100))
        # all Stage-3-D fields present, right types/shapes
        self.assertEqual(res.n_iterations_run, 100)
        self.assertIsInstance(res.convergence_history, tuple)
        self.assertGreater(len(res.convergence_history), 0)
        for arr in (res.root_q_values, res.root_advantages_blueprint,
                    res.root_advantages_refined):
            self.assertIsNotNone(arr)
            self.assertEqual(np.asarray(arr).shape, (7,))
        # JSON-serializable summary
        summary = summarize_solve_result(res)
        json.dumps(summary)  # raises if not serializable
        self.assertIn("policy_shift_l1", summary)
        self.assertEqual(len(summary["convergence_history"]), len(res.convergence_history))

    def test_convergence_history_monotonic(self):
        from src.nlhe.subgame_solver import solve_subgame
        res = solve_subgame(self._tree(), _make_ctx(self._tree(), n_iterations=100))
        l1s = [l1 for _, l1 in res.convergence_history]
        self.assertGreaterEqual(len(l1s), 2)
        # non-increasing (allow a tiny relative uptick for float noise)
        for a, b in zip(l1s, l1s[1:]):
            self.assertLessEqual(b, a * (1.0 + 1e-6) + 1e-12,
                                 f"convergence history rose: {l1s}")

    def test_q_values_advantages_consistency(self):
        from src.nlhe.subgame_solver import solve_subgame
        res = solve_subgame(self._tree(), _make_ctx(self._tree(), n_iterations=100))
        legal = np.where(res.legal_mask == 1)[0]
        q = np.asarray(res.root_q_values)[legal]
        refined = np.asarray(res.root_advantages_refined)
        # RM+ clamp: refined advantages are non-negative everywhere
        self.assertTrue(np.all(refined >= 0.0))
        # the highest-value legal action also carries the most refined regret
        self.assertEqual(int(legal[np.argmax(q)]),
                         int(legal[np.argmax(refined[legal])]))

    def test_k1_bit_identity_preserved_with_diagnostics(self):
        from scripts.ablation_decision_level import stub_root_policy
        from src.nlhe.subgame_solver import solve_subgame, _mask_from_children
        tree = _depth1_tree(seed=42)
        self.assertGreaterEqual(len(tree.root.children), 3)
        _set_leaf_values(tree, hero_entry_fn=lambda a: 0.05 * a - 0.1)
        ctx = _make_ctx(tree, n_iterations=1)
        res = solve_subgame(tree, ctx)
        hero = tree.root.current_player
        adv = _MockNets().predict_advantages(hero, None)
        mask = _mask_from_children(tree.root)
        q = _independent_q(tree, ctx)
        sigma_stub, _ = stub_root_policy(adv, q, mask)
        # Stage-3-D enrichment did NOT perturb the K=1 output
        self.assertTrue(np.array_equal(res.root_policy, sigma_stub))
        # and the K=1 path populates diagnostics too
        self.assertEqual(res.n_iterations_run, 1)
        self.assertTrue(np.array_equal(np.asarray(res.root_q_values), q))

    def test_diagnostics_reproducible(self):
        from src.nlhe.subgame_solver import solve_subgame
        r1 = solve_subgame(self._tree(), _make_ctx(self._tree(), n_iterations=100,
                                                   rng=random.Random(3)))
        r2 = solve_subgame(self._tree(), _make_ctx(self._tree(), n_iterations=100,
                                                   rng=random.Random(3)))
        self.assertTrue(np.array_equal(r1.root_policy, r2.root_policy))
        self.assertTrue(np.array_equal(np.asarray(r1.root_q_values),
                                       np.asarray(r2.root_q_values)))
        self.assertTrue(np.array_equal(np.asarray(r1.root_advantages_refined),
                                       np.asarray(r2.root_advantages_refined)))
        self.assertEqual(r1.convergence_history, r2.convergence_history)


# ============================================================
# Stage 3-E: production K locked at 1000
# ============================================================

class TestProductionK(unittest.TestCase):
    def test_default_k_is_1000(self):
        from src.nlhe.icm import sng_payouts_6max_double_up
        from src.nlhe.subgame_solver import SubgameSolveContext
        ctx = SubgameSolveContext(blueprint=object(), starting_stacks=[10000] * 6,
                                  payouts=sng_payouts_6max_double_up(), hero_seat=0)
        self.assertEqual(ctx.n_iterations, 1000)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestProductionKGated(unittest.TestCase):
    def _tree(self, depth, seed=42):
        from src.nlhe.subgame import build_subgame_tree
        t = build_subgame_tree(_first_decision_state(seed=seed),
                               max_action_depth=depth, rng=random.Random(seed + 1))
        _set_all_leaf_values(t, seed=seed + 50)
        return t

    def test_production_k_convergence_locked(self):
        # Locks convergence quality at the production K — a regression in the
        # average-strategy accumulator would push this back up.
        from src.nlhe.subgame_solver import solve_subgame
        tree = self._tree(depth=2)
        res = solve_subgame(tree, _make_ctx(tree, n_iterations=1000))
        self.assertLess(res.converged_l1_tail, 1e-5,
                        f"converged_l1_tail={res.converged_l1_tail:.2e} at K=1000")

    def test_production_k_cost_gate_loop_under_1s(self):
        # Locks that no expensive work (e.g. a network forward) leaks into the loop:
        # the CFR loop itself must stay well under 1s (measured ~0.1s in Stage 3-C).
        import random as _r
        import time
        from src.nlhe.subgame_solver import _build_warmup, _run_cfr
        tree = self._tree(depth=3)
        ctx = _make_ctx(tree, n_iterations=1000)
        cache = _build_warmup(tree, ctx, _r.Random(0))  # warm-up excluded from the gate
        t0 = time.perf_counter()
        _run_cfr(tree, ctx, cache)
        dt = time.perf_counter() - t0
        self.assertLess(dt, 1.0,
                        f"CFR loop took {dt:.3f}s at K=1000 on a {len(tree.all_nodes)}-node tree")


# ============================================================
# Sub-step 4: extract_action (policy extraction)
# ============================================================

def _result_with_policy(p, mask, degraded=False):
    from src.nlhe.subgame_solver import SubgameSolveResult
    p = np.asarray(p, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    return SubgameSolveResult(
        root_policy=p, root_blueprint=p.copy(), legal_mask=mask,
        hero_seat=0, n_iterations=100, n_decision_nodes_cached=1, degraded=degraded)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestExtractActionRealState(unittest.TestCase):
    def setUp(self):
        from src.nlhe.subgame import _discretize_at_decision
        self.state = _first_decision_state(seed=42)
        self.dmap = _discretize_at_decision(self.state)   # {DiscreteAction: chip}
        self.legal_idxs = sorted(int(da) for da in self.dmap)
        self.mask = np.zeros(7, dtype=np.float32)
        for i in self.legal_idxs:
            self.mask[i] = 1.0

    def test_argmax_returns_max_action(self):
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        astar = self.legal_idxs[-1]
        p = np.zeros(7)
        p[astar] = 0.8
        for i in self.legal_idxs:
            if i != astar:
                p[i] = 0.2 / (len(self.legal_idxs) - 1)
        res = _result_with_policy(p, self.mask)
        chip = extract_action(res, self.state, random.Random(0), mode="argmax")
        self.assertEqual(chip, self.dmap[DiscreteAction(astar)])

    def test_sample_converges_to_distribution(self):
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        self.assertIn(0, self.legal_idxs)  # FOLD legal, chip 0
        self.assertIn(1, self.legal_idxs)  # CALL legal, chip 1
        p = np.zeros(7)
        p[0] = 0.6
        p[1] = 0.4
        res = _result_with_policy(p, self.mask)
        rng = random.Random(123)
        N, cnt_fold = 20000, 0
        fold_chip = self.dmap[DiscreteAction.FOLD]
        for _ in range(N):
            if extract_action(res, self.state, rng, mode="sample") == fold_chip:
                cnt_fold += 1
        freq = cnt_fold / N
        self.assertAlmostEqual(freq, 0.6, delta=0.02,
                               msg=f"FOLD frequency {freq:.4f} not within 0.02 of 0.60")

    def test_masked_elements_never_selected(self):
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        illegal = next(i for i in range(7) if i not in self.legal_idxs)
        legal = self.legal_idxs[0]
        # deliberately put 0.3 mass on an ILLEGAL index — extract must mask it away,
        # leaving all weight on `legal` (so every draw returns legal's chip).
        p = np.zeros(7)
        p[legal] = 0.7
        p[illegal] = 0.3
        res = _result_with_policy(p, self.mask)
        rng = random.Random(7)
        legal_chip = self.dmap[DiscreteAction(legal)]
        for _ in range(2000):
            self.assertEqual(extract_action(res, self.state, rng, mode="sample"), legal_chip)
        self.assertEqual(extract_action(res, self.state, random.Random(1), mode="argmax"),
                         legal_chip)

    def test_allin_not_aliased_when_max_bet_positive(self):
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        if int(DiscreteAction.ALLIN) not in self.legal_idxs:
            self.skipTest("ALLIN not legal at this root")
        allin_chip = self.dmap[DiscreteAction.ALLIN]
        self.assertGreater(allin_chip, 0)  # a real all-in (max_bet), not the chip-0 alias
        p = np.zeros(7)
        p[int(DiscreteAction.ALLIN)] = 1.0
        res = _result_with_policy(p, self.mask)
        chip = extract_action(res, self.state, random.Random(0), mode="argmax")
        self.assertEqual(chip, allin_chip)  # NOT remapped to CALL

    def test_degraded_still_returns_legal_action(self):
        from src.nlhe.subgame_solver import extract_action
        p = np.zeros(7)
        for i in self.legal_idxs:
            p[i] = 1.0 / len(self.legal_idxs)
        res = _result_with_policy(p, self.mask, degraded=True)  # degraded -> no fallback here
        chip = extract_action(res, self.state, random.Random(0), mode="sample")
        self.assertIn(chip, set(self.dmap.values()))

    def test_reproducible_same_seed(self):
        from src.nlhe.subgame_solver import extract_action
        p = np.zeros(7)
        for i in self.legal_idxs:
            p[i] = 1.0 / len(self.legal_idxs)
        res = _result_with_policy(p, self.mask)
        seq1 = [extract_action(res, self.state, random.Random(5), mode="sample")
                for _ in range(1)]  # fresh seed each construction -> deterministic
        r1, r2 = random.Random(5), random.Random(5)
        s1 = [extract_action(res, self.state, r1, mode="sample") for _ in range(50)]
        s2 = [extract_action(res, self.state, r2, mode="sample") for _ in range(50)]
        self.assertEqual(s1, s2)


class TestExtractActionAlias(unittest.TestCase):
    """ALLIN->CALL alias on the exact {FOLD:0, CALL:1, ALLIN:0} map (monkeypatched —
    the real no-re-raise-room shove is hard to construct deterministically, and the
    alias is map-driven, so injecting the map is the precise unit test)."""

    def _result_force_allin(self):
        mask = np.zeros(7, dtype=np.float32)
        mask[0] = mask[1] = mask[6] = 1.0  # FOLD, CALL, ALLIN legal
        p = np.zeros(7, dtype=np.float32)
        p[6] = 1.0  # force ALLIN
        return _result_with_policy(p, mask)

    def test_alias_fires_returns_call_not_fold(self):
        from unittest import mock
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        fake = {DiscreteAction.FOLD: 0, DiscreteAction.CALL: 1, DiscreteAction.ALLIN: 0}
        res = self._result_force_allin()
        with mock.patch("src.nlhe.subgame_solver._discretize_at_decision", return_value=fake):
            chip = extract_action(res, object(), random.Random(0), mode="argmax")
        self.assertEqual(chip, 1,
                         "ALLIN in {FOLD:0,CALL:1,ALLIN:0} must alias to CALL(1), not FOLD(0)")

    def test_no_alias_when_allin_nonzero(self):
        from unittest import mock
        from src.nlhe.actions import DiscreteAction
        from src.nlhe.subgame_solver import extract_action
        fake = {DiscreteAction.FOLD: 0, DiscreteAction.CALL: 1, DiscreteAction.ALLIN: 500}
        res = self._result_force_allin()
        with mock.patch("src.nlhe.subgame_solver._discretize_at_decision", return_value=fake):
            chip = extract_action(res, object(), random.Random(0), mode="argmax")
        self.assertEqual(chip, 500)


if __name__ == "__main__":
    unittest.main()
