"""Tests for src.nlhe.fast_view (B1c sub-step 2, Stage A).

The fast path MUST produce field-by-field-identical output to the canonical
cfr6._build_view_6max + actions.discretize_legal_actions on every decision state.
These tests assert that exact equality across 50 random production-game rollouts
plus hand-picked edge cases, verify the chance/terminal refusal contract, and
benchmark the fast path against the ≤0.30 ms/step gate.

All tests use the production game pyspiel.load_game(six_max_sng(...)), never a
hand-rolled gamedef.
"""
from __future__ import annotations

import random
import time
import unittest


def _open_spiel_available() -> bool:
    try:
        import pyspiel  # noqa: F401
        return True
    except Exception:
        return False


_HAS_OPEN_SPIEL = _open_spiel_available()

_STARTING_STACK = 10000


def _game():
    import pyspiel
    from src.nlhe.game_strings import six_max_sng
    return pyspiel.load_game(six_max_sng(starting_stack=_STARTING_STACK))


def _advance_past_chance(state, rng):
    while state.is_chance_node():
        a, p = zip(*state.chance_outcomes())
        state = state.child(int(rng.choices(a, weights=p, k=1)[0]))
    return state


def _first_decision(seed):
    rng = random.Random(seed)
    return _advance_past_chance(_game().new_initial_state(), rng)


def _canonical(state):
    """(view, discrete_to_chip) from the canonical path for a decision state."""
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.actions import discretize_legal_actions
    parsed = parse_state_6max(state)
    view = _build_view_6max(state, parsed)
    disc = discretize_legal_actions(list(state.legal_actions()), view)
    return parsed, view, disc


def _walk_rollout_decision_states(seed, max_decisions=12, max_steps=200):
    """Yield decision states from a random rollout (chance sampled, varied actions).

    Action selection draws a random DiscreteAction from the canonical discretized
    set so the walk hits realistic states — folds, calls, bets, and all-ins
    (the last produces facing-all-in states with the ALLIN/FOLD chip-0 alias).
    """
    rng = random.Random(seed)
    state = _game().new_initial_state()
    states = []
    decisions = 0
    steps = 0
    while (not state.is_terminal()) and decisions < max_decisions and steps < max_steps:
        steps += 1
        if state.is_chance_node():
            a, p = zip(*state.chance_outcomes())
            state = state.child(int(rng.choices(a, weights=p, k=1)[0]))
            continue
        states.append(state)
        decisions += 1
        _parsed, _view, disc = _canonical(state)
        da = rng.choice(list(disc.keys()))
        state = state.child(int(disc[da]))
    return states


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestFastViewEqualsCanonical(unittest.TestCase):
    """Field-by-field equality with the canonical view + discretize."""

    def test_view_equals_canonical_over_50_rollouts(self):
        from src.nlhe.fast_view import fast_build_view
        total_decisions = 0
        deep_rollouts = 0
        for seed in range(50):
            states = _walk_rollout_decision_states(seed)
            if len(states) >= 4:
                deep_rollouts += 1
            for st in states:
                parsed, view, _ = _canonical(st)
                fview = fast_build_view(parsed, st.legal_actions())
                # GameStateView is a frozen dataclass -> structural equality.
                self.assertEqual(
                    fview, view,
                    f"view mismatch seed={seed}: fast={fview} canonical={view}")
                total_decisions += 1
        # Sanity: we actually exercised a substantial, depth>=4 surface.
        self.assertGreater(total_decisions, 200)
        self.assertGreaterEqual(deep_rollouts, 25)

    def test_discretize_equals_canonical_over_50_rollouts(self):
        from src.nlhe.fast_view import fast_discretize
        from src.nlhe.actions import discretize_legal_actions
        n = 0
        for seed in range(50):
            for st in _walk_rollout_decision_states(seed):
                _parsed, view, disc = _canonical(st)
                # Same view input to both -> isolates the discretize logic.
                fdisc = fast_discretize(st.legal_actions(), view)
                self.assertEqual(
                    fdisc, disc,
                    f"discretize mismatch seed={seed}: fast={fdisc} canon={disc}")
                n += 1
        self.assertGreater(n, 200)

    def test_combined_entry_point_equals_canonical(self):
        from src.nlhe.fast_view import fast_view_and_discretize
        for seed in range(50):
            for st in _walk_rollout_decision_states(seed):
                parsed, view, disc = _canonical(st)
                fview, fdisc, legal = fast_view_and_discretize(st, parsed)
                self.assertEqual(fview, view)
                self.assertEqual(fdisc, disc)
                self.assertEqual(list(legal), list(st.legal_actions()))


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestFastViewEdgeCases(unittest.TestCase):
    """Hand-picked edge cases called out in the Stage A spec."""

    def _assert_equal_at(self, state):
        from src.nlhe.fast_view import fast_build_view, fast_discretize
        from src.nlhe.actions import discretize_legal_actions
        parsed, view, disc = _canonical(state)
        self.assertEqual(fast_build_view(parsed, state.legal_actions()), view)
        self.assertEqual(fast_discretize(state.legal_actions(), view), disc)

    def test_preflop_six_active(self):
        # First decision: all 6 seats live, full action set.
        self._assert_equal_at(_first_decision(42))

    def test_facing_all_in_chip0_alias(self):
        # UTG shoves; next actor faces an all-in with no re-raise room. The
        # canonical discretizer returns {FOLD:0, CALL:1, ALLIN:0} (the chip-0
        # alias documented in b2dded5) — the fast path must reproduce it exactly.
        from src.nlhe.actions import DiscreteAction
        s = _first_decision(42)
        _p, _v, disc0 = _canonical(s)
        allin_chip = disc0[DiscreteAction.ALLIN]
        shoved = _advance_past_chance(s.child(int(allin_chip)), random.Random(1))
        _p, _v, disc = _canonical(shoved)
        # confirm this really is the restricted alias state
        self.assertEqual(disc.get(DiscreteAction.ALLIN), 0)
        self.assertIn(DiscreteAction.FOLD, disc)
        self.assertIn(DiscreteAction.CALL, disc)
        self._assert_equal_at(shoved)

    def test_restricted_and_minimal_legal_sets(self):
        # Scan rollouts for decision states with small legal sets (<=2 raw
        # actions, e.g. facing all-in) and assert equality there too.
        found = 0
        for seed in range(60):
            for st in _walk_rollout_decision_states(seed):
                if len(st.legal_actions()) <= 2:
                    self._assert_equal_at(st)
                    found += 1
        self.assertGreater(found, 0, "expected some restricted-legal states")

    def test_terminal_adjacent_decision_state(self):
        # A decision state one fold away from ending the hand.
        s = _first_decision(42)
        guard = 0
        while not s.child(0).is_terminal():
            s = s.child(0)
            guard += 1
            self.assertLess(guard, 6)
        self.assertFalse(s.is_terminal())
        self._assert_equal_at(s)


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestFastViewRefusesNonDecision(unittest.TestCase):
    """Decision-nodes-only contract: chance and terminal states raise."""

    def test_raises_on_chance_node(self):
        from src.nlhe.fast_view import fast_view_and_discretize
        state = _game().new_initial_state()  # initial deal = chance node
        self.assertTrue(state.is_chance_node())
        with self.assertRaises(ValueError):
            fast_view_and_discretize(state, parsed={})

    def test_raises_on_terminal(self):
        from src.nlhe.fast_view import fast_view_and_discretize
        # Fold the hand to completion.
        s = _first_decision(42)
        guard = 0
        while not s.is_terminal():
            s = s.child(0)
            guard += 1
            self.assertLess(guard, 8)
        self.assertTrue(s.is_terminal())
        with self.assertRaises(ValueError):
            fast_view_and_discretize(s, parsed={})


# ============================================================
# Benchmark (≤ 0.30 ms/step gate)
# ============================================================

def _benchmark(n_iters=10000, n_states=24, report=print):
    """Benchmark fast vs canonical view+discretize on CACHED state objects.

    'Cached' = the same Python state objects reused across all iterations (no
    re-construction / re-deal inside the timing loop). Also reports a no-op
    baseline (empty loop over the same state list) so the view/discretize cost
    is isolated from loop overhead. Uses min-per-step (robust to CPU preemption
    on the oversubscribed host — noise only adds time, so the minimum is closest
    to true compute cost).

    Returns dict of per-step ms: canonical_min/mean, fast_min/mean, noop_min.
    """
    from src.nlhe.infoset6 import parse_state_6max
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.actions import discretize_legal_actions
    from src.nlhe.fast_view import fast_view_and_discretize

    # Build a cached set of representative DECISION states once.
    states = []
    seed = 0
    while len(states) < n_states:
        for st in _walk_rollout_decision_states(seed):
            states.append(st)
            if len(states) >= n_states:
                break
        seed += 1
    states = states[:n_states]
    ns = len(states)

    # Pre-parse once per state (both paths need parsed; parse is not what we
    # are optimizing, so keep it out of the inner timing to isolate view+disc).
    parsed_by_state = [parse_state_6max(st) for st in states]

    def canonical_step(i):
        st = states[i]
        v = _build_view_6max(st, parsed_by_state[i])
        return discretize_legal_actions(list(st.legal_actions()), v)

    def fast_step(i):
        st = states[i]
        return fast_view_and_discretize(st, parsed_by_state[i])

    def noop_step(i):
        return states[i]

    # warm-up
    for i in range(ns):
        canonical_step(i); fast_step(i); noop_step(i)

    def run(fn):
        # several batches; report min and mean per-step
        batch = max(1, n_iters // 10)
        per_step = []
        for _ in range(10):
            t0 = time.perf_counter()
            for k in range(batch):
                fn(k % ns)
            per_step.append(1000.0 * (time.perf_counter() - t0) / batch)
        return min(per_step), sum(per_step) / len(per_step)

    can_min, can_mean = run(canonical_step)
    fast_min, fast_mean = run(fast_step)
    noop_min, _ = run(noop_step)

    res = {
        "n_states": ns,
        "canonical_min": can_min, "canonical_mean": can_mean,
        "fast_min": fast_min, "fast_mean": fast_mean,
        "noop_min": noop_min,
    }
    if report:
        report(f"  cached decision states     : {ns}")
        report(f"  no-op baseline (min)       : {noop_min:.5f} ms/step")
        report(f"  canonical view+disc (min)  : {can_min:.4f} ms/step  (mean {can_mean:.4f})")
        report(f"  FAST view+disc (min)       : {fast_min:.4f} ms/step  (mean {fast_mean:.4f})")
        report(f"  speedup (min)              : {can_min / fast_min:.1f}x")
    return res


@unittest.skipUnless(_HAS_OPEN_SPIEL, "Requires open_spiel")
class TestFastViewBenchmark(unittest.TestCase):
    def test_benchmark_gate_0_30ms(self):
        res = _benchmark(n_iters=10000, report=None)
        self.assertLessEqual(
            res["fast_min"], 0.30,
            f"fast path {res['fast_min']:.4f} ms/step exceeds 0.30 ms gate")


if __name__ == "__main__":
    print("=== fast_view benchmark ===")
    _benchmark()
    unittest.main()
