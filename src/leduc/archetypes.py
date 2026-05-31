"""Step-1 Leduc archetypes for the Family-2 adaptive-policy proof.

Six rule-based opponents spanning the exploitable axes (folder, calling
station, maniac, over-folder-to-aggression, over-caller, uniform noise floor),
each with a STRENGTH KNOB that blends pure archetype with uniform-random
over legal actions. A (name, strength) cell is the unit; default grid =
6 archetypes x {0.5, 1.0} = 12 cells.

These cells are BOTH the phase-2 training opponents AND the eval opponents.
To make the Step-4 pass bar measure generalization (not memorization), the
registry exposes train/test partitions via tags: see HELD_OUT_DEFAULT and
train_test_split() below. The default holds out the weak (strength=0.5)
version of three behaviorally-distinct archetypes (always_fold, over_folder,
over_caller); their strong forms remain in train, so the model must
interpolate along the strength axis for opponents whose pure form it has
seen, never having seen the noisy form. random_uniform's strength knob is
degenerate (uniform blended with uniform = uniform) so it is intentionally
NOT in the held-out default — both random_uniform cells stay in train.

Action IDs (OpenSpiel leduc_poker, verified 2026-05-31):
    0 = Fold   (only legal when facing a bet)
    1 = Call   (= Check when no bet)
    2 = Raise  (legal unless the per-street raise cap is reached)

Each archetype is a callable:
    fn(state) -> {action_id: prob}  over state.legal_actions().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

FOLD, CALL, RAISE = 0, 1, 2

PolicyFn = Callable[[object], Dict[int, float]]


# --------------------------------------------------------------------------
# Pure archetype primitives (strength=1.0 behavior).
# --------------------------------------------------------------------------

def _always_fold(state) -> Dict[int, float]:
    """Folds whenever legal; checks when no bet faced. Maximally exploitable
    by aggression — the bot should learn to bet/raise relentlessly."""
    legal = state.legal_actions()
    if FOLD in legal:
        return {FOLD: 1.0}
    return {CALL: 1.0}  # no bet to face -> check


def _always_call(state) -> Dict[int, float]:
    """Calling station: never folds, never raises. Exploit by value-betting
    thin and never bluffing."""
    return {CALL: 1.0}


def _always_raise(state) -> Dict[int, float]:
    """Maniac: raises whenever legal; calls if the raise cap is hit. Exploit
    by tightening fold range and trapping with strong holdings."""
    legal = state.legal_actions()
    if RAISE in legal:
        return {RAISE: 1.0}
    return {CALL: 1.0}


def _over_folder(state, fold_rate: float = 0.7) -> Dict[int, float]:
    """Folds to aggression at a high rate; otherwise calls. Never raises.
    Exploit by betting wider for fold equity."""
    legal = state.legal_actions()
    if FOLD in legal:
        return {FOLD: fold_rate, CALL: 1.0 - fold_rate}
    return {CALL: 1.0}


def _over_caller(state, fold_rate: float = 0.10,
                 raise_rate: float = 0.05) -> Dict[int, float]:
    """Calls too wide: low fold rate to aggression, rarely raises even with
    strong holdings. Exploit by value-betting thinner and bluffing less."""
    legal = state.legal_actions()
    probs: Dict[int, float] = {a: 0.0 for a in legal}
    if FOLD in legal:
        probs[FOLD] = fold_rate
    if RAISE in legal:
        probs[RAISE] = raise_rate
    probs[CALL] = 1.0 - sum(p for a, p in probs.items() if a != CALL)
    return probs


def _random_uniform(state) -> Dict[int, float]:
    """Uniform over legal actions: the noise floor. Strength knob is
    degenerate here (uniform blended with uniform is still uniform)."""
    legal = state.legal_actions()
    p = 1.0 / len(legal)
    return {a: p for a in legal}


PURE_ARCHETYPES: Dict[str, PolicyFn] = {
    "always_fold":     _always_fold,
    "always_call":     _always_call,
    "always_raise":    _always_raise,
    "over_folder":     _over_folder,
    "over_caller":     _over_caller,
    "random_uniform":  _random_uniform,
}


# --------------------------------------------------------------------------
# Strength knob: convex mixture of pure archetype with uniform-over-legal.
# --------------------------------------------------------------------------

def with_strength(pure_fn: PolicyFn, strength: float) -> PolicyFn:
    """Blend a pure archetype with uniform-random over legal actions.

        strength = 1.0  -> pure archetype
        strength = 0.0  -> pure uniform (== random_uniform)
        otherwise       -> prob(a) = strength * pure(a) + (1 - strength) / |legal|

    The blend preserves the legal-action support; probs sum to 1.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    def blended(state) -> Dict[int, float]:
        legal = state.legal_actions()
        u = 1.0 / len(legal)
        pure = pure_fn(state)
        return {
            a: strength * pure.get(a, 0.0) + (1.0 - strength) * u
            for a in legal
        }
    return blended


# --------------------------------------------------------------------------
# Registry: ArchetypeCell (name, strength) -> callable.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchetypeCell:
    """A single (archetype, strength) point in the grid. Hashable + JSON-friendly
    via .id and (name, strength)."""
    name: str          # one of PURE_ARCHETYPES keys
    strength: float    # in [0, 1]

    @property
    def id(self) -> str:
        return f"{self.name}@s{self.strength:.2f}"


# Default grid: 6 archetypes x 2 strength levels = 12 cells.
DEFAULT_STRENGTHS: Tuple[float, ...] = (0.5, 1.0)


def default_grid() -> Tuple[ArchetypeCell, ...]:
    """The 12-cell default grid (6 archetypes x strengths {0.5, 1.0})."""
    return tuple(
        ArchetypeCell(name=n, strength=s)
        for n in PURE_ARCHETYPES
        for s in DEFAULT_STRENGTHS
    )


# Held-out partition (Step-4 generalization-vs-memorization design).
#
# Holds out the WEAK (strength=0.5) cell of three behaviorally-distinct
# archetypes. The model trains on their STRONG (s=1.0) form (and on all forms
# of the other three archetypes) and must still beat the weak form it never
# saw in training. This tests "can the adaptive policy interpolate along the
# strength axis for a known archetype?" — a generalization test, not an
# extrapolation-to-novel-rule test.
#
# random_uniform is intentionally NOT held out: its strength knob is
# degenerate (uniform blended with uniform = uniform), so holding s=0.5 out
# would not be a real generalization test (the train cell is identical).
HELD_OUT_DEFAULT: Tuple[Tuple[str, float], ...] = (
    ("always_fold",  0.5),
    ("over_folder",  0.5),
    ("over_caller",  0.5),
)


def train_test_split(
    grid: Iterable[ArchetypeCell] = None,
    held_out: Iterable[Tuple[str, float]] = None,
) -> Tuple[Tuple[ArchetypeCell, ...], Tuple[ArchetypeCell, ...]]:
    """Partition `grid` into (train_cells, test_cells).

    Default: split default_grid() using HELD_OUT_DEFAULT (3 held-out cells,
    9 train cells). Both arguments are overrideable so a Step-3 training
    config can drive a custom partition without a code change.
    """
    grid = tuple(grid) if grid is not None else default_grid()
    held = set(held_out) if held_out is not None else set(HELD_OUT_DEFAULT)
    train, test = [], []
    for c in grid:
        (test if (c.name, c.strength) in held else train).append(c)
    return tuple(train), tuple(test)


def build_cell(cell: ArchetypeCell) -> PolicyFn:
    """Materialize an ArchetypeCell into its action_probabilities callable."""
    if cell.name not in PURE_ARCHETYPES:
        raise KeyError(f"unknown archetype {cell.name!r}; "
                       f"valid: {sorted(PURE_ARCHETYPES)}")
    return with_strength(PURE_ARCHETYPES[cell.name], cell.strength)
