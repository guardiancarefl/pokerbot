"""Layer 4 / C1d-1 — bias_factory adapter from a live MatchObserver.

Produces a closure suitable for passing as SubgamePolicy(bias_factory=...).
The closure reads stats from a shared MatchObserver and calls the chosen
C1b builder per opponent seat.

Two paths supported, selected by `path` argument:
  - "raw":       stats_to_bias_configs_raw — context-free, just needs stats.
  - "archetype": stats_to_bias_configs_archetype — requires a per-call
                 leaf_context_resolver to supply parsed/state/bucket_id/
                 in_position. bucket_id is NOT available at the
                 _bias_dist_fn dispatch point (Recon 1.3) — the caller must
                 derive it from the abstraction at resolve time.

Bit-identity contract: at observer.confidence(seat) == 0 BOTH paths return
all-ones BiasConfigs (locked by C1b's confidence-zero tests), which
SubgamePolicy's build_per_seat_biased_blueprints detects as the identity
short-circuit and discards (returns None to LeafEvalContext). Net effect:
passing this factory instead of None preserves baseline behavior until the
observer accumulates evidence. Safe drop-in.

Path: docs/scratch/session_handoff/layer4_decisions_locked.md §6-Q3 (both
paths permitted in C1b; this module is the integration glue for C1d-2's
ablation comparison).
"""
from __future__ import annotations

from typing import Callable, Literal, Optional

from src.nlhe.bias_configs import (
    ALPHA_C1_DEFAULT,
    stats_to_bias_configs_archetype,
    stats_to_bias_configs_raw,
)
from src.nlhe.biased_policy import BiasConfig
from src.nlhe.within_match import MatchObserver


PathName = Literal["raw", "archetype"]


def make_bias_factory(
    observer: MatchObserver,
    path: PathName,
    alpha: float = ALPHA_C1_DEFAULT,
    leaf_context_resolver: Optional[Callable[[int], dict]] = None,
) -> Callable[[int], list[BiasConfig]]:
    """Build a bias_factory closure for SubgamePolicy.

    Parameters
    ----------
    observer : MatchObserver
        The live observer fed by the eval loop. The factory READS from this
        observer on every invocation — observations made before the call are
        reflected in the next factory output (the "live read" invariant tested
        by test_factory_reads_observer_live).
    path : {"raw", "archetype"}
        Which C1b builder to invoke.
    alpha : float, default = ALPHA_C1_DEFAULT (2.0)
        The per-action multiplier clip bound, locked §5-Q1.
    leaf_context_resolver : Optional[Callable[[int], dict]]
        REQUIRED when path == "archetype"; IGNORED when path == "raw". A
        callable `resolver(seat) -> dict` that supplies the per-seat leaf
        context the archetype builder needs. The returned dict must contain
        keys: 'parsed', 'state', 'bucket_id', 'in_position', 'calibration'.
        `calibration` is the EquityCalibration matching the abstraction in
        use; it is per-abstraction (not per-seat), but flows through the
        resolver so a single call site owns the abstraction binding. C1d-2
        will decide how to derive bucket_id at the call site (the §6-Q3
        sub-decision: caller-supplied vs uniform marginalization vs
        abstraction-mean conditioning is an empirical bake-off).

    Returns
    -------
    Callable[[int], list[BiasConfig]]
        factory(seat) -> list of length 4 (the k=4 menu). Pass to
        SubgamePolicy(bias_factory=...).

    Raises
    ------
    ValueError
        If `path` is not 'raw' or 'archetype', or if `path == "archetype"`
        is requested without a `leaf_context_resolver`.
    """
    if path not in ("raw", "archetype"):
        raise ValueError(
            f"path must be 'raw' or 'archetype', got {path!r}"
        )
    if path == "archetype" and leaf_context_resolver is None:
        raise ValueError(
            "leaf_context_resolver is required when path='archetype'; "
            "provide a callable(seat) -> dict with keys "
            "{'parsed', 'state', 'bucket_id', 'in_position', 'calibration'}"
        )

    if path == "raw":
        def factory_raw(seat: int) -> list[BiasConfig]:
            stats = observer.get_stats(seat)
            conf = observer.confidence(seat)
            return stats_to_bias_configs_raw(stats, conf, alpha=alpha)
        return factory_raw

    # path == "archetype"
    resolver = leaf_context_resolver  # capture for the closure

    def factory_archetype(seat: int) -> list[BiasConfig]:
        stats = observer.get_stats(seat)
        conf = observer.confidence(seat)
        ctx = resolver(seat)
        return stats_to_bias_configs_archetype(
            stats, conf,
            parsed=ctx["parsed"],
            state=ctx["state"],
            bucket_id=ctx["bucket_id"],
            in_position=ctx["in_position"],
            calibration=ctx.get("calibration"),
            alpha=alpha,
        )
    return factory_archetype
