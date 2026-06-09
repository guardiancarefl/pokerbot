"""Unit tests for DecisionCache (decide-once / commit-to-one-action).

DecisionCache locks in ONE sample per actual decision so the polling-frame
re-sample lottery (live dryrun 2026-06-08 surfaced shove-vs-call drift on
the same hand+street+spot) becomes impossible. These tests pin the identity
discriminator down — every "what changed" axis the user called out is
exercised against the same-decision (cache hit) baseline.
"""
from __future__ import annotations

import pytest

from src.nlhe.integration.live_loop import DecisionCache
from src.nlhe.integration.scraper_schema import BlindsLevel, ScraperFrame


LV2 = BlindsLevel(sb=25, bb=50, ante=10)
LV3 = BlindsLevel(sb=50, bb=100, ante=15)


def _frame(*, dealer_seat: int = 2, hero_seat: int = 0,
            hero_cards: tuple = ("9d", "Kc"),
            board: tuple = (),
            stack: tuple = (1500,) * 6,
            bet: tuple = (0,) * 6,
            alive: tuple | None = None,
            blinds: BlindsLevel = LV2,
            pot_total: int = 75) -> ScraperFrame:
    if alive is None:
        alive = tuple(s > 0 for s in stack)
    max_opp_bet = max(
        (bet[i] for i in range(6) if i != hero_seat and alive[i]),
        default=0,
    )
    return ScraperFrame(
        captured_at="test",
        blinds=blinds,
        dealer_seat=dealer_seat,
        hero_seat=hero_seat,
        hero_cards=hero_cards,
        board=board,
        stack=stack,
        bet=bet,
        folded=(False,) * 6,
        empty=tuple(not a for a in alive),
        alive=alive,
        pot_total=pot_total,
        controls_present=True,
        hero_facing_bet=max_opp_bet > bet[hero_seat],
    )


def test_same_frame_twice_is_same_identity():
    """Re-sending the same frame must NOT count as a new decision."""
    c = DecisionCache()
    f1 = _frame()
    f2 = _frame()  # same content, second polling capture
    assert c.identity(f1) == c.identity(f2)


def test_put_then_get_returns_cached():
    c = DecisionCache()
    f = _frame()
    key = c.identity(f)
    assert c.get(key) is None
    c.put(key, {"kind": "fold", "chip_amount": None,
                 "raw_openspiel_chip_int": 0}, 0)
    cached = c.get(key)
    assert cached is not None
    action, chip_int = cached
    assert action["kind"] == "fold"
    assert chip_int == 0


def test_pot_drift_alone_is_not_new_decision():
    """Pot UI lag (or scraper collection lag) changes pot_total but not the
    decision. The cache must not key on pot_total."""
    c = DecisionCache()
    f1 = _frame(pot_total=75)
    f2 = _frame(pot_total=85)  # pot ticked up due to UI lag
    assert c.identity(f1) == c.identity(f2)


def test_hero_stack_drift_alone_is_not_new_decision():
    """A scraper-side stack refresh tick mustn't invalidate the cache."""
    c = DecisionCache()
    f1 = _frame(stack=(1500, 1500, 1500, 1500, 1500, 1500))
    f2 = _frame(stack=(1495, 1500, 1500, 1500, 1500, 1500))  # micro drift on hero
    assert c.identity(f1) == c.identity(f2)


def test_opponent_call_without_raise_does_not_change_max_opp_bet():
    """If an opponent CALLS hero's previous size, the max-opp-bet is
    unchanged. (In practice this frame wouldn't be hero-to-act anyway —
    but the cache must not falsely re-decide on this transition.)"""
    c = DecisionCache()
    # Initial: hero faces a 50 open from seat 3.
    f1 = _frame(bet=(0, 0, 0, 50, 0, 0))
    # Opponent in seat 5 calls — bet[5]=50, max_opp_bet still 50.
    f2 = _frame(bet=(0, 0, 0, 50, 0, 50))
    assert c.identity(f1) == c.identity(f2)


def test_opponent_raise_is_new_decision():
    """A re-raise puts hero back in a fresh decision — must re-sample."""
    c = DecisionCache()
    f1 = _frame(bet=(0, 0, 0, 50, 0, 0))   # facing 50
    f2 = _frame(bet=(0, 0, 0, 50, 0, 200)) # seat 5 re-raised to 200
    assert c.identity(f1) != c.identity(f2)


def test_board_card_added_is_new_decision():
    """New street = new decision."""
    c = DecisionCache()
    f1 = _frame(board=("Jc", "5h", "2d"))           # flop
    f2 = _frame(board=("Jc", "5h", "2d", "7s"))     # turn
    assert c.identity(f1) != c.identity(f2)


def test_new_hand_dealer_moved_is_new_decision():
    """Button moves → new hand → new decision."""
    c = DecisionCache()
    f1 = _frame(dealer_seat=2)
    f2 = _frame(dealer_seat=3)
    assert c.identity(f1) != c.identity(f2)


def test_new_hand_hero_cards_changed_is_new_decision():
    """New hole cards = new hand even if dealer briefly mis-reads."""
    c = DecisionCache()
    f1 = _frame(hero_cards=("9d", "Kc"))
    f2 = _frame(hero_cards=("Ah", "Jh"))
    assert c.identity(f1) != c.identity(f2)


def test_level_change_is_new_decision():
    """Blinds escalate mid-frame-stream → new decision identity."""
    c = DecisionCache()
    f1 = _frame(blinds=LV2)
    f2 = _frame(blinds=LV3)
    assert c.identity(f1) != c.identity(f2)


def test_player_busted_is_new_decision():
    """A seat going from alive→busted means we're in a new hand or the
    seat count changed mid-hand (impossible mid-hand without a new deal)."""
    c = DecisionCache()
    f1 = _frame(stack=(1500, 1500, 1500, 1500, 1500, 1500))
    f2 = _frame(stack=(1500, 1500, 1500, 1500, 1500, 0))  # seat 5 busted
    assert c.identity(f1) != c.identity(f2)


def test_put_overwrites_prior_identity():
    """The cache holds at most one decision; a new identity's put replaces
    the previous identity entirely (no stale-key resurrection)."""
    c = DecisionCache()
    f1 = _frame()
    k1 = c.identity(f1)
    c.put(k1, {"kind": "fold", "chip_amount": None,
                "raw_openspiel_chip_int": 0}, 0)

    f2 = _frame(board=("Jc", "5h", "2d"))  # new street, new identity
    k2 = c.identity(f2)
    c.put(k2, {"kind": "raise_to", "chip_amount": 100,
                "raw_openspiel_chip_int": 100}, 100)

    # Old identity no longer cached.
    assert c.get(k1) is None
    # New identity returns its own action.
    assert c.get(k2)[0]["kind"] == "raise_to"


def test_seq_352_353_symptom_is_one_decision():
    """Regression for live dryrun 2026-06-08 seq=352/353: hero 9d Kc, pot
    410, facing bet → frame 352 said RAISE_TO 3058 (43.6×BB), frame 353
    said CALL. Same decision, different samples. With the cache, the
    second frame returns cached."""
    c = DecisionCache()
    # Construct same identity for both frames (just like the live capture
    # where two polling captures of the SAME spot were taken).
    f_352 = _frame(
        hero_cards=("9d", "Kc"), board=(),
        dealer_seat=3, hero_seat=0,
        bet=(0, 0, 0, 0, 0, 75),   # seat 5 has bet 75 (faced by hero)
        pot_total=410,
    )
    f_353 = _frame(
        hero_cards=("9d", "Kc"), board=(),
        dealer_seat=3, hero_seat=0,
        bet=(0, 0, 0, 0, 0, 75),
        pot_total=410,
    )
    k = c.identity(f_352)
    assert k == c.identity(f_353)
    # Caller samples and stores once on f_352:
    c.put(k, {"kind": "call", "chip_amount": None,
               "raw_openspiel_chip_int": 1}, 1)
    # f_353 is the SAME decision → cache hit → no resample.
    assert c.get(c.identity(f_353)) is not None
