"""Regression tests for the postflop emission cs_folded fix (seq=101,
2026-06-09).

The fix replaces `pf_alive_post.remove(seat)` with `cs_folded.add(seat)`
in the postflop emission loop, mirroring the preflop loop's
`folded_emitted` pattern. The prior remove-while-iterating pattern
shifted the cycle's modular index and caused the loop to skip past
hero (and any seat between the removed position and hero) on its next
visit — producing both a fabricated downstream fold AND a missed
hero-stop.

Class of bug: ANY postflop scenario where a seat defensive-folds at
cycle position k AND hero is at position k+1 or later. seat1-zone-fix
unlocked dealer=BTN frames where hero is at the END of the cycle,
making this defect surface; the fix is general (not seat1-specific).
"""
from __future__ import annotations

import pyspiel
import pytest

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.replay import replay_to_decision
from src.nlhe.integration.scraper_schema import (
    BlindsLevel, ScraperFrame, derive_action_sequence,
)


STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
LV1 = BlindsLevel(sb=15, bb=25, ante=5)


def _frame(*, dealer_seat, hero_seat, hero_cards, stack, bet,
            blinds=LV1, board=(), folded=None, empty=None,
            pot_total=None, controls_present=True, captured_at="test"):
    if folded is None:
        folded = (False,) * 6
    if empty is None:
        empty = tuple(stack[i] == 0 and bet[i] == 0 for i in range(6))
    alive = tuple(
        (not empty[i]) and (stack[i] > 0 or bet[i] > 0)
        for i in range(6)
    )
    if pot_total is None:
        pot_total = sum(bet) + sum(blinds.ante for _ in alive if _)
    max_opp_bet = max(
        (bet[i] for i in range(6) if i != hero_seat and alive[i]),
        default=0,
    )
    hero_facing_bet = alive[hero_seat] and max_opp_bet > bet[hero_seat]
    return ScraperFrame(
        captured_at=captured_at, blinds=blinds,
        dealer_seat=dealer_seat, hero_seat=hero_seat,
        hero_cards=hero_cards, board=board,
        stack=stack, bet=bet, folded=folded, empty=empty, alive=alive,
        pot_total=pot_total, controls_present=controls_present,
        hero_facing_bet=hero_facing_bet,
    )


# --------------------------------------------------------------------------
# Test 1: seq=101 — hero=BTN, single defensive fold (CO) before hero
# --------------------------------------------------------------------------

def test_seq101_hero_btn_single_defensive_fold_reconstructs():
    """seq=101 from live_dryrun_20260609_192543.jsonl: dealer=seat1=hero
    on button, 4-handed flop (UTG/MP folded preflop, SB+BB+CO+BTN
    limped). BB bets 98 on flop; CO defensive-folds; hero (BTN) to-act.

    Pre-fix: postflop loop emitted (5,0) CO fold then wrapped past
    hero to emit a spurious (1,0) SB fold (10 actions total); replay
    failed with "action_seq[9] expects seat 1 but state.current_player()=0".

    Post-fix: 9-action sequence stopping cleanly at hero. Both the
    spurious SB fold is gone AND hero-stop fires at the right point.
    """
    pre = (1713, 1545, 1425, 1327, 1425, 1565)
    frame = _frame(
        dealer_seat=0, hero_seat=0,
        hero_cards=("Qd", "9h"),
        stack=(1683, 1515, 1297, 1322, 1420, 1535),
        bet=(0, 0, 98, 0, 0, 0),
        board=("3c", "Ks", "2s"),
        pot_total=228,
        empty=(False,) * 6,
        captured_at="seq101_synthetic",
    )
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    expected = [
        (3, 0), (4, 0),               # UTG fold, MP fold
        (5, 1), (0, 1), (1, 1), (2, 1),  # CO limp, BTN limp, SB call, BB check
        (1, 1), (2, 123), (5, 0),     # flop: SB check, BB bet (123 = 25+98), CO fold
                                       # STOP at hero (BTN)
    ]
    assert actions == expected, (
        f"expected {expected};\n"
        f"got      {actions};\n"
        f"pre-fix would have emitted an extra (1,0) SB fold at the end"
    )
    # End-to-end replay + invariant
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    pack = replay_to_decision(frame, structure, pre_hand_override=pre)
    inv = check_mid_hand_invariant(frame, pack)
    assert inv.ok, f"invariant FAILED; deltas={inv.deltas}"


# --------------------------------------------------------------------------
# Test 2: hero=CO with a fold before hero in the postflop walk
# --------------------------------------------------------------------------

def test_hero_co_with_earlier_fold_reconstructs():
    """6-handed flop where hero is CO (idx 4) and a seat earlier in
    the postflop walk defensive-folds. Postflop walk: SB → BB → UTG →
    MP → CO(hero) → BTN. SB checks, BB bets, UTG folds defensively.
    Pre-fix: removing UTG would shift the cycle index and skip MP,
    landing on CO/hero without the MP fold emitted. Post-fix: UTG
    marked in cs_folded, MP gets its visit, then CO/hero stops.
    """
    pre = (1500, 1500, 1500, 1500, 1500, 1500)
    # All 6 limped preflop to 25, BB checked. Flop: SB check (0),
    # BB bets 100 (= 100 chips on flop), UTG faces 100 with 0 stack
    # commitment voluntary → defensive-fold, MP faces 100 → MP also
    # folds, CO (hero) to act.
    #
    # For the test, hero (CO) is mid-decision: bet=0, facing the
    # 100 from BB. UTG and MP show bet=0 too (chip-equality repair
    # marks them via fold detection? Actually they limped preflop
    # so their preflop deduction equals others; the simple-model
    # path treats them as alive non-folded by default. The postflop
    # defensive-fold branch handles their flop fold.)
    # All 6 limped preflop = lose 30 each (ante 5 + call 25). Flop:
    # BB(idx 1) bets 100, lowering BB stack to 1370. Others' stacks
    # stay at 1470. UTG(idx 2), MP(idx 3) defensive-fold on flop;
    # CO(idx 4=hero) to act facing 100.
    frame = _frame(
        dealer_seat=5, hero_seat=4,            # dealer=BTN(5), hero=CO(4)
        hero_cards=("Ah", "Kh"),
        stack=(1470, 1370, 1470, 1470, 1470, 1470),  # BB lower by 100
        bet=(0, 100, 0, 0, 0, 0),              # BB(idx 1) bet 100
        board=("2c", "7h", "Jd"),
        # Pot: 6*ante(5) + 6*limp(25) + BB flop bet(100) = 30+150+100 = 280
        pot_total=280,
        empty=(False,) * 6,
        captured_at="hero_co_earlier_fold_synthetic",
    )
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    # Expected preflop: all 6 limp 25. UTG=2, MP=3, CO=4, BTN=5, SB=0, BB=1.
    # Preflop order from dealer=5: UTG(2)→MP(3)→CO(4)→BTN(5)→SB(0)→BB(1).
    # Postflop order from SB: SB(0)→BB(1)→UTG(2)→MP(3)→CO(4)→BTN(5).
    # Flop actions: SB check, BB bet 100, UTG fold, MP fold, STOP at CO/hero.
    # (chip_int for BB bet = preflop_max_chip_int + 100 = 25 + 100 = 125)
    expected = [
        (2, 1), (3, 1), (4, 1), (5, 1), (0, 1), (1, 1),  # preflop limps + BB check
        (0, 1), (1, 125), (2, 0), (3, 0),                # flop: SB check, BB bet, UTG fold, MP fold
                                                          # STOP at CO (hero)
    ]
    assert actions == expected, (
        f"expected {expected};\n"
        f"got      {actions};\n"
        f"hero=CO at cycle position 4; pre-fix would skip MP after UTG removal"
    )


# --------------------------------------------------------------------------
# Test 3: cascading — multiple defensive folds before hero
# --------------------------------------------------------------------------

def test_cascading_multiple_folds_all_emitted_in_order():
    """Multiple defensive folds before hero on the same postflop street.
    The fix must produce the COMPLETE correct action sequence (every
    intermediate fold present, in order), not just stop at hero with
    folds dropped along the way.

    Setup: hero=BTN(5), 6-handed flop, all limped preflop. Flop: SB
    checks, BB bets, UTG/MP/CO each defensive-fold in turn, BTN/hero
    to-act. Three defensive folds before hero.

    Pre-fix: with three removals from pf_alive_post, the modular cycle
    index would skip past multiple seats — e.g., after UTG fold the
    cycle would jump to CO (skipping MP), so MP's fold would never be
    emitted. OpenSpiel would see only UTG + CO + (spurious-something)
    folded but MP still "alive" in the hand → state mismatch.

    Post-fix: cs_folded set preserves the cycle indexing, so UTG → MP
    → CO each get visited in order and each emit their fold. Hero stop
    fires cleanly after CO fold.
    """
    pre = (1500, 1500, 1500, 1500, 1500, 1500)
    # All 6 limped preflop = lose 30 each (ante 5 + call 25). Flop:
    # BB(idx 1) bets 80, lowering BB stack to 1390. UTG/MP/CO
    # defensive-fold (stacks stay at 1470). BTN(hero) to-act.
    # dealer=5=BTN → SB=0, BB=1, UTG=2, MP=3, CO=4, BTN=5.
    frame = _frame(
        dealer_seat=5, hero_seat=5,
        hero_cards=("Qc", "Qd"),
        stack=(1470, 1390, 1470, 1470, 1470, 1470),  # BB lower by 80
        bet=(0, 80, 0, 0, 0, 0),     # BB(idx 1) bet 80
        board=("3h", "8s", "Td"),
        pot_total=260,  # 30 antes + 150 limps + 80 BB bet = 260
        empty=(False,) * 6,
        captured_at="cascading_synthetic",
    )
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    # Preflop order from dealer=5: UTG(2)→MP(3)→CO(4)→BTN(5)→SB(0)→BB(1).
    # All limp preflop. chip_int=1 for each (call = match BB amount).
    # Postflop order from SB: SB(0)→BB(1)→UTG(2)→MP(3)→CO(4)→BTN(5).
    # Flop: SB check, BB bet 80, UTG fold, MP fold, CO fold, STOP at BTN/hero.
    expected = [
        (2, 1), (3, 1), (4, 1), (5, 1), (0, 1), (1, 1),  # preflop limps
        (0, 1), (1, 105), (2, 0), (3, 0), (4, 0),        # flop: SB check, BB bet (25+80=105),
                                                          # UTG fold, MP fold, CO fold
                                                          # STOP at BTN (hero)
    ]
    assert actions == expected, (
        f"expected {expected};\n"
        f"got      {actions}"
    )
    # Critical: confirm ALL three folds (UTG, MP, CO) are in the sequence
    # AND in the correct postflop order (UTG before MP before CO).
    flop_actions = actions[6:]  # after preflop's 6
    fold_seats_in_order = [s for s, c in flop_actions if c == 0]
    assert fold_seats_in_order == [2, 3, 4], (
        f"ALL three intermediate folds must be present in postflop order; "
        f"got fold sequence {fold_seats_in_order}, expected [2, 3, 4] "
        f"(UTG, MP, CO). The pre-fix bug dropped intermediate folds via "
        f"the cycle-index-shift-on-removal — this test catches that."
    )


# --------------------------------------------------------------------------
# Test 4: no defensive folds — bit-identical no-op
# --------------------------------------------------------------------------

def test_no_defensive_folds_bit_identical():
    """Postflop frame with NO defensive folds (= cs_folded set stays
    empty). The cs_folded code path is a no-op for this case, and the
    emission must match exactly what the pre-fix code produced. This
    guards against the fix accidentally altering the common case where
    no defensive folds happen.
    """
    pre = (1500, 1500, 1500, 1500, 1500, 1500)
    # 3-way flop scenario: 4 limped preflop, UTG/MP folded preflop.
    # Flop: SB checks, BB checks, CO checks, hero (BTN) to-act with
    # no bet facing — NO defensive folds anywhere.
    frame = _frame(
        dealer_seat=5, hero_seat=5,  # dealer=hero=BTN
        hero_cards=("Ah", "Ks"),
        stack=(1470, 1470, 1495, 1495, 1470, 1500),  # UTG/MP only paid ante (5); others limped 25
        bet=(0, 0, 0, 0, 0, 0),  # everyone checked on flop
        board=("2c", "9d", "Jh"),
        pot_total=170,  # 30 antes + 4×25 limps + 4×10 difference? Let me re-compute
        empty=(False,) * 6,
        captured_at="no_folds_synthetic",
    )
    # 4 limpers (SB, BB, CO, BTN) committed 30 chips each = 120
    # UTG, MP folded preflop committing only ante = 5 each = 10
    # Total = 130. Pot in frame should match.
    # Adjust pot_total: 4*30 (limpers) + 2*5 (folded ante) = 130
    frame = _frame(
        dealer_seat=5, hero_seat=5,
        hero_cards=("Ah", "Ks"),
        stack=(1470, 1470, 1495, 1495, 1470, 1470),
        bet=(0, 0, 0, 0, 0, 0),
        board=("2c", "9d", "Jh"),
        pot_total=130,
        empty=(False,) * 6,
        captured_at="no_folds_synthetic",
    )
    actions = derive_action_sequence(frame, pre_hand_override=pre)
    # Preflop: UTG(2) fold, MP(3) fold, CO(4) limp, BTN(5) limp, SB(0) call, BB(1) check.
    # Postflop order: SB(0)→BB(1)→CO(4)→BTN(5).
    # Flop: SB check, BB check, CO check, STOP at BTN (hero). NO folds.
    expected = [
        (2, 0), (3, 0), (4, 1), (5, 1), (0, 1), (1, 1),  # preflop
        (0, 1), (1, 1), (4, 1),                          # flop checks
                                                          # STOP at hero (BTN)
    ]
    assert actions == expected, (
        f"expected {expected};\n"
        f"got      {actions};\n"
        f"this is the no-op guard — cs_folded should stay empty for "
        f"this frame and behavior should match pre-fix exactly."
    )
    # Confirm no fold (chip_int=0) appears in the flop portion
    flop_actions = actions[6:]
    assert all(c != 0 for s, c in flop_actions), (
        f"no defensive folds expected in this frame; got flop_actions={flop_actions}"
    )
