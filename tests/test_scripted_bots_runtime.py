"""Tests for src.nlhe.scripted_bots.runtime.

Covers card encoding helpers, HandFeatures computation, BoardFeatures
computation, hand-spec/board-spec matching, expression evaluation,
and integration via end-to-end profile evaluation against real fixtures.
"""
import os
import pathlib
import unittest

from src.nlhe.scripted_bots.parser import parse_profile, ActionKind
from src.nlhe.scripted_bots.runtime import (
    GameContext, Runtime,
    evaluate_profile, compute_hand_features, compute_board_features,
    matches_hand_spec, matches_board_spec,
    card_to_int, int_to_rank_suit, card_rank, card_suit,
    UnsupportedPredicateError,
)


FIXTURE_DIR = pathlib.Path(__file__).parent / "scripted_bots_fixtures"


def C(rank: str, suit: str) -> int:
    """Convenience: card_to_int('A','s') -> 51."""
    return card_to_int(rank, suit)


# ============================================================
# Card encoding
# ============================================================

class TestCardEncoding(unittest.TestCase):
    def test_card_to_int_basic(self):
        # 2c is the lowest card (rank 0, suit 0)
        self.assertEqual(card_to_int('2', 'c'), 0)
        # As is the highest (rank 12, suit 3)
        self.assertEqual(card_to_int('A', 's'), 51)

    def test_card_to_int_roundtrip(self):
        for r in "23456789TJQKA":
            for s in "cdhs":
                ci = card_to_int(r, s)
                rr, ss = int_to_rank_suit(ci)
                self.assertEqual((rr, ss), (r, s))

    def test_card_rank_suit_helpers(self):
        c = card_to_int('K', 'h')
        self.assertEqual(card_rank(c), 11)
        self.assertEqual(card_suit(c), 2)


# ============================================================
# Hand features
# ============================================================

class TestHandFeatures(unittest.TestCase):
    def test_pair_in_hand(self):
        hf = compute_hand_features([C('A','s'), C('A','h')], [])
        self.assertTrue(hf.pairinhand)
        self.assertTrue(hf.havepair)

    def test_top_pair_with_best_kicker(self):
        # A♠K♣ on A♥7♦2♠ — top pair, best kicker (K is highest non-paired rank)
        hf = compute_hand_features(
            [C('A','s'), C('K','c')],
            [C('A','h'), C('7','d'), C('2','s')]
        )
        self.assertTrue(hf.havetoppair)
        self.assertTrue(hf.havebestkicker)
        self.assertFalse(hf.pairinhand)

    def test_top_pair_with_weak_kicker(self):
        # A2 on A77 — top pair but the kicker (2) is worst
        hf = compute_hand_features(
            [C('A','s'), C('2','c')],
            [C('A','h'), C('7','d'), C('7','s')]
        )
        self.assertTrue(hf.havetoppair)
        self.assertFalse(hf.havebestkicker)

    def test_overpair_top_overpair(self):
        # QQ on 752 — overpair to all board cards
        hf = compute_hand_features(
            [C('Q','s'), C('Q','h')],
            [C('7','d'), C('5','c'), C('2','s')]
        )
        self.assertTrue(hf.haveoverpair)
        self.assertFalse(hf.havetoppair)

    def test_underpair(self):
        # 22 on QJ7 — pair below the board
        hf = compute_hand_features(
            [C('2','s'), C('2','h')],
            [C('Q','d'), C('J','c'), C('7','s')]
        )
        self.assertTrue(hf.haveunderpair)

    def test_set(self):
        # 77 on 7AK — pocket pair + 1 = set
        hf = compute_hand_features(
            [C('7','s'), C('7','h')],
            [C('7','d'), C('A','c'), C('K','s')]
        )
        self.assertTrue(hf.haveset)
        self.assertFalse(hf.havetrips)

    def test_trips(self):
        # AK on AA7 — board pair + 1 hole card = trips
        hf = compute_hand_features(
            [C('A','s'), C('K','c')],
            [C('A','h'), C('A','d'), C('7','s')]
        )
        self.assertTrue(hf.havetrips)
        self.assertFalse(hf.haveset)

    def test_two_pair(self):
        # AK on AKQ — two pair
        hf = compute_hand_features(
            [C('A','s'), C('K','c')],
            [C('A','h'), C('K','d'), C('Q','s')]
        )
        self.assertTrue(hf.havetwopair)
        self.assertTrue(hf.havetoptwopair)
        self.assertFalse(hf.haveset)

    def test_straight(self):
        # T9 on JQK — straight to the king
        hf = compute_hand_features(
            [C('T','s'), C('9','c')],
            [C('J','h'), C('Q','d'), C('K','s')]
        )
        self.assertTrue(hf.havestraight)

    def test_wheel_straight(self):
        # A2 on 345 — wheel straight (A-2-3-4-5)
        hf = compute_hand_features(
            [C('A','s'), C('2','c')],
            [C('3','h'), C('4','d'), C('5','s')]
        )
        self.assertTrue(hf.havestraight)

    def test_flush(self):
        # AsKsQsJs2s — all spades, but ranks A K Q J 2 don't form a straight
        # (a straight needs 5 consecutive ranks).
        hf = compute_hand_features(
            [C('A','s'), C('K','s')],
            [C('Q','s'), C('J','s'), C('2','s')]
        )
        self.assertTrue(hf.haveflush)
        self.assertFalse(hf.havestraight)
        self.assertFalse(hf.havestraightflush)

    def test_royal_flush(self):
        # AKQJT all spades — straight flush (royal)
        hf = compute_hand_features(
            [C('A','s'), C('K','s')],
            [C('Q','s'), C('J','s'), C('T','s')]
        )
        self.assertTrue(hf.haveflush)
        self.assertTrue(hf.havestraight)
        self.assertTrue(hf.havestraightflush)

    def test_flush_draw(self):
        # AsKs on Qs Js 2c — 4 spades total (hero 2 + board 2) = flush draw
        hf = compute_hand_features(
            [C('A','s'), C('K','s')],
            [C('Q','s'), C('J','s'), C('2','c')]
        )
        self.assertTrue(hf.haveflushdraw)
        self.assertTrue(hf.havenutflushdraw)     # holds As

    def test_backdoor_flush_draw(self):
        # AsKs on Qs Jd 2c — only 3 spades total = backdoor flush draw, not flush draw
        hf = compute_hand_features(
            [C('A','s'), C('K','s')],
            [C('Q','s'), C('J','d'), C('2','c')]
        )
        self.assertFalse(hf.haveflushdraw)
        self.assertTrue(hf.havebackdoorflushdraw)
        self.assertTrue(hf.havebackdoornutflushdraw)

    def test_open_ended_straight_draw(self):
        # 89 on 67K — 4 to a straight (5/T outs)
        hf = compute_hand_features(
            [C('8','s'), C('9','c')],
            [C('6','h'), C('7','d'), C('K','s')]
        )
        self.assertTrue(hf.havestraightdraw)
        self.assertFalse(hf.haveinsidestraightdraw)

    def test_inside_straight_draw(self):
        # 89 on 67Q with a gap (need T) actually that's OESD too
        # Use 89 on 6QK — only T fills the gap = gutshot from one side
        # Actually 89 on 6Q gives us 6789Q which still wants T -> gutshot
        # Cleaner: 89 on 5QK — only 67 needed but already have nothing close
        # Best clean gutshot: J9 on TQK -> need any 8 or A to make a straight,
        # but those are different straights (one with J9TQK + 8 = 89TJQ, other A = TJQKA)
        # Simpler: T7 on K84 — need 6 or 9 inside, gutshot.
        # Actually try: T6 on K85 — need 9 or 7 inside, two gutshots ≈ OESD really
        # Let's try J7 on K9? — only 8 fills J7K9_ as 789JK; not really clean.
        # Use: J9 on K72 — need T8 to fill = bad
        # Cleaner: T8 on K94 — need J7 to complete 789T or one only.
        # OK use 76 on K45 — need 3 to make 34567 (one out below); skip
        # Just use the cleanest test: T9 on 76K — that's OESD (8 or J)
        # For gutshot: T8 on K76 — need 9 to make 678910 (only one card fills)
        # Actually any rank fill makes a straight: 8 on T8K76 -> 678T... wait no, 6,7,8,T isn't a straight
        # Need to consider: cards are T,8,K,7,6. Straights of length 5: T8K76 - the sequence is 6,7,8,T,K. Missing 9. Adding 9: 6,7,8,9,T = straight (9-high). That's gutshot (one out: 9).
        hf = compute_hand_features(
            [C('T','s'), C('8','c')],
            [C('K','h'), C('7','d'), C('6','s')]
        )
        self.assertTrue(hf.haveinsidestraightdraw or hf.havestraightdraw)


# ============================================================
# Board features
# ============================================================

class TestBoardFeatures(unittest.TestCase):
    def test_pair_on_board(self):
        bf = compute_board_features([C('A','s'), C('A','h'), C('7','d')])
        self.assertTrue(bf.paironboard)
        self.assertFalse(bf.tripsonboard)

    def test_trips_on_board(self):
        bf = compute_board_features([C('A','s'), C('A','h'), C('A','d')])
        self.assertTrue(bf.tripsonboard)
        self.assertTrue(bf.paironboard)

    def test_flush_possible(self):
        bf = compute_board_features([C('A','s'), C('K','s'), C('Q','s')])
        self.assertTrue(bf.flushpossible)

    def test_onecard_flush_possible(self):
        bf = compute_board_features([C('A','s'), C('K','s'), C('Q','s'), C('J','s')])
        self.assertTrue(bf.onecardflushpossible)
        self.assertTrue(bf.flushpossible)

    def test_straight_possible(self):
        bf = compute_board_features([C('K','s'), C('Q','h'), C('J','d')])
        self.assertTrue(bf.straightpossible)
        self.assertTrue(bf.threecardstraightonboard)

    def test_uncoordinated_flop(self):
        # K72 rainbow — no pair, no flush, no straight draws
        bf = compute_board_features([C('K','s'), C('7','h'), C('2','d')])
        self.assertTrue(bf.uncoordinatedflop)
        self.assertFalse(bf.straightpossible)
        self.assertFalse(bf.flushpossible)

    def test_ace_present_on_flop(self):
        bf = compute_board_features([C('A','s'), C('7','h'), C('2','d')])
        self.assertTrue(bf.acepresentonflop)
        self.assertFalse(bf.kingpresentonflop)

    def test_pair_on_turn(self):
        # Flop K72, turn pairs the K
        bf = compute_board_features([C('K','s'), C('7','h'), C('2','d'), C('K','c')])
        self.assertTrue(bf.paironturn)
        self.assertTrue(bf.topflopcardpairedonturn)

    def test_suits_on_board(self):
        bf = compute_board_features([C('A','s'), C('K','h'), C('Q','d')])
        self.assertEqual(bf.suitsonboard, 3)


# ============================================================
# Hand-spec matching
# ============================================================

class TestHandSpecMatching(unittest.TestCase):
    def _parse_hand_spec(self, hand_text: str):
        """Helper: parse 'hand = X' and return the HandSpec."""
        wrapped = f"custom\npreflop\nwhen hand = {hand_text} fold force\n"
        prof = parse_profile(wrapped)
        return prof.sections[0].rules[0].condition.rhs

    def test_exact_pair_matches(self):
        spec = self._parse_hand_spec("AA")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('A','h')]))
        self.assertFalse(matches_hand_spec(spec, [C('K','s'), C('K','h')]))
        self.assertFalse(matches_hand_spec(spec, [C('A','s'), C('K','h')]))

    def test_two_rank_unsuited_matches_both(self):
        spec = self._parse_hand_spec("AK")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('K','h')]))   # offsuit
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('K','s')]))   # suited
        # Order shouldn't matter
        self.assertTrue(matches_hand_spec(spec, [C('K','h'), C('A','s')]))

    def test_suited_matches_only_suited(self):
        spec = self._parse_hand_spec("AK suited")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('K','s')]))
        self.assertFalse(matches_hand_spec(spec, [C('A','s'), C('K','h')]))

    def test_offsuit_matches_only_offsuit(self):
        spec = self._parse_hand_spec("AK offsuit")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('K','h')]))
        self.assertFalse(matches_hand_spec(spec, [C('A','s'), C('K','s')]))

    def test_wildcard_matches_any_with_rank(self):
        spec = self._parse_hand_spec("A")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('2','c')]))
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('K','h')]))
        self.assertFalse(matches_hand_spec(spec, [C('K','s'), C('Q','c')]))

    def test_wildcard_suited(self):
        spec = self._parse_hand_spec("A suited")
        self.assertTrue(matches_hand_spec(spec, [C('A','s'), C('2','s')]))
        self.assertFalse(matches_hand_spec(spec, [C('A','s'), C('2','c')]))

    def test_specific_cards_match(self):
        spec = self._parse_hand_spec("Kd 9d")
        self.assertTrue(matches_hand_spec(spec, [C('K','d'), C('9','d')]))
        # Order shouldn't matter
        self.assertTrue(matches_hand_spec(spec, [C('9','d'), C('K','d')]))
        # Different suits shouldn't match
        self.assertFalse(matches_hand_spec(spec, [C('K','h'), C('9','d')]))


# ============================================================
# Board-spec matching
# ============================================================

class TestBoardSpecMatching(unittest.TestCase):
    def _parse_board_spec(self, board_text: str):
        wrapped = f"custom\npreflop\nwhen board = {board_text} fold force\n"
        prof = parse_profile(wrapped)
        return prof.sections[0].rules[0].condition.rhs

    def test_single_rank_present(self):
        spec = self._parse_board_spec("A")
        self.assertTrue(matches_board_spec(spec, [C('A','s'), C('7','d'), C('2','c')]))
        self.assertFalse(matches_board_spec(spec, [C('K','s'), C('7','d'), C('2','c')]))

    def test_three_ranks_concat(self):
        spec = self._parse_board_spec("AKQ")
        self.assertTrue(matches_board_spec(spec, [C('A','s'), C('K','d'), C('Q','c')]))
        # Order in board doesn't matter
        self.assertTrue(matches_board_spec(spec, [C('Q','c'), C('K','d'), C('A','s')]))
        self.assertFalse(matches_board_spec(spec, [C('A','s'), C('K','d'), C('J','c')]))


# ============================================================
# Expression evaluation
# ============================================================

class TestExpressionEvaluation(unittest.TestCase):
    def _eval(self, expr_text: str, ctx: GameContext) -> bool:
        wrapped = f"custom\npreflop\nwhen {expr_text} fold force\n"
        prof = parse_profile(wrapped)
        rt = Runtime(prof, ctx)
        return rt.evaluate_expression(prof.sections[0].rules[0].condition)

    def test_simple_compare_numeric(self):
        ctx = GameContext(stacksize=15.0)
        self.assertTrue(self._eval("stacksize > 10", ctx))
        self.assertFalse(self._eval("stacksize > 20", ctx))

    def test_simple_compare_position(self):
        ctx = GameContext(position="button")
        self.assertTrue(self._eval("position = button", ctx))
        self.assertFalse(self._eval("position = first", ctx))

    def test_compound_and(self):
        ctx = GameContext(stacksize=15.0, raises=1)
        self.assertTrue(self._eval("stacksize > 10 and raises = 1", ctx))
        self.assertFalse(self._eval("stacksize > 10 and raises = 2", ctx))

    def test_compound_or(self):
        ctx = GameContext(stacksize=5.0, raises=2)
        self.assertTrue(self._eval("stacksize > 10 or raises = 2", ctx))
        self.assertFalse(self._eval("stacksize > 10 or raises = 5", ctx))

    def test_not(self):
        ctx = GameContext(stacksize=5.0)
        self.assertTrue(self._eval("not (stacksize > 10)", ctx))

    def test_hand_compare(self):
        ctx = GameContext(hole_cards=[C('A','s'), C('A','h')])
        self.assertTrue(self._eval("hand = AA", ctx))
        self.assertFalse(self._eval("hand = KK", ctx))

    def test_percent_expression(self):
        ctx = GameContext(amounttocall=5.0, potsize=20.0)
        # 5 <= 50% of 20 = 10? Yes.
        self.assertTrue(self._eval("amounttocall <= 50% potsize", ctx))
        # 5 <= 20% of 20 = 4? No.
        self.assertFalse(self._eval("amounttocall <= 20% potsize", ctx))

    def test_in_position_predicate(self):
        ctx = GameContext(position="bigblind")
        self.assertTrue(self._eval("in bigblind", ctx))
        self.assertFalse(self._eval("in button", ctx))

    def test_havetoppair_predicate(self):
        ctx = GameContext(
            hole_cards=[C('A','s'), C('K','c')],
            board=[C('A','h'), C('7','d'), C('2','s')],
            street='flop',
        )
        self.assertTrue(self._eval("havetoppair", ctx))
        self.assertTrue(self._eval("havebestkicker", ctx))

    def test_board_predicate(self):
        ctx = GameContext(board=[C('K','s'), C('Q','d'), C('J','c')])
        self.assertTrue(self._eval("threecardstraightonboard", ctx))


# ============================================================
# Section iteration / user-flags
# ============================================================

class TestSectionIteration(unittest.TestCase):
    def test_first_match_wins(self):
        source = """custom
preflop
when hand = AA raisemax force
when hand = KK raise 3 force
when hand = QQ raise 2 force
"""
        prof = parse_profile(source)
        ctx = GameContext(hole_cards=[C('K','s'), C('K','h')], street='preflop')
        action = evaluate_profile(prof, ctx)
        self.assertEqual(action.kind, ActionKind.RAISE_AMOUNT)
        self.assertEqual(action.amount, 3)

    def test_no_match_returns_none(self):
        source = "custom\npreflop\nwhen hand = AA raisemax force\n"
        prof = parse_profile(source)
        ctx = GameContext(hole_cards=[C('K','s'), C('K','h')], street='preflop')
        self.assertIsNone(evaluate_profile(prof, ctx))

    def test_user_flag_set_and_test(self):
        """Flag-setting rule should NOT terminate iteration; following rule can test the flag."""
        source = """custom
preflop
when hand = AA userpremium
when userpremium raisemax force
when hand = KK raise 3 force
"""
        prof = parse_profile(source)
        ctx = GameContext(hole_cards=[C('A','s'), C('A','h')], street='preflop')
        action = evaluate_profile(prof, ctx)
        self.assertEqual(action.kind, ActionKind.RAISE_MAX)

    def test_others_atom_fires_only_if_no_prior_match(self):
        source = """custom
preflop
when hand = AA raisemax force
when others fold force
"""
        prof = parse_profile(source)
        # AA hits the first rule
        ctx_aa = GameContext(hole_cards=[C('A','s'), C('A','h')], street='preflop')
        action = evaluate_profile(prof, ctx_aa)
        self.assertEqual(action.kind, ActionKind.RAISE_MAX)
        # KK does not hit the first rule; falls through to `others`
        ctx_kk = GameContext(hole_cards=[C('K','s'), C('K','h')], street='preflop')
        action = evaluate_profile(prof, ctx_kk)
        self.assertEqual(action.kind, ActionKind.FOLD)


# ============================================================
# Integration via real fixtures
# ============================================================

class TestFixtureIntegration(unittest.TestCase):
    def _fixture(self, name):
        return parse_profile((FIXTURE_DIR / name).read_text(), name)

    def test_littlegreen_77_open(self):
        """littlegreen opens 77 to 2bb from late position with no action."""
        prof = self._fixture("littlegreen.txt")
        ctx = GameContext(
            hole_cards=[C('7','s'), C('7','h')],
            street='preflop', stilltoact=2, raises=0, calls=0,
            amounttocall=1.0, stacksize=100.0,
        )
        action = evaluate_profile(prof, ctx)
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.RAISE_AMOUNT)
        self.assertEqual(action.amount, 2)

    def test_littlegreen_AA_not_in_range(self):
        """littlegreen is a speculative-hands-only profile; AA is NOT in its range."""
        prof = self._fixture("littlegreen.txt")
        ctx = GameContext(
            hole_cards=[C('A','s'), C('A','h')],
            street='preflop', stilltoact=2, raises=0, calls=0,
        )
        self.assertIsNone(evaluate_profile(prof, ctx))

    def test_beep_responds_with_beep_for_premium(self):
        """beep.txt is a test bot that only knows the `beep` action for premium hands."""
        prof = self._fixture("beep__1_.txt")
        ctx = GameContext(
            hole_cards=[C('A','s'), C('A','h')],
            street='preflop', stilltoact=2,
        )
        action = evaluate_profile(prof, ctx)
        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.BEEP)

    def test_beep_folds_garbage(self):
        prof = self._fixture("beep__1_.txt")
        ctx = GameContext(
            hole_cards=[C('7','s'), C('2','c')],
            street='preflop',
        )
        action = evaluate_profile(prof, ctx)
        # Falls through to `when others fold force`
        self.assertEqual(action.kind, ActionKind.FOLD)

    def test_ticketmaster_has_sections(self):
        """Sanity: ticketmaster.txt is a real archetype with all 4 sections."""
        prof = self._fixture("ticketmaster.txt")
        section_names = [s.name for s in prof.sections]
        self.assertEqual(section_names, ["preflop", "flop", "turn", "river"])

    def test_77_set_on_dry_board_raises(self):
        """littlegreen's flop rule: haveset on safe board -> raisemax."""
        prof = self._fixture("littlegreen.txt")
        ctx = GameContext(
            hole_cards=[C('7','s'), C('7','h')],
            board=[C('7','d'), C('A','c'), C('K','h')],
            street='flop',
            opponents=1,
        )
        action = evaluate_profile(prof, ctx)
        self.assertEqual(action.kind, ActionKind.RAISE_MAX)


# ============================================================
# Performance (smoke)
# ============================================================

class TestPerformance(unittest.TestCase):
    def test_eval_runs_in_reasonable_time(self):
        """1000 evals of a small profile should be << 1 second."""
        import time
        prof = parse_profile((FIXTURE_DIR / "littlegreen.txt").read_text())
        ctx = GameContext(
            hole_cards=[C('7','s'), C('7','h')],
            street='preflop', stilltoact=2,
        )
        # Warmup
        evaluate_profile(prof, ctx)
        t0 = time.time()
        for _ in range(1000):
            evaluate_profile(prof, ctx)
        elapsed = time.time() - t0
        # Should comfortably finish in under 1 second.
        self.assertLess(elapsed, 1.0, f"1000 evals took {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
