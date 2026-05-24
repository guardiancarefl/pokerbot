"""Tests for src.nlhe.scripted_bots.parser.

Covers tokenization edge cases, AST structure for each rule form, and
integration via real-world fixture files copied from the archetype set.

Run with: python -m pytest tests/test_scripted_bots_parser.py -v
or:       python -m unittest tests.test_scripted_bots_parser
"""
import os
import pathlib
import unittest

from src.nlhe.scripted_bots.parser import (
    parse_profile, tokenize, Profile, Section, Rule, Action, ActionKind,
    BoolOp, Not, Compare, PredCall, PositionPred, OthersAtom,
    NumberLit, IdentLit, PercentExpr, HandSpec, BoardSpec, Card,
    ParseError,
    TokenKind,
)


FIXTURE_DIR = pathlib.Path(__file__).parent / "scripted_bots_fixtures"


# ============================================================
# Tokenizer
# ============================================================

class TestTokenizer(unittest.TestCase):
    def test_simple_idents_and_keywords(self):
        toks = tokenize("when fold force")
        # Three keyword tokens + EOF.
        self.assertEqual(len(toks), 4)
        self.assertEqual([(t.kind, t.value) for t in toks[:3]], [
            (TokenKind.KEYWORD, "when"),
            (TokenKind.IDENT, "fold"),     # fold is an action ident, not reserved
            (TokenKind.KEYWORD, "force"),
        ])

    def test_comparison_operators(self):
        toks = tokenize("a = 5 b <= 10 c >= 20 d <> 30 e < 1 f > 2")
        ops = [t.value for t in toks if t.kind == TokenKind.OP]
        self.assertEqual(ops, ["=", "<=", ">=", "<>", "<", ">"])

    def test_card_numeric_token(self):
        """`9c`, `8s` etc. tokenize as a single IDENT, not NUMBER + IDENT."""
        toks = tokenize("hand = 9c 8s")
        idents = [t.value for t in toks if t.kind == TokenKind.IDENT]
        self.assertIn("9c", idents)
        self.assertIn("8s", idents)

    def test_bom_stripping(self):
        """A leading UTF-8 BOM should be silently stripped."""
        toks = tokenize("\ufeffwhen fold")
        self.assertEqual(toks[0].value, "when")

    def test_comments_ignored(self):
        toks = tokenize("// this is a comment\nwhen fold")
        self.assertEqual([t.value for t in toks if t.kind != TokenKind.EOF],
                         ["when", "fold"])

    def test_orhand_normalization(self):
        """The `orhand` typo (missing space in `or hand`) is fixed pre-tokenization."""
        toks = tokenize("when hand = A orhand = K fold")
        values = [t.value for t in toks]
        # Should contain `or` and `hand` as separate tokens after the typo fix.
        idx = values.index("or")
        self.assertEqual(values[idx + 1], "hand")

    def test_concatenated_suited(self):
        """`A6suited` and `AKoffsuit` split into rank-pair + modifier."""
        toks = tokenize("when hand = A6suited fold")
        values = [t.value for t in toks]
        # `a6` is a 2-rank pair token; `suited` is a separate ident.
        self.assertIn("a6", values)
        self.assertIn("suited", values)


# ============================================================
# Expression parsing — primary forms
# ============================================================

class TestExpressionPrimary(unittest.TestCase):
    def _parse_single_rule(self, source: str) -> Rule:
        """Helper: parse a one-rule profile and return that rule."""
        wrapped = f"custom\npreflop\n{source}\n"
        prof = parse_profile(wrapped)
        return prof.sections[0].rules[0]

    def test_bare_predicate(self):
        rule = self._parse_single_rule("when havetoppair fold force")
        self.assertIsInstance(rule.condition, PredCall)
        self.assertEqual(rule.condition.name, "havetoppair")

    def test_compare_number(self):
        rule = self._parse_single_rule("when stacksize <= 10 fold force")
        self.assertIsInstance(rule.condition, Compare)
        self.assertEqual(rule.condition.lhs, "stacksize")
        self.assertEqual(rule.condition.op, "<=")
        self.assertIsInstance(rule.condition.rhs, NumberLit)
        self.assertEqual(rule.condition.rhs.value, 10)

    def test_compare_ident(self):
        rule = self._parse_single_rule("when position = first fold force")
        self.assertIsInstance(rule.condition, Compare)
        self.assertEqual(rule.condition.lhs, "position")
        self.assertIsInstance(rule.condition.rhs, IdentLit)
        self.assertEqual(rule.condition.rhs.name, "first")

    def test_percent_expression(self):
        rule = self._parse_single_rule("when amounttocall <= 50% potsize fold force")
        self.assertIsInstance(rule.condition, Compare)
        self.assertIsInstance(rule.condition.rhs, PercentExpr)
        self.assertEqual(rule.condition.rhs.pct, 50)
        self.assertEqual(rule.condition.rhs.target, "potsize")

    def test_hand_spec_simple(self):
        rule = self._parse_single_rule("when hand = AK fold force")
        self.assertIsInstance(rule.condition.rhs, HandSpec)
        cards = rule.condition.rhs.cards
        self.assertEqual([c.rank for c in cards], ["A", "K"])

    def test_hand_spec_suited(self):
        rule = self._parse_single_rule("when hand = AK suited fold force")
        spec = rule.condition.rhs
        self.assertEqual(spec.suitedness, "suited")

    def test_hand_spec_offsuit(self):
        rule = self._parse_single_rule("when hand = AK offsuit fold force")
        spec = rule.condition.rhs
        self.assertEqual(spec.suitedness, "offsuit")

    def test_hand_spec_pair(self):
        rule = self._parse_single_rule("when hand = AA fold force")
        cards = rule.condition.rhs.cards
        self.assertEqual([c.rank for c in cards], ["A", "A"])

    def test_hand_spec_specific_suits(self):
        rule = self._parse_single_rule("when hand = Kd 9d fold force")
        cards = rule.condition.rhs.cards
        self.assertEqual(cards[0].rank, "K")
        self.assertEqual(cards[0].suit, "d")
        self.assertEqual(cards[1].rank, "9")
        self.assertEqual(cards[1].suit, "d")

    def test_hand_spec_wildcard(self):
        rule = self._parse_single_rule("when hand = A fold force")
        spec = rule.condition.rhs
        self.assertEqual(len(spec.cards), 1)
        self.assertEqual(spec.cards[0].rank, "A")

    def test_board_spec_single_rank(self):
        rule = self._parse_single_rule("when board = A fold force")
        self.assertEqual(rule.condition.rhs.ranks, ["A"])

    def test_board_spec_multi_rank_concat(self):
        rule = self._parse_single_rule("when board = AKQ fold force")
        self.assertEqual(rule.condition.rhs.ranks, ["A", "K", "Q"])

    def test_board_spec_multi_rank_spaced(self):
        rule = self._parse_single_rule("when board = A K Q fold force")
        self.assertEqual(rule.condition.rhs.ranks, ["A", "K", "Q"])

    def test_in_position(self):
        rule = self._parse_single_rule("when in smallblind fold force")
        self.assertIsInstance(rule.condition, PositionPred)
        self.assertEqual(rule.condition.position, "smallblind")

    def test_others_atom(self):
        rule = self._parse_single_rule("when others fold force")
        self.assertIsInstance(rule.condition, OthersAtom)


# ============================================================
# Expression parsing — boolean composition
# ============================================================

class TestExpressionBoolean(unittest.TestCase):
    def _parse_cond(self, expr: str):
        wrapped = f"custom\npreflop\nwhen {expr} fold force\n"
        return parse_profile(wrapped).sections[0].rules[0].condition

    def test_and(self):
        cond = self._parse_cond("havetoppair and havebestkicker")
        self.assertIsInstance(cond, BoolOp)
        self.assertEqual(cond.op, "and")
        self.assertEqual(len(cond.args), 2)

    def test_or(self):
        cond = self._parse_cond("hand = AA or hand = KK")
        self.assertIsInstance(cond, BoolOp)
        self.assertEqual(cond.op, "or")

    def test_not(self):
        cond = self._parse_cond("not flushpossible")
        self.assertIsInstance(cond, Not)
        self.assertIsInstance(cond.arg, PredCall)

    def test_precedence_or_over_and(self):
        # `a and b or c` should parse as `(a and b) or c`.
        cond = self._parse_cond("havepair and havebestkicker or havetrips")
        self.assertEqual(cond.op, "or")
        self.assertEqual(cond.args[0].op, "and")     # left of OR is the AND

    def test_parentheses(self):
        cond = self._parse_cond("(hand = AA or hand = KK) and raises = 0")
        self.assertEqual(cond.op, "and")
        self.assertEqual(cond.args[0].op, "or")


# ============================================================
# Actions
# ============================================================

class TestActions(unittest.TestCase):
    def _parse_action(self, action_text: str) -> Action:
        wrapped = f"custom\npreflop\nwhen havepair {action_text}\n"
        return parse_profile(wrapped).sections[0].rules[0].action

    def test_fold(self):
        a = self._parse_action("fold force")
        self.assertEqual(a.kind, ActionKind.FOLD)
        self.assertTrue(a.force)

    def test_call(self):
        self.assertEqual(self._parse_action("call force").kind, ActionKind.CALL)

    def test_check(self):
        self.assertEqual(self._parse_action("check force").kind, ActionKind.CHECK)

    def test_raise_amount(self):
        a = self._parse_action("raise 3 force")
        self.assertEqual(a.kind, ActionKind.RAISE_AMOUNT)
        self.assertEqual(a.amount, 3)

    def test_raise_percent(self):
        a = self._parse_action("raise 75% potsize force")
        self.assertEqual(a.kind, ActionKind.RAISE_PERCENT)
        self.assertEqual(a.amount, 75)
        self.assertEqual(a.amount_target, "potsize")

    def test_raise_bare(self):
        """Bare `raise force` (no amount) — engine uses default raise size."""
        a = self._parse_action("raise force")
        self.assertEqual(a.kind, ActionKind.RAISE_AMOUNT)
        self.assertIsNone(a.amount)

    def test_raisemin_raisepot_raisemax(self):
        self.assertEqual(self._parse_action("raisemin force").kind, ActionKind.RAISE_MIN)
        self.assertEqual(self._parse_action("raisepot force").kind, ActionKind.RAISE_POT)
        self.assertEqual(self._parse_action("raisemax force").kind, ActionKind.RAISE_MAX)

    def test_bet_variants(self):
        self.assertEqual(self._parse_action("bet 50% potsize force").kind, ActionKind.BET_PERCENT)
        self.assertEqual(self._parse_action("bet 5 force").kind, ActionKind.BET_AMOUNT)
        self.assertEqual(self._parse_action("betmin force").kind, ActionKind.BET_MIN)
        self.assertEqual(self._parse_action("betpot force").kind, ActionKind.BET_POT)
        self.assertEqual(self._parse_action("betmax force").kind, ActionKind.BET_MAX)

    def test_user_flag_as_action(self):
        a = self._parse_action("usertrap1 force")
        self.assertEqual(a.kind, ActionKind.SET_USER_FLAG)
        self.assertEqual(a.user_flag, "usertrap1")

    def test_sitout(self):
        self.assertEqual(self._parse_action("sitout force").kind, ActionKind.SITOUT)

    def test_beep(self):
        self.assertEqual(self._parse_action("beep force").kind, ActionKind.BEEP)

    def test_delay_modifier(self):
        a = self._parse_action("raise 3 force delay 2")
        self.assertEqual(a.delay, 2)


# ============================================================
# Scoped rule blocks
# ============================================================

class TestScopedBlocks(unittest.TestCase):
    def test_scoped_block_ands_into_children(self):
        """A `when <expr>` with no action scopes subsequent rules."""
        source = (
            "custom\npreflop\n"
            "when (hand = AK)\n"
            "when raises = 0 raisepot force\n"
            "when raises = 1 call force\n"
        )
        sec = parse_profile(source).sections[0]
        self.assertEqual(len(sec.rules), 2)
        # Each rule's condition should be a BoolOp(and, [parent, child]).
        for rule in sec.rules:
            self.assertIsInstance(rule.condition, BoolOp)
            self.assertEqual(rule.condition.op, "and")

    def test_new_scope_header_closes_prev(self):
        """A new scope header replaces the previous scope, not nests."""
        source = (
            "custom\npreflop\n"
            "when (hand = AK)\n"
            "when raises = 0 raisepot force\n"
            "when (hand = QQ)\n"
            "when raises = 0 raisepot force\n"
        )
        rules = parse_profile(source).sections[0].rules
        # First rule's parent is `hand = AK`; second's parent is `hand = QQ`.
        # The conditions' first arg (Compare) should differ between rules.
        first_parent = rules[0].condition.args[0]
        second_parent = rules[1].condition.args[0]
        # Compare's RHS HandSpec.cards differ.
        self.assertNotEqual(
            [c.rank for c in first_parent.rhs.cards],
            [c.rank for c in second_parent.rhs.cards],
        )


# ============================================================
# Integration: real fixture files
# ============================================================

class TestFixtures(unittest.TestCase):
    """Smoke-test every real Shanky profile we have on disk."""

    def _fixture(self, name):
        path = FIXTURE_DIR / name
        return parse_profile(path.read_text(), name)

    def test_littlegreen_parses(self):
        prof = self._fixture("littlegreen.txt")
        self.assertTrue(prof.has_custom)
        self.assertEqual([s.name for s in prof.sections], ["preflop", "flop"])
        self.assertEqual(sum(len(s.rules) for s in prof.sections), 10)

    def test_beep_minimal_bot(self):
        """beep.txt is a 20-line test bot using only the rare `beep` action."""
        prof = self._fixture("beep__1_.txt")
        self.assertTrue(prof.has_custom)
        # 4 sections (preflop/flop/turn/river), 1 rule + 1 default each.
        self.assertEqual(len(prof.sections), 4)
        # First rule uses BEEP action.
        beep_count = sum(
            1 for s in prof.sections for r in s.rules
            if r.action.kind == ActionKind.BEEP
        )
        self.assertGreater(beep_count, 0)

    def test_timidtom_parses_with_user_flags(self):
        """timidtom uses user-flags extensively."""
        prof = self._fixture("timidtom.txt")
        self.assertTrue(prof.has_custom)
        self.assertGreater(len(prof.user_flags), 0)

    def test_ticketmaster_parses_with_defaults(self):
        """ticketmaster has all four sections; some have explicit `when others` defaults."""
        prof = self._fixture("ticketmaster.txt")
        section_names = [s.name for s in prof.sections]
        self.assertEqual(section_names, ["preflop", "flop", "turn", "river"])
        # At least some sections should have a default action.
        sections_with_default = sum(1 for s in prof.sections if s.default_action is not None)
        self.assertGreater(sections_with_default, 0,
                           "expected at least one section with a `when others` default")
        # Total rule count is substantial (this is a real archetype).
        total = sum(len(s.rules) for s in prof.sections)
        self.assertGreater(total, 20)

    def test_6pack_parses(self):
        """6pack uses user-flags as a back-up strategy hook."""
        prof = self._fixture("6pack__1_.txt")
        self.assertTrue(prof.has_custom)

    def test_littlegreen_v2_parses(self):
        """littlegreen2 is a near-clone of littlegreen with minor variations."""
        prof = self._fixture("littlegreen2.txt")
        self.assertTrue(prof.has_custom)

    def test_fixedlimitheadsup_uses_scoped_blocks(self):
        """fixedlimitheadsup uses scoped rule blocks (`When (...)` with no action)."""
        prof = self._fixture("fixedlimitheadsup__1_.txt")
        self.assertTrue(prof.has_custom)
        # Has both options block AND custom rules.
        self.assertGreater(len(prof.options), 0)
        # At least some rules should have BoolOp(and, ...) conditions
        # indicating scope application.
        scoped = 0
        for sec in prof.sections:
            for rule in sec.rules:
                if isinstance(rule.condition, BoolOp) and rule.condition.op == "and":
                    scoped += 1
        self.assertGreater(scoped, 5, "expected scoped rules in fixedlimit profile")


# ============================================================
# Error reporting
# ============================================================

class TestErrors(unittest.TestCase):
    def test_unknown_action_errors_cleanly(self):
        with self.assertRaises(ParseError) as cm:
            parse_profile("custom\npreflop\nwhen havepair somethingelse force\n")
        self.assertIn("unknown action", str(cm.exception))

    def test_malformed_compare_errors_cleanly(self):
        with self.assertRaises(ParseError):
            parse_profile("custom\npreflop\nwhen stacksize <= fold force\n")

    def test_invalid_board_rank_errors(self):
        with self.assertRaises(ParseError) as cm:
            parse_profile("custom\npreflop\nwhen board = X fold force\n")
        self.assertIn("invalid rank", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
