"""Shanky/WinHoldem profile language parser.

Parses .txt/.ppl bot-profile source into an AST. Source format (informal):

    # Options block (optional, key=value):
    MaxSessionHands = 10000
    AggressivePreFlop = 3
    ...

    custom
    preflop
    When stilltoact >= 5 and (pairinhand or hand = AK) useropenrangeutg
    When raises = 0 and useropenrangeutg raise 2 force
    When raises = 1 and amounttocall <= 3 call force
    when (hand = AA or hand = KK) and amounttocall <= 4 raisemax force delay 3
    When others
        when others fold force

    flop
    When havetoppair and not (flushpossible or paironboard) raisepot force
    When havepair and bets = 0 and raises = 0 bet 50% force
    When others
        when others fold force

    turn
    ...

    river
    ...

The language has these notable features:
  - Case-insensitive everything (keywords, identifiers, predicates).
  - Sections delimited by SECTION_KEYWORD lines: preflop, flop, turn, river.
    (An optional `custom` keyword precedes the section list when the file
    has custom rules; options-only files have no `custom`/sections at all.)
  - Rules: `When <expr> <action> [force] [delay N]`. The `force` keyword is
    universally used in practice; we accept rules with or without it.
  - `When others` followed by an indented inner rule starting with `when others`
    is a section-terminator convention: "if no rule above matched, do X."
    The inner action is the section default.
  - Expressions: boolean combinations with `and`, `or`, `not`, parens, around
    comparisons (`<lhs> <op> <rhs>`), bare boolean predicates (`havetoppair`),
    and user-flag references (`useropenrangeutg`).
  - Operators: `=`, `<`, `<=`, `>`, `>=`, `<>`. Right-hand sides can be
    numbers, hand specifications (`AK`, `AK suited`, `Kd 9d`), board
    specifications (`AKQ`, `A`), other identifiers, or percentage expressions
    (`50% potsize`, `21% stacksize`).
  - Actions: `fold`, `call`, `check`, `raise <N>`, `raise <N>%`, `raisemin`,
    `raisepot`, `raisemax`, `bet <N>`, `bet <N>% [target]`, `beep` (rare,
    appears in beep.txt — likely a debug/test bot).
  - User flags: any identifier starting with `user` used as an action sets
    that flag true for the current decision; used as an expression atom,
    tests whether a prior rule on this decision set it.
  - Comments: `//` to end of line. Blank lines ignored.

This parser builds an AST and validates basic structure but does not
execute or evaluate anything. The Profile object returned can be inspected
by the runtime evaluator (added in a later commit).

Design notes:
  - Hand-written recursive-descent parser. Grammar is simple enough that
    a generator (lark, PLY, etc.) would be overkill.
  - All keywords/identifiers normalize to lowercase. Case in source is
    purely a stylistic convention.
  - The parser is permissive about whitespace and rule ordering. It is
    strict about structural correctness (well-formed expressions, valid
    action keywords, balanced parens).
  - Errors carry a ParseError with line/column of the offending token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


# ============================================================
# Tokens
# ============================================================

class TokenKind(Enum):
    """Token categories produced by the tokenizer."""
    KEYWORD = "keyword"         # when, and, or, not, force, delay, others, custom, section names
    IDENT = "ident"             # everything else alphabetic — predicates, actions, user-flags
    NUMBER = "number"           # integer or decimal literal
    OP = "op"                   # = < <= > >= <>
    LPAREN = "lparen"
    RPAREN = "rparen"
    PERCENT = "percent"         # % sign (used in 50% potsize)
    EOF = "eof"


# Lowercase-normalized keywords. These never appear as predicate or action
# names, so reserving them is safe.
_KEYWORDS = frozenset({
    "when", "and", "or", "not", "force", "delay",
    "others",
    "custom",
    "preflop", "flop", "turn", "river",
    "in",       # `in smallblind`, `in bigblind`, `in button` — 2-token position predicate
})

# Section names — subset of keywords above.
_SECTION_NAMES = frozenset({"preflop", "flop", "turn", "river"})


@dataclass
class Token:
    """One lexed token, with source position for error reporting."""
    kind: TokenKind
    value: str         # always lowercase for KEYWORD/IDENT; raw text for NUMBER/OP
    line: int          # 1-indexed
    col: int           # 1-indexed

    def __repr__(self) -> str:
        return f"Token({self.kind.value}, {self.value!r}, line={self.line})"


class ParseError(Exception):
    """Raised on any malformed source — tokenizer or parser."""

    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(f"{msg} (line {line}, col {col})" if line else msg)
        self.msg = msg
        self.line = line
        self.col = col


# ============================================================
# Tokenizer
# ============================================================

# Match a token from the current position. The order matters: longer/multi-char
# tokens before single-char ones (<= before <, etc.).
#
# The `card_numeric` pattern catches `2c`, `9d`, `8s` etc. — cards with
# numeric ranks. Without it, `9c` would tokenize as NUMBER(9) + IDENT(c)
# because the number regex matches first. Letter-rank cards like `Tc`,
# `Kd`, `As` are already covered by the `ident` regex below (they're
# 2-char alphabetic tokens). This pattern produces an IDENT token so
# downstream parsing handles all card forms uniformly.
_TOKEN_PATTERNS = [
    ("ws",      re.compile(r"[ \t\r]+")),
    ("newline", re.compile(r"\n")),
    ("comment", re.compile(r"//[^\n]*")),
    ("card_numeric", re.compile(r"[2-9][cdhs]")),
    ("number",  re.compile(r"\d+(?:\.\d+)?")),
    ("op",      re.compile(r"<>|<=|>=|=|<|>")),
    ("lparen",  re.compile(r"\(")),
    ("rparen",  re.compile(r"\)")),
    ("percent", re.compile(r"%")),
    # Identifiers / keywords: letter/underscore + alnum/underscore.
    # Note: identifiers can begin with digit-letter combos for things like
    # "have2ndtoppair" — we match those by allowing digits internally.
    ("ident",   re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")),
]


def tokenize(source: str) -> List[Token]:
    """Convert source text to a flat list of Tokens, ending with EOF.

    Lowercases all KEYWORD/IDENT values; passes NUMBER/OP through as-is.
    Skips whitespace, newlines, and comments. Raises ParseError on
    unrecognized characters.

    A UTF-8 BOM (`\ufeff`) at the very start of the source is silently
    stripped — several Shanky distributions ship profiles with a BOM
    that would otherwise fail tokenization.

    Pre-tokenization normalization:
      - `<rank><rank>suited` and `<rank><rank>offsuit` tokens are common
        in some files (`a6suited`) where the writer omitted a space.
        We split these into `<rank><rank> suited`.
      - The literal token `orhand` (a known typo for `or hand`) is split.
    """
    # Strip BOM if present.
    if source.startswith("\ufeff"):
        source = source[1:]

    # Pre-tokenization normalizations. These fix known concatenation
    # artifacts in upstream profile files. Order matters — apply more-specific
    # patterns first.
    # `<rank-letter><rank-or-suited/offsuit>suited` etc.
    # e.g., 'A6suited' -> 'A6 suited'; 'AKoffsuit' -> 'AK offsuit'.
    source = re.sub(
        r"\b([AKQJT2-9][AKQJT2-9])(suited|offsuit)\b",
        r"\1 \2",
        source,
        flags=re.IGNORECASE,
    )
    # Single-rank wildcard with concatenated suited/offsuit: 'Ksuited' -> 'K suited'
    # but only when it's clearly a rank letter and not part of a longer identifier.
    # We can't reliably detect this without false positives on words like
    # 'BotsLastAction', so only handle the most common case in `hand = X<modifier>`.
    # (Skipped: too risky for marginal benefit.)
    # `orhand` typo in kamakazi.txt and possibly others.
    source = re.sub(r"\borhand\b", r"or hand", source, flags=re.IGNORECASE)

    tokens: List[Token] = []
    line = 1
    col = 1
    i = 0
    n = len(source)

    while i < n:
        matched = False
        for kind, pat in _TOKEN_PATTERNS:
            m = pat.match(source, i)
            if not m:
                continue
            text = m.group(0)
            start_col = col
            if kind == "ws" or kind == "comment":
                col += len(text)
            elif kind == "newline":
                line += 1
                col = 1
            elif kind == "card_numeric":
                # `9c`, `8s`, etc. — treat as an IDENT so downstream
                # card-parsing handles uniformly with letter-rank cards.
                tokens.append(Token(TokenKind.IDENT, text.lower(), line, start_col))
                col += len(text)
            elif kind == "number":
                tokens.append(Token(TokenKind.NUMBER, text, line, start_col))
                col += len(text)
            elif kind == "op":
                tokens.append(Token(TokenKind.OP, text, line, start_col))
                col += len(text)
            elif kind == "lparen":
                tokens.append(Token(TokenKind.LPAREN, "(", line, start_col))
                col += 1
            elif kind == "rparen":
                tokens.append(Token(TokenKind.RPAREN, ")", line, start_col))
                col += 1
            elif kind == "percent":
                tokens.append(Token(TokenKind.PERCENT, "%", line, start_col))
                col += 1
            elif kind == "ident":
                low = text.lower()
                if low in _KEYWORDS:
                    tokens.append(Token(TokenKind.KEYWORD, low, line, start_col))
                else:
                    tokens.append(Token(TokenKind.IDENT, low, line, start_col))
                col += len(text)
            i = m.end()
            matched = True
            break
        if not matched:
            raise ParseError(
                f"unexpected character {source[i]!r}", line=line, col=col,
            )

    tokens.append(Token(TokenKind.EOF, "", line, col))
    return tokens


# ============================================================
# AST node definitions
# ============================================================

# ---- Expressions ----

@dataclass
class BoolOp:
    """`a and b`, `a or b`. Multi-arity (chained ands/ors flattened)."""
    op: str             # 'and' | 'or'
    args: List["Expr"]


@dataclass
class Not:
    """`not x`"""
    arg: "Expr"


@dataclass
class Compare:
    """`lhs <op> rhs` — predicates like `stacksize <= 10`, `hand = AK`,
    `amounttocall <= 50% potsize`."""
    lhs: str            # an identifier (predicate name on the left)
    op: str             # '=' '<' '<=' '>' '>=' '<>'
    rhs: "RhsValue"


@dataclass
class PredCall:
    """Bare boolean predicate (no comparison): `havetoppair`, `flushpossible`,
    `pairinhand`, etc. Also covers user-flag tests like `userflopcbet`.
    """
    name: str


@dataclass
class PositionPred:
    """Two-token position predicate: `in smallblind`, `in bigblind`, `in button`.

    Semantically a boolean predicate equivalent to `position = <pos>`, but
    Shanky writes it as `in <position>` so we recognize that form natively.
    """
    position: str    # 'smallblind' | 'bigblind' | 'button' | other position name


@dataclass
class OthersAtom:
    """The `others` atom — appears in `when others and <more conditions>` form.

    Semantically a boolean predicate that is always true; functions as a
    placeholder so additional conditions can be ANDed in. The runtime treats
    OthersAtom as True; the rule still fires only if it's reached (i.e., no
    prior rule matched), which gives the standard 'when others' semantic.
    """
    pass


# RHS of a comparison.
@dataclass
class NumberLit:
    value: float
    is_int: bool        # if it parsed as integer, preserve for round-trip


@dataclass
class IdentLit:
    """Bare identifier on RHS — e.g. `position = first`, `botslastaction = call`."""
    name: str


@dataclass
class PercentExpr:
    """`50% potsize`, `21% stacksize`. The pct number and the target."""
    pct: float
    target: str         # 'potsize' | 'stacksize' | other identifier


@dataclass
class HandSpec:
    """A poker hand specification: `AK`, `AKs`, `AK suited`, `K9 offsuit`,
    `Kd 9d`, `A` (wildcard meaning A-x), `A suited`, etc.

    Internally normalized to a list of card tokens plus a suitedness flag:
      - cards: list of (rank, suit) tuples. suit is None when unspecified.
      - suitedness: 'suited' | 'offsuit' | None (=either)

    Wildcard forms (single rank without partner, or rank with `suited`) are
    represented with a single card. Two-card forms always have two cards.
    """
    cards: List["Card"]
    suitedness: Optional[str] = None     # 'suited' | 'offsuit' | None


@dataclass
class Card:
    """One card in a HandSpec or BoardSpec.
    rank: 'A','K','Q','J','T','9','8','7','6','5','4','3','2'
    suit: 'c','d','h','s' or None (unspecified rank-only)
    """
    rank: str
    suit: Optional[str] = None


@dataclass
class BoardSpec:
    """Board specification on RHS of `board = ...`. Examples:
       `board = A`          (any board with an ace)
       `board = AKQ`        (specific three-rank board, suits unspecified)
       `board = AAA`        (trips on board)
       `board = 2456`       (four-card board)
    Internally a list of ranks (suits not specified in any profile we've seen).
    """
    ranks: List[str]


RhsValue = Union[NumberLit, IdentLit, PercentExpr, HandSpec, BoardSpec]
Expr = Union[BoolOp, Not, Compare, PredCall, PositionPred, OthersAtom]


# ---- Actions ----

class ActionKind(Enum):
    FOLD = "fold"
    CALL = "call"
    CHECK = "check"
    RAISE_AMOUNT = "raise_amount"         # raise N (chips/BB-multiples)
    RAISE_PERCENT = "raise_percent"       # raise N% [target]
    RAISE_MIN = "raisemin"
    RAISE_POT = "raisepot"
    RAISE_MAX = "raisemax"
    BET_AMOUNT = "bet_amount"
    BET_PERCENT = "bet_percent"
    BEEP = "beep"                          # appears in beep.txt only
    SITOUT = "sitout"                      # `sitout force` — voluntarily sit out hand
    BET_MIN = "betmin"                     # `betmin` — minimum bet (sibling of raisemin)
    BET_POT = "betpot"                     # `betpot` — pot-sized bet (sibling of raisepot)
    BET_MAX = "betmax"                     # `betmax` — max bet (sibling of raisemax)
    SET_USER_FLAG = "set_user_flag"        # `userfoo` as action -> sets flag

    # Section-default-action sentinel: section terminator (`When others`).
    SECTION_DEFAULT = "section_default"


@dataclass
class Action:
    """A rule's action component.

    kind: which sort of action
    amount: numeric amount (for RAISE_AMOUNT, RAISE_PERCENT, BET_AMOUNT, BET_PERCENT)
    amount_target: for percent actions, the reference quantity (potsize/stacksize)
    user_flag: for SET_USER_FLAG, the flag name
    force: True if `force` modifier present (universally yes in practice)
    delay: optional integer delay (`delay N` modifier, used by some profiles)
    """
    kind: ActionKind
    amount: Optional[float] = None
    amount_target: Optional[str] = None        # 'potsize' | 'stacksize' | None
    user_flag: Optional[str] = None
    force: bool = False
    delay: Optional[int] = None


# ---- Rule, Section, Profile ----

@dataclass
class Rule:
    """One `When <expr> <action>` line."""
    condition: Expr
    action: Action
    line: int                                  # source line for diagnostics


@dataclass
class Section:
    """A street section: preflop, flop, turn, or river."""
    name: str                                  # 'preflop' | 'flop' | 'turn' | 'river'
    rules: List[Rule]
    # Default action when no rule matches. Defined by trailing `When others / when others <action> force`.
    default_action: Optional[Action] = None


@dataclass
class Profile:
    """Parsed bot profile.

    options: top-of-file key=value pairs (may be empty)
    sections: list of sections in declaration order. Empty if file is options-only.
    has_custom: True if `custom` keyword appears (i.e., there are real rules
        to interpret; otherwise the file is configuring the engine's
        built-in Doodle strategy via options only).
    user_flags: set of all user-flag names referenced anywhere (as actions
        or tested in expressions). Useful for the runtime to validate
        consistency.
    """
    options: dict                              # str -> str (raw values)
    sections: List[Section]
    has_custom: bool
    user_flags: set = field(default_factory=set)
    source_name: Optional[str] = None          # filename, for diagnostics


# ============================================================
# Parser
# ============================================================

# Known boolean predicates (no RHS). These are recognized as PredCall
# rather than IdentLit. Source: catalog of predicate identifiers seen
# across the 36 profiles. Not exhaustive — unknown identifiers used as
# bare expression atoms are treated as PredCall too, since user-flag
# references (testing whether `userfoo` was set) work the same way
# syntactically.
_KNOWN_PREDICATES = frozenset({
    # Hand-class predicates
    "pairinhand", "havepair", "havetoppair", "haveoverpair", "have2ndtoppair",
    "have2ndoverpair", "have3rdtoppair", "havetwopair", "havetoptwopair",
    "haveset", "havebottomset", "havetrips", "havestraight", "haveflush",
    "havefullhouse", "havequads", "havestraightflush", "havenuts",
    "havenutflush", "havenutflushdraw", "havenutstraight", "havestraightdraw",
    "haveinsidestraightdraw", "haveflushdraw", "havebackdoorflushdraw",
    "havebestkicker", "have2ndbestkicker", "have3rdbestkicker",
    "have2ndnutflush", "have3rdnutflush",
    # Board-shape predicates
    "flushpossible", "straightpossible", "onecardflushpossible",
    "onecardstraightpossible", "morethanonestraightpossibleonturn",
    "paironboard", "twopaironboard", "tripsonboard", "straightonboard",
    "onecardstraightpossibleonturn", "paironflop",
    # Betting-state predicates
    "opponentisallin", "botislastraiser", "insmallblind", "inbigblind",
    "inbutton",
    "botraisedonflop", "botraisedbeforeflop", "raisesonflop", "raisesonturn",
    "nobettingonflop", "nobettingonturn", "calledonflop",
})


class Parser:
    """Recursive-descent parser. One instance per parse.

    The parser maintains a position cursor into the token list. Methods
    consume tokens and advance the cursor. On unexpected input, raises
    ParseError with the offending token's line/col.
    """

    def __init__(self, tokens: List[Token], source_name: Optional[str] = None):
        self.tokens = tokens
        self.pos = 0
        self.source_name = source_name
        self.user_flags: set = set()

    # ---- Token cursor helpers ----

    def _peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def _next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _eat(self, kind: TokenKind, value: Optional[str] = None) -> Token:
        tok = self._peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = f"{kind.value}({value!r})" if value else kind.value
            raise ParseError(
                f"expected {expected}, got {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )
        return self._next()

    def _accept(self, kind: TokenKind, value: Optional[str] = None) -> Optional[Token]:
        tok = self._peek()
        if tok.kind == kind and (value is None or tok.value == value):
            return self._next()
        return None

    # ---- Top-level ----

    def parse_profile(self) -> Profile:
        """Parse top-level: optional options block, optional 'custom',
        then zero-or-more sections, then EOF."""
        options = self._parse_options_block()
        has_custom = False
        if self._accept(TokenKind.KEYWORD, "custom"):
            has_custom = True

        sections = []
        while self._peek().kind == TokenKind.KEYWORD and self._peek().value in _SECTION_NAMES:
            sections.append(self._parse_section())

        # Anything left should be EOF.
        if self._peek().kind != TokenKind.EOF:
            tok = self._peek()
            raise ParseError(
                f"unexpected trailing content: {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )

        return Profile(
            options=options,
            sections=sections,
            has_custom=has_custom,
            user_flags=self.user_flags,
            source_name=self.source_name,
        )

    # ---- Options block ----

    def _parse_options_block(self) -> dict:
        """Parse zero-or-more `KEY = VALUE` lines.

        Stops at the first token that isn't an identifier followed by `=`,
        which heralds either the `custom` keyword or a section header.
        Values are parsed loosely — we accept idents, numbers, and the
        special ON/OFF tokens that Shanky uses for boolean options.
        """
        options = {}
        while True:
            t0 = self._peek(0)
            t1 = self._peek(1)
            # An options line looks like IDENT '=' (NUMBER | IDENT). If next
            # token isn't IDENT or the one after isn't '=', we're done.
            if t0.kind != TokenKind.IDENT:
                break
            if not (t1.kind == TokenKind.OP and t1.value == "="):
                break
            # Some IDENTs are also action/predicate names ('call', 'fold', etc.)
            # — but those wouldn't be followed by '=' at top level, so the
            # lookahead-2 check above is sufficient.
            key_tok = self._next()
            self._eat(TokenKind.OP, "=")
            # Value: number, identifier, or sequence of identifiers (rare).
            val_parts = []
            while True:
                vt = self._peek()
                if vt.kind == TokenKind.NUMBER:
                    val_parts.append(self._next().value)
                elif vt.kind == TokenKind.IDENT:
                    # Stop if this looks like the start of another `KEY = VALUE`
                    # pair (heuristic: ident followed by '=').
                    nxt = self._peek(1)
                    if nxt.kind == TokenKind.OP and nxt.value == "=":
                        break
                    val_parts.append(self._next().value)
                else:
                    break
                if not val_parts:
                    break
            if not val_parts:
                raise ParseError(
                    f"missing value for option {key_tok.value!r}",
                    line=key_tok.line, col=key_tok.col,
                )
            options[key_tok.value] = " ".join(val_parts)
        return options

    # ---- Section ----

    def _parse_section(self) -> Section:
        """Parse a section: name keyword, then rules until next section or EOF.

        Handles three rule forms uniformly:
          1. Regular rule: `when <expr> <action> [force] [delay N]`
          2. Scoped block header: `when <expr>` (no action) — scopes subsequent
             rules until the next scope header or section end.
          3. `when others ...` (any of the above) — `others` is parsed as an
             atom that the runtime treats as "no rule above this one matched".

        Form 3 unifies what was previously a special case. The resulting AST
        has `others` appearing as an OthersAtom in rule conditions, and the
        runtime handles the semantics. A regular `when others fold force`
        becomes a rule with condition=OthersAtom and action=FOLD; a header
        like `when others and bigblindsize < 40` becomes a scope header with
        scope_cond = BoolOp(and, [OthersAtom, Compare(bigblindsize<40)]).
        """
        name_tok = self._eat(TokenKind.KEYWORD)
        if name_tok.value not in _SECTION_NAMES:
            raise ParseError(
                f"expected section name, got {name_tok.value!r}",
                line=name_tok.line, col=name_tok.col,
            )

        rules: List[Rule] = []
        current_scope_cond: Optional[Expr] = None

        while True:
            t = self._peek()
            if t.kind == TokenKind.EOF:
                break
            if t.kind == TokenKind.KEYWORD and t.value in _SECTION_NAMES:
                break
            if t.kind != TokenKind.KEYWORD or t.value != "when":
                raise ParseError(
                    f"expected 'when' to start a rule, got {t.kind.value}({t.value!r})",
                    line=t.line, col=t.col,
                )

            # Eat `when` and parse the expression. Note: the expression parser
            # handles `others` natively (as OthersAtom), `in <pos>`, and all
            # other forms — no special case needed here.
            when_tok = self._eat(TokenKind.KEYWORD, "when")
            cond = self._parse_expression()

            # Is the next token an action-position token, or did this rule
            # have no action (= scope header)?
            nxt = self._peek()
            is_scope_header = (
                nxt.kind == TokenKind.EOF
                or (nxt.kind == TokenKind.KEYWORD
                    and (nxt.value == "when" or nxt.value in _SECTION_NAMES))
            )
            if is_scope_header:
                current_scope_cond = cond
                continue

            # Normal rule: parse action and modifiers.
            action = self._parse_action()
            self._maybe_eat_force_into(action)
            delay = self._maybe_eat_delay()
            if delay is not None:
                action.delay = delay

            # Apply current scope (if any).
            effective_cond = cond
            if current_scope_cond is not None:
                effective_cond = BoolOp(op="and", args=[current_scope_cond, cond])

            rules.append(Rule(condition=effective_cond, action=action, line=when_tok.line))

        # Compute default_action for convenience — the last rule in the
        # list whose condition includes OthersAtom (at root, or AND-ed
        # with a scope) acts as a default. The runtime can use this as a
        # shortcut; otherwise it's a regular rule and rule-list iteration
        # handles it naturally.
        default_action = None
        if rules and _contains_others_atom(rules[-1].condition):
            default_action = rules[-1].action

        return Section(name=name_tok.value, rules=rules, default_action=default_action)

    # ---- Modifiers (force, delay) ----

    def _maybe_eat_force(self) -> bool:
        if self._accept(TokenKind.KEYWORD, "force"):
            return True
        return False

    def _maybe_eat_force_into(self, action: Action) -> None:
        if self._maybe_eat_force():
            action.force = True

    def _maybe_eat_delay(self) -> Optional[int]:
        if self._accept(TokenKind.KEYWORD, "delay"):
            n_tok = self._eat(TokenKind.NUMBER)
            try:
                return int(float(n_tok.value))
            except ValueError:
                raise ParseError(
                    f"delay must be an integer, got {n_tok.value!r}",
                    line=n_tok.line, col=n_tok.col,
                )
        return None

    # ---- Expression parsing (precedence: not > and > or) ----

    def _parse_expression(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        args = [left]
        while self._accept(TokenKind.KEYWORD, "or"):
            args.append(self._parse_and())
        if len(args) == 1:
            return left
        return BoolOp(op="or", args=args)

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        args = [left]
        while self._accept(TokenKind.KEYWORD, "and"):
            args.append(self._parse_not())
        if len(args) == 1:
            return left
        return BoolOp(op="and", args=args)

    def _parse_not(self) -> Expr:
        if self._accept(TokenKind.KEYWORD, "not"):
            return Not(arg=self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        tok = self._peek()
        if tok.kind == TokenKind.LPAREN:
            self._next()
            inner = self._parse_expression()
            self._eat(TokenKind.RPAREN)
            return inner

        # `in <position>` — two-token position predicate.
        if tok.kind == TokenKind.KEYWORD and tok.value == "in":
            self._next()
            pos_tok = self._peek()
            if pos_tok.kind != TokenKind.IDENT:
                raise ParseError(
                    f"expected position name after 'in', got {pos_tok.kind.value}({pos_tok.value!r})",
                    line=pos_tok.line, col=pos_tok.col,
                )
            self._next()
            return PositionPred(position=pos_tok.value)

        # `others` as a bare atom (e.g., `when others and <more conditions>`).
        # We don't normally see `others` outside the section-terminator context
        # but the section-terminator handler in _parse_when_others
        # consumes 'when' 'others' before reaching here. This branch handles
        # the case where `others` appears mid-expression after the conditional
        # parts have already been parsed (shouldn't happen, defensive).
        if tok.kind == TokenKind.KEYWORD and tok.value == "others":
            self._next()
            return OthersAtom()

        if tok.kind != TokenKind.IDENT:
            raise ParseError(
                f"expected predicate or '(', got {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )

        lhs_tok = self._next()
        lhs = lhs_tok.value

        # Lookahead: is this `IDENT OP RHS` (a comparison) or a bare predicate?
        nxt = self._peek()
        if nxt.kind == TokenKind.OP:
            op_tok = self._next()
            rhs = self._parse_rhs(lhs)
            return Compare(lhs=lhs, op=op_tok.value, rhs=rhs)

        # Bare predicate. Track user-flag references.
        if lhs.startswith("user"):
            self.user_flags.add(lhs)
        return PredCall(name=lhs)

    def _parse_rhs(self, lhs_hint: str) -> RhsValue:
        """Parse RHS of a comparison.

        lhs_hint disambiguates hand/board-shape parsers: when LHS is 'hand'
        the RHS is a HandSpec; when LHS is 'board' it's a BoardSpec; else
        it's number/ident/percent.
        """
        tok = self._peek()

        if lhs_hint == "hand":
            return self._parse_hand_spec()
        if lhs_hint == "board":
            return self._parse_board_spec()

        if tok.kind == TokenKind.NUMBER:
            num_tok = self._next()
            value = float(num_tok.value)
            is_int = "." not in num_tok.value
            # Check for percent suffix: '50% potsize'
            if self._accept(TokenKind.PERCENT):
                # Target is optional — `amounttocall <= 5%` appears, but
                # `<= 50% potsize` is more common.
                target = "potsize"  # default
                if self._peek().kind == TokenKind.IDENT:
                    target = self._next().value
                return PercentExpr(pct=value, target=target)
            return NumberLit(value=value, is_int=is_int)

        if tok.kind == TokenKind.IDENT:
            ident_tok = self._next()
            return IdentLit(name=ident_tok.value)

        raise ParseError(
            f"expected number/identifier on RHS, got {tok.kind.value}({tok.value!r})",
            line=tok.line, col=tok.col,
        )

    def _parse_hand_spec(self) -> HandSpec:
        """Parse RHS of `hand = ...`.

        Forms seen in profiles:
          AA / KK / 22                   -> pair, both cards same rank
          AK / AQ                        -> two-rank, unsuited (=either)
          AK suited / AK offsuit         -> two-rank with suitedness modifier
          A K suited                     -> with whitespace (parsed as multi tokens)
          AKs / AKo                      -> compact suited/offsuit form
          Kd 9d                          -> fully-specified cards with suits
          A                              -> wildcard: any A-x
          A suited                       -> wildcard suited: any A-x suited

        Strategy: greedily consume tokens that look like card-rank or rank-suit
        until we hit something that isn't, then check for suited/offsuit suffix.
        """
        cards: List[Card] = []

        while True:
            tok = self._peek()
            if tok.kind != TokenKind.IDENT and tok.kind != TokenKind.NUMBER:
                break
            text = tok.value
            # Try to interpret as 1-card or 2-card token.
            parsed = _parse_card_token(text)
            if parsed is None:
                break
            self._next()
            cards.extend(parsed)

        if not cards:
            tok = self._peek()
            raise ParseError(
                f"expected a hand specification, got {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )

        suitedness = None
        if self._peek().kind == TokenKind.IDENT:
            t = self._peek().value
            if t in ("suited", "offsuit"):
                self._next()
                suitedness = t

        return HandSpec(cards=cards, suitedness=suitedness)

    def _parse_board_spec(self) -> BoardSpec:
        """Parse RHS of `board = ...`.

        Forms in profiles:
          `board = A`              (any board with an ace)
          `board = AKQ`            (specific three-rank board, suits unspecified)
          `board = A K Q`          (same but space-separated)
          `board = 2456`           (four-card board)
          `board = AAA`            (trips on board)

        We greedily consume tokens (IDENT or NUMBER) that look entirely like
        rank characters, concatenating them into one rank string. Stops at
        any token that doesn't parse cleanly as ranks (operators, keywords,
        identifiers with non-rank chars, etc.).
        """
        valid_ranks = set("AKQJT98765432")
        ranks: List[str] = []
        first = True

        while True:
            tok = self._peek()
            if tok.kind != TokenKind.IDENT and tok.kind != TokenKind.NUMBER:
                break
            text = tok.value.upper()
            # Every char must be a rank for this token to be part of the board.
            if not all(ch in valid_ranks for ch in text):
                if first:
                    # First token must be at least partly valid — otherwise
                    # raise a clean error.
                    raise ParseError(
                        f"invalid rank(s) in board spec {tok.value!r}",
                        line=tok.line, col=tok.col,
                    )
                break
            self._next()
            ranks.extend(text)
            first = False

        if not ranks:
            tok = self._peek()
            raise ParseError(
                f"expected board ranks, got {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )

        return BoardSpec(ranks=ranks)

    # ---- Action ----

    def _parse_action(self) -> Action:
        """Parse the action part of a rule.

        After expression parsing, we expect one of:
          fold | call | check
          raise <N> [%]
          raisemin | raisepot | raisemax
          bet <N> [%] [target]
          beep
          user<flag>            (sets the named flag true)
        """
        tok = self._peek()

        if tok.kind != TokenKind.IDENT:
            raise ParseError(
                f"expected action keyword, got {tok.kind.value}({tok.value!r})",
                line=tok.line, col=tok.col,
            )

        name = tok.value

        # User-flag setter: any identifier starting with 'user'.
        if name.startswith("user"):
            self._next()
            self.user_flags.add(name)
            return Action(kind=ActionKind.SET_USER_FLAG, user_flag=name)

        if name == "fold":
            self._next()
            return Action(kind=ActionKind.FOLD)
        if name == "call":
            self._next()
            return Action(kind=ActionKind.CALL)
        if name == "check":
            self._next()
            return Action(kind=ActionKind.CHECK)
        if name == "beep":
            self._next()
            return Action(kind=ActionKind.BEEP)
        if name == "sitout":
            self._next()
            return Action(kind=ActionKind.SITOUT)
        if name == "raisemin":
            self._next()
            return Action(kind=ActionKind.RAISE_MIN)
        if name == "raisepot":
            self._next()
            return Action(kind=ActionKind.RAISE_POT)
        if name == "raisemax":
            self._next()
            return Action(kind=ActionKind.RAISE_MAX)
        if name == "betmin":
            self._next()
            return Action(kind=ActionKind.BET_MIN)
        if name == "betpot":
            self._next()
            return Action(kind=ActionKind.BET_POT)
        if name == "betmax":
            self._next()
            return Action(kind=ActionKind.BET_MAX)

        if name == "raise":
            self._next()
            # Bare `raise` (no amount) is valid: e.g. `raise force` — the
            # engine uses its default raise size. Detect by looking at the
            # next token — if it's not a number, this is the bare form.
            if self._peek().kind != TokenKind.NUMBER:
                return Action(kind=ActionKind.RAISE_AMOUNT, amount=None)
            return self._parse_amount_action(ActionKind.RAISE_AMOUNT, ActionKind.RAISE_PERCENT)
        if name == "bet":
            self._next()
            # Bare `bet` (no amount): same as bare raise — use default.
            if self._peek().kind != TokenKind.NUMBER:
                return Action(kind=ActionKind.BET_AMOUNT, amount=None)
            return self._parse_amount_action(ActionKind.BET_AMOUNT, ActionKind.BET_PERCENT)

        raise ParseError(
            f"unknown action {name!r}",
            line=tok.line, col=tok.col,
        )

    def _parse_amount_action(self, abs_kind: ActionKind, pct_kind: ActionKind) -> Action:
        """Used by `raise N[%]` and `bet N[%]` to parse the amount portion.

        After the action keyword, expects a number; then optional `%` with
        optional target identifier.
        """
        n_tok = self._eat(TokenKind.NUMBER)
        amount = float(n_tok.value)
        if self._accept(TokenKind.PERCENT):
            target = None
            if self._peek().kind == TokenKind.IDENT:
                t = self._peek().value
                # Stop if it's the `force` keyword (which is a KEYWORD, not IDENT,
                # so this branch wouldn't fire — but defensively check).
                # Common targets: potsize, stacksize.
                if t in ("potsize", "stacksize"):
                    self._next()
                    target = t
            return Action(kind=pct_kind, amount=amount, amount_target=target)
        return Action(kind=abs_kind, amount=amount)


# ============================================================
# Card-token parsing helper
# ============================================================

_VALID_RANKS = set("AKQJT98765432")
_VALID_SUITS = set("cdhs")


def _parse_card_token(text: str) -> Optional[List[Card]]:
    """Try to interpret a single token as one or more cards.

    Accepts (case-insensitive):
      'A'      -> single-card wildcard
      'AK'     -> two ranks, unsuited
      'AKs'    -> two ranks, suited
      'AKo'    -> two ranks, offsuit
      'Kd'     -> single card with suit
      'Kd9d'   -> two cards with suits (rare; usually whitespace-separated)
      '22'     -> pair

    Returns a list of 1-2 Cards, or None if the token doesn't parse as cards.
    """
    if not text:
        return None
    text = text.upper()
    # Suit chars in source are lowercase by convention; we uppercased above,
    # so re-check against uppercase suit letters.
    valid_suits_upper = set("CDHS")

    # Strip optional trailing 's' or 'o' (suited / offsuit shorthand) — but
    # only if the rest forms a valid card spec.
    suitedness_suffix = None
    if len(text) >= 3 and text[-1] in ("S", "O"):
        # Risk: 'KS' could be K-of-spades (1 card) or K-then-S-suffix.
        # Disambiguate: if the prefix is two ranks, the trailing S/O is the
        # suitedness suffix; otherwise S is a suit letter.
        prefix = text[:-1]
        if len(prefix) == 2 and prefix[0] in _VALID_RANKS and prefix[1] in _VALID_RANKS:
            suitedness_suffix = text[-1]
            text = prefix

    # Now text is one of:
    #   length 1: rank
    #   length 2: 'rr' (two ranks) or 'rs' (rank+suit)
    #   length 4: 'rsrs' (two suit-specified cards)
    if len(text) == 1:
        if text in _VALID_RANKS:
            return [Card(rank=text)]
        return None
    if len(text) == 2:
        if text[0] in _VALID_RANKS and text[1] in _VALID_RANKS:
            return [Card(rank=text[0]), Card(rank=text[1])]
        if text[0] in _VALID_RANKS and text[1] in valid_suits_upper:
            return [Card(rank=text[0], suit=text[1].lower())]
        return None
    if len(text) == 4:
        if (text[0] in _VALID_RANKS and text[1] in valid_suits_upper
                and text[2] in _VALID_RANKS and text[3] in valid_suits_upper):
            return [
                Card(rank=text[0], suit=text[1].lower()),
                Card(rank=text[2], suit=text[3].lower()),
            ]
        return None
    return None


def _contains_others_atom(expr: Expr) -> bool:
    """Return True iff `expr` includes an OthersAtom anywhere in its tree.

    Used by section parsing to flag rules whose condition contains an
    `others` reference — those are the section's "final fallback" rules
    and should be exposed as the section default for runtime convenience.
    """
    if isinstance(expr, OthersAtom):
        return True
    if isinstance(expr, BoolOp):
        return any(_contains_others_atom(a) for a in expr.args)
    if isinstance(expr, Not):
        return _contains_others_atom(expr.arg)
    return False


# ============================================================
# Public entry point
# ============================================================

def parse_profile(source: str, source_name: Optional[str] = None) -> Profile:
    """Parse a Shanky profile from source text.

    Args:
        source: full text of the profile file.
        source_name: optional filename for diagnostics (e.g., 'timidtom.txt').

    Returns:
        Profile AST.

    Raises:
        ParseError: on malformed input. The exception carries line/col of
        the offending token.
    """
    tokens = tokenize(source)
    return Parser(tokens, source_name=source_name).parse_profile()
