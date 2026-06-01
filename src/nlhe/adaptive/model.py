"""6-max NLHE adaptive-policy model (scaffold step 1b).

The minimal scaffold's model class. Architecture decisions are locked
upstream:
  - Continuous tendency target (Decision A in scaffold proposal): the
    opp-classifier head emits a K=10 sigmoided tendency vector, not class
    logits. The gate's read is computed from this vector's L2 distance to
    the blueprint's reference tendency vector (Q1c: head-derived, oracle-
    free, in tendency space — no per-decision projection).
  - Distill dominant (Decision B): lam_distill = 1.0 hard-locked; the only
    aux lever is lam_aux. (Enforced at training-script start in step 3, not
    in this module.)
  - match_conf multiplier (Q4): g_total = match_conf · g_within_hand, with
    match_conf from MatchObserver.confidence (0 at n_actions<20 → 1 by
    n>=300). The model exposes the raw heads; the inference-time blend
    combines them with anchor + match_conf in the training/inference script
    (kept out of forward() to honor the leak-clean signature surface).

Reusable correctness surface (per docs/DECISIONS.md → "Leduc proof complete
— S1 verdict" → "Leakage invariant + Tests A/B/C/D"):
  - tokens (the InfosetEncoder6Max output sequence) is §8/Test-A6/B6-clean
    by construction (proved by the preflight tests, commit 79c3b3a).
  - opp_stats is a SeatStats-derived vector (public actions only, §8-clean
    by construction per within_match.py docstring + Test C6).
  - forward() signature accepts only the 5 declared inputs — no tendency
    target / archetype id / oracle parameter. Enforced by Test D6.

Forward returns:
  policy_raw     [B, NUM_ACTIONS=9]  — softmax(masked policy logits)
  tendency_pred  [B, K_TENDENCY=10]  — sigmoided per-dim
  combined       [B, d_model + d_stats_proj]  — trunk repr, for inspection
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Constants (frozen surface; bumping any of these is a schema change that
# invalidates trained checkpoints — bump together with all dependent tests
# and pinned manifests).
# --------------------------------------------------------------------------

F_TOKEN: int = 236        # InfosetEncoder6Max.feature_dim (Test A6.2 pin)
F_STATS: int = 12         # per-target-opponent live SeatStats input dim
K_TENDENCY: int = 10      # continuous opponent tendency dimensions
NUM_ACTIONS: int = 9      # DiscreteAction count (FOLD..ALLIN)
NEG_INF: float = -1e9


# Concrete tendency layout (the 10 dims the head predicts, in order):
TENDENCY_LAYOUT: tuple[str, ...] = (
    "VPIP",         # 0
    "PFR",          # 1
    "AF_flop",      # 2
    "AF_turn",      # 3
    "AF_river",     # 4
    "F2B_preflop",  # 5
    "F2B_flop",     # 6
    "F2B_turn",     # 7
    "F2B_river",    # 8
    "mean_bet_size_over_pot",  # 9  (clipped/normalized at the SeatStats source)
)
assert len(TENDENCY_LAYOUT) == K_TENDENCY


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Adaptive6MaxNet(nn.Module):
    """6-max NLHE adaptive policy net (Adaptive6MaxNet).

    Trunk: transformer encoder over the sequence of `InfosetEncoder6Max`
    feature vectors at each public decision in the history-so-far. The query
    index selects the hero's current decision; the head outputs are taken at
    that position.

    Heads:
      policy_head        : Linear(d_model + d_stats_proj, NUM_ACTIONS=9)
                            → hero policy. Distill target.
      opp_head_tendency  : Linear(d_model + d_stats_proj, K_TENDENCY=10)
                            → sigmoided continuous tendency vector. Aux
                            target (auto-supervised against the live
                            MatchObserver SeatStats at training time).

    Default sizes (per the approved scaffold proposal, trunk A):
      d_model=128, nhead=4, num_layers=4, dim_ff=512, max_seq=128,
      d_stats_proj=32.

    NOTE: forward() accepts ONLY the 5 declared inputs. The match-confidence
    multiplier (Q4 g_total = match_conf · g_within_hand), the gate, and the
    confidence-gated blend live OUTSIDE forward — in the training/inference
    script — so the model surface stays §8/Test-D6-clean (no tendency-target
    / archetype-id / oracle parameter).
    """

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 4, dim_ff: int = 512,
                 dropout: float = 0.0, max_seq: int = 128,
                 d_stats_proj: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.d_stats_proj = d_stats_proj
        self.token_proj = nn.Linear(F_TOKEN, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.opp_stats_proj = nn.Linear(F_STATS, d_stats_proj)
        self.policy_head = nn.Linear(d_model + d_stats_proj, NUM_ACTIONS)
        self.opp_head_tendency = nn.Linear(
            d_model + d_stats_proj, K_TENDENCY)

    def forward(self, tokens, pad_mask, query_idx, legal_mask, opp_stats):
        """tokens [B,T,F_TOKEN=236] : sequence of encoder feature vectors.
        pad_mask [B,T] bool         : True = pad position (masked out).
        query_idx [B] long          : hero-current-decision index in T.
        legal_mask [B,NUM_ACTIONS]  : True = legal hero action.
        opp_stats [B,F_STATS=12]    : target-opp live SeatStats vector.

        Returns (policy_raw, tendency_pred, combined)."""
        B, T, _ = tokens.shape
        x = self.token_proj(tokens) + self.pos_embed[:T].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        repr_ = h[torch.arange(B, device=tokens.device), query_idx]
        s = self.opp_stats_proj(opp_stats)
        combined = torch.cat([repr_, s], dim=-1)
        logits = self.policy_head(combined)
        logits = logits.masked_fill(~legal_mask, NEG_INF)
        policy_raw = F.softmax(logits, dim=-1)
        tendency_pred = torch.sigmoid(self.opp_head_tendency(combined))
        return policy_raw, tendency_pred, combined


# --------------------------------------------------------------------------
# Read primitive (Q1c, head-derived, in tendency space, NO oracle)
# --------------------------------------------------------------------------

def tendency_l2_read(tendency_pred: np.ndarray,
                     tendency_blueprint: np.ndarray) -> float:
    """Per-decision read d_t = ||tendency_pred - tendency_blueprint||_2.

    Both inputs are K_TENDENCY-dim vectors in [0, 1]^K. tendency_blueprint
    is the blueprint's OWN expected-tendency reference, measured ONCE on
    blueprint self-play (see Q3 anchor + step-3 training script). NOT
    opponent-specific — it's the universal GTO reference, analogous to
    pi_GTO in the Leduc head-derived KL primitive. No oracle in the read
    path.

    Returns a non-negative scalar; bounded above by sqrt(K_TENDENCY).
    Returns 0 exactly when tendency_pred == tendency_blueprint (e.g. when
    the head correctly identifies a GTO-like opponent — gate stays closed
    by construction in that case)."""
    a = np.asarray(tendency_pred, dtype=np.float64)
    b = np.asarray(tendency_blueprint, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"tendency_l2_read: shape mismatch {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b, ord=2))


def gate_g_within_hand(d_bar: float, kappa: float = 1.0) -> float:
    """g_within_hand = tanh(kappa * D̄). gate(0) = 0 exactly; bounded in
    [0, 1). Same form as the Leduc validated gate."""
    return float(np.tanh(kappa * d_bar))


def gate_g_total(g_within_hand: float, match_conf: float) -> float:
    """Q4: g_total = match_conf · g_within_hand.

    match_conf is MatchObserver.confidence(target_seat) ∈ [0, 1] — ramps
    from 0 at n_actions<20 to 1.0 at n>=300. At match start (no evidence),
    match_conf = 0 → g_total = 0 → blend = anchor (floor by construction).
    This is THE 6-max-advantage-over-Leduc mechanism: the gate stays closed
    until match-level evidence accumulates, then opens."""
    return float(max(0.0, min(1.0, match_conf)) * g_within_hand)


def blend(g_total: float, anchor_probs: np.ndarray,
          raw_probs: np.ndarray) -> np.ndarray:
    """pi_final = (1 - g_total) * anchor + g_total * raw.

    At g_total=0 (match-conf or within-hand=0): pi_final = anchor exactly
    (the zero-evidence floor). Bounded by anchor on one side, raw on the
    other; never exceeds the anchor's exploitability when g_total=0."""
    return (1.0 - g_total) * anchor_probs + g_total * raw_probs


# --------------------------------------------------------------------------
# Batching: pack variable-length token sequences from a list of per-decision
# feature vectors.
# --------------------------------------------------------------------------

def collate(samples, device="cpu"):
    """samples: list of dicts, each carrying:
        'tokens'    : np.ndarray [Ti, F_TOKEN=236]
        'query_idx' : int
        'legal_mask': np.ndarray [NUM_ACTIONS=9] bool
        'opp_stats' : np.ndarray [F_STATS=12]
    Returns padded torch tensors keyed by forward()'s arg names."""
    B = len(samples)
    T = max(s["tokens"].shape[0] for s in samples)
    tok = np.zeros((B, T, F_TOKEN), dtype=np.float32)
    pad = np.ones((B, T), dtype=bool)
    qidx = np.zeros(B, dtype=np.int64)
    lmask = np.zeros((B, NUM_ACTIONS), dtype=bool)
    stats = np.zeros((B, F_STATS), dtype=np.float32)
    for i, s in enumerate(samples):
        ti = s["tokens"].shape[0]
        tok[i, :ti] = s["tokens"]
        pad[i, :ti] = False
        qidx[i] = s["query_idx"]
        lmask[i] = s["legal_mask"]
        stats[i] = s["opp_stats"]
    return {
        "tokens": torch.from_numpy(tok).to(device),
        "pad_mask": torch.from_numpy(pad).to(device),
        "query_idx": torch.from_numpy(qidx).to(device),
        "legal_mask": torch.from_numpy(lmask).to(device),
        "opp_stats": torch.from_numpy(stats).to(device),
    }
