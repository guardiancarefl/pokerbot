"""Adaptive-policy network for Leduc (arch doc §4-§7 + §12.3 gate).

Forward pass (§4):
    x       = token_proj(tokens) + pos_embed[:T]
    h       = encoder(x, src_key_padding_mask=pad_mask)
    repr    = h[:, query_idx]
    s       = stats_proj(opp_stats)
    combined= concat([repr, s])              # [B, 96]
    policy_raw = masked_softmax(policy_head(combined))   # hero policy (§5)
    opp_logits = opp_head(combined)          # aux opp-action model (§5)

The confidence-gated blend (§12.2) and the read-driven gate (§12.3) live in
free functions here so callers can compose them per phase:
  - Phase-1 distillation uses policy_raw directly (g implicitly = 1).
  - Phase-2 / safety eval uses blend(g, anchor, raw) with g = gate(read).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.leduc.adaptive.tokens import F_RAW, F_STATS, NUM_ACTIONS

NEG_INF = -1e9


class AdaptivePolicyNet(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 128, dropout: float = 0.0, max_seq: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.token_proj = nn.Linear(F_RAW, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.stats_proj = nn.Linear(F_STATS, 32)
        self.policy_head = nn.Linear(d_model + 32, NUM_ACTIONS)
        self.opp_head = nn.Linear(d_model + 32, NUM_ACTIONS)

    def forward(self, tokens, pad_mask, query_idx, legal_mask, opp_stats):
        """tokens [B,T,17], pad_mask [B,T] (True=pad), query_idx [B],
        legal_mask [B,3] (True=legal), opp_stats [B,6].
        Returns (policy_raw [B,3], opp_logits [B,3], combined [B,96])."""
        B, T, _ = tokens.shape
        x = self.token_proj(tokens) + self.pos_embed[:T].unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        repr_ = h[torch.arange(B), query_idx]            # [B, d_model]
        s = self.stats_proj(opp_stats)                   # [B, 32]
        combined = torch.cat([repr_, s], dim=-1)         # [B, 96]
        logits = self.policy_head(combined)
        logits = logits.masked_fill(~legal_mask, NEG_INF)
        policy_raw = F.softmax(logits, dim=-1)
        opp_logits = self.opp_head(combined)
        return policy_raw, opp_logits, combined


# --------------------------------------------------------------------------
# Batching: pack variable-length token sequences from tokenize_decision().
# --------------------------------------------------------------------------

def collate(samples, device="cpu"):
    """samples: list of dicts from tokens.tokenize_decision (each has
    'tokens' [Ti,17], 'query_idx', 'legal_mask' [3], 'opp_stats' [6]).
    Returns a dict of padded tensors for AdaptivePolicyNet.forward."""
    B = len(samples)
    T = max(s["tokens"].shape[0] for s in samples)
    tok = np.zeros((B, T, F_RAW), dtype=np.float32)
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


def masked_softmax_np(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    z = np.where(legal_mask, logits, NEG_INF)
    z = z - z.max()
    e = np.exp(z) * legal_mask
    return e / e.sum()


# --------------------------------------------------------------------------
# §12.3 read signal + gate, and §12.2 confidence-gated blend.
# --------------------------------------------------------------------------

def read_terms(q_probs: np.ndarray, pi_eq: np.ndarray, legal_mask: np.ndarray):
    """Per opp-decision read contribution.
        d_t = KL(q_t || pi_eq)          (how non-equilibrium the read is)
        w_t = 1 - H(q_t)/log|legal|     (sharpness/reliability in [0,1])
    q_probs, pi_eq, legal_mask are length-3 over {fold,call,raise}; only legal
    entries carry mass. Returns (d_t, w_t)."""
    eps = 1e-12
    legal = legal_mask.astype(bool)
    q = np.clip(q_probs[legal], eps, 1.0)
    p = np.clip(pi_eq[legal], eps, 1.0)
    q = q / q.sum()
    p = p / p.sum()
    d_t = float(np.sum(q * (np.log(q) - np.log(p))))
    n_legal = int(legal.sum())
    if n_legal <= 1:
        w_t = 1.0
    else:
        H = -float(np.sum(q * np.log(q)))
        w_t = 1.0 - H / np.log(n_legal)
    return d_t, max(0.0, min(1.0, w_t))


def read_terms_argmax(q_probs: np.ndarray, pi_eq: np.ndarray,
                      legal_mask: np.ndarray, tie_tol: float = 1e-9):
    """Per opp-decision read contribution, argmax-disagreement variant (D-pivot).
        d_t = 1[ argmax(q_t) not in argmax_set(pi_eq) ]  (binary disagreement)
        w_t = max(q_t) - second_max(q_t)                 (sharpness margin)
    `argmax_set(pi_eq)` is the set of actions whose pi_eq probability is within
    `tie_tol` of the maximum (handles GTO mixed strategies). With one legal
    action (cap-forced spots), d_t = 0 by construction since the head and GTO
    are both forced to the same single action. Returns (d_t, w_t)."""
    legal = legal_mask.astype(bool)
    q = q_probs.astype(np.float64).copy()
    p = pi_eq.astype(np.float64).copy()
    q[~legal] = 0.0
    p[~legal] = 0.0
    sq = q.sum()
    sp = p.sum()
    if sq > 0:
        q = q / sq
    if sp > 0:
        p = p / sp
    n_legal = int(legal.sum())
    if n_legal <= 1:
        return 0.0, 0.0
    q_legal = q[legal]
    p_legal = p[legal]
    legal_indices = np.flatnonzero(legal)
    q_argmax_local = int(np.argmax(q_legal))
    q_argmax = int(legal_indices[q_argmax_local])
    p_max = float(p_legal.max())
    p_argmax_set = {int(legal_indices[i]) for i, v in enumerate(p_legal)
                    if (p_max - v) <= tie_tol}
    d_t = 0.0 if q_argmax in p_argmax_set else 1.0
    q_sorted = np.sort(q_legal)[::-1]
    margin = float(q_sorted[0] - q_sorted[1])
    w_t = max(0.0, min(1.0, margin))
    return d_t, w_t


def running_read(d_list, w_list, w0: float = 2.0) -> float:
    """Evidence-weighted, prior-shrunk running read D-bar (§12.3).
        D_bar = sum(w_t d_t) / (w0 + sum w_t)
    With no opp decisions, D_bar = 0 exactly (=> gate closed)."""
    if not w_list:
        return 0.0
    num = float(np.sum(np.array(w_list) * np.array(d_list)))
    den = w0 + float(np.sum(w_list))
    return num / den


def gate(d_bar: float, kappa: float = 1.0) -> float:
    """g = tanh(kappa * D_bar). gate(0) = 0 exactly; bounded in [0,1)."""
    return float(np.tanh(kappa * d_bar))


def blend(g: float, anchor_probs: np.ndarray, raw_probs: np.ndarray) -> np.ndarray:
    """pi_final = (1-g) * anchor + g * raw  (§12.2). At g=0, returns the anchor
    bit-exactly (the zero-evidence guarantee)."""
    return (1.0 - g) * anchor_probs + g * raw_probs
