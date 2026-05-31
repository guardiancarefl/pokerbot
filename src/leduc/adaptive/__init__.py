"""Leduc adaptive-policy net (Step-3 build).

Sequence model that maps within-match history -> hero policy (regularized
toward the CFR+ anchor) + an auxiliary opponent-action prediction. See
docs/scratch/leduc_adaptive_policy_arch.md for the approved architecture
(§2-§7) and the Step-3 training spec (§12).
"""
