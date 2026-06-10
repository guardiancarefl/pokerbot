"""Checkpoint ante-convention resolution for live serving (C3, Step 3).

Two game-string conventions exist in this project's history:

  "real"         — per-seat native antes (canonical
                   `to_inner_game_string_for_state`, patched pyspiel).
                   Empty/busted seats post nothing. The live bridge's
                   `invariant.openspiel_to_scraper_view` is a 1:1 identity
                   under this convention (no ghost-ante correction).
  "inflated_bb"  — the legacy bb + N*ante hack. Models trained under it
                   (e.g. the pre-real-ante rebel value net and candC k200)
                   require the retired Option B ghost-ante bug-match view
                   (removed from the bridge at ae8307e, 2026-06-05;
                   DECISIONS.md "Integration ghost-ante bug-match
                   workaround").

C3 checkpoints stamp `ante_convention` in their saved config_dict
(TrainConfig6Max.ante_convention). Checkpoints that predate the flag are
resolved ONLY through the explicit sha256 whitelist below — a missing
flag is never assumed to mean "real", and convention dispatch is always
per-checkpoint, never a global flip.

Live entry points call `require_live_servable(ckpt_path)` after loading:
anything that does not resolve to "real" is refused loudly, because the
clean bridge view would silently feed an inflated-convention model
out-of-distribution states (the exact failure Option B existed to
prevent).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ANTE_REAL = "real"
ANTE_INFLATED = "inflated_bb"

# Pre-flag checkpoints with operator-verified conventions. The deployed
# production model (DECISIONS.md "Deployment reversal record") trained on
# the canonical real-ante emitter, 2026-06-05 -> 07.
KNOWN_CKPT_CONVENTIONS = {
    # runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt
    "b79e82dd0ce9e78e4eb666b7379df953dadbf2a6e026c6bd4b6eec695e9b1b11":
        ANTE_REAL,
}


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_ante_convention(ckpt_path: str | Path,
                             config_dict: dict | None = None) -> str | None:
    """Resolve a checkpoint's ante convention.

    Resolution order:
      1. `ante_convention` stamped in the checkpoint's config_dict
         (every C3+ checkpoint).
      2. The KNOWN_CKPT_CONVENTIONS sha256 whitelist (pre-flag
         checkpoints with operator-verified provenance).
      3. None — unknown. Callers on the live path must treat this as
         not servable.

    Args:
        ckpt_path: path to the .pt checkpoint.
        config_dict: the checkpoint's saved RAW config dict, if the caller
            already has it. When None, it is read from the checkpoint file.
            Never pass a config that went through _load_solver's
            saved.get(..., "real") defaulting — that default is loader
            metadata, not provenance.
    """
    if config_dict is None:
        import torch
        ckpt = torch.load(str(ckpt_path), weights_only=False,
                          map_location="cpu")
        config_dict = ckpt.get("config_dict", {})
    if config_dict and "ante_convention" in config_dict:
        return str(config_dict["ante_convention"])
    sha = sha256_of_file(ckpt_path)
    return KNOWN_CKPT_CONVENTIONS.get(sha)


def require_live_servable(ckpt_path: str | Path,
                           config_dict: dict | None = None) -> str:
    """Gate a checkpoint for live/bridge serving. Returns the resolved
    convention ("real") or raises with the reason it cannot be served."""
    conv = resolve_ante_convention(ckpt_path, config_dict)
    if conv == ANTE_REAL:
        return conv
    if conv == ANTE_INFLATED:
        raise RuntimeError(
            f"checkpoint {ckpt_path} is inflated_bb-convention; the Option B "
            "ghost-ante bug-match view was retired from the bridge "
            "(ae8307e) and this model cannot be served on the clean view. "
            "See DECISIONS.md ghost-ante entries."
        )
    raise RuntimeError(
        f"checkpoint {ckpt_path} has no ante_convention stamp and its "
        "sha256 is not in conventions.KNOWN_CKPT_CONVENTIONS — refusing to "
        "serve live. Verify its training provenance and whitelist it "
        "explicitly."
    )
