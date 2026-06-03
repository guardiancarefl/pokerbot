"""ReBeL Phase-2 sample persistence — portable, shardable, mergeable shards.

A training sample (docs/REBEL_PBS_6MAX.md §5) is `(PBS encoding, target)` where
the target is the depth-limited search's own root value (the bootstrapped
regression target). This module persists those samples to disk so a multi-core /
multi-box generation run can write independent shards that merge losslessly.

What we persist per sample (schema v1):
  PBS encoding (lossless, reproducible — the validated InfosetEncoder6Max root
  feature, whose layout is [bucket one-hot (max_bucket_dim)] ++ [public block]):
    - feat        float32[FEAT_DIM]   the full 236-dim root encoding
    - bucket      int16               acting-seat bucket id (argmax of the one-hot;
                                       -1 if none), i.e. hero's degenerate range
  (The pure public block is `feat[BUCKET_DIM:]`; the per-seat 6×k *soft* belief
   block is the Phase-3 net-input assembly step, pre-registered in §3–4 of the
   PBS doc. Everything needed to build it is carried here: many samples over the
   same public node ARE the empirical belief.)
  Target (search output, docs/REBEL_PBS_6MAX.md §5):
    - root_value      float32         hero root EV = Σ_a root_policy[a]·q[a]
    - root_policy     float32[N_ACT]  refined root strategy
    - root_q_values   float32[N_ACT]  hero per-action Q at the root
    - legal_mask      int8[N_ACT]
  Provenance (every gate carries it; catches silent drift across boxes):
    - hero_seat, street, depth, n_iterations, degraded (per-sample)
    - host, worker, schema, blueprint_sha, abstraction_sha, k_postflop,
      feat_dim, n_actions, git_sha, created_unix (per-shard meta)

Shard format: a directory of `.npz` files. NPZ holds only numpy arrays (no
pickled custom classes) so a shard written on Contabo loads byte-identically
here — portability by construction. Each shard is named

    shard-<host>-w<worker>-<ts>-<seq>.npz

so shards from different boxes/workers never collide and a plain glob over a
merged directory concatenates them. Writes are atomic (tmp + os.replace) and
flushed every `flush_every` samples, so a disconnect loses at most one partial
buffer, never a completed shard.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2
DEFAULT_ROOT = "/workspace/rebel_samples"


def file_sha1(path: str, _buf: int = 1 << 20) -> str:
    """Short SHA1 of a file's bytes (artifact fingerprint for provenance)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class RebelSample:
    """One (PBS, target) training sample. Arrays are plain numpy (portable).

    Schema v2: per-seat 6-vector value target (`value6`) + belief block
    (`seat_buckets`, the 6-seat joint config → 6×k one-hot at train time) +
    tournament context (`blind_level`, `alive_count`, `dealer_seat`) so the net
    learns ICM/stack-depth/player-count structure across the whole reachable
    Double-Up SNG state space, not one spot.
    """
    feat: np.ndarray            # float32[feat_dim] — root PBS encoding (public + acting bucket)
    bucket: int                 # acting-seat (hero) bucket id (-1 if none)
    # belief: float16[6,k] — per-seat range. hero row = one-hot at its KNOWN bucket;
    # each active opponent row = the reach-POSTERIOR over buckets inferred from its
    # blueprint actions (a true BELIEF — what the bot knows from public betting, NOT
    # the opponent's actual cards); folded/busted rows = all-zero. This is the value-
    # net's opponent-uncertainty input; it transfers to deployment (no perfect info).
    belief: np.ndarray
    hero_seat: int
    street: int
    depth: int
    n_iterations: int
    degraded: bool
    blind_level: int            # tournament blind level (1..N)
    alive_count: int            # players still in the tournament (4..6; 4 = bubble)
    dealer_seat: int            # button seat
    value6: np.ndarray          # float32[6] — PER-SEAT root value target (THE fix)
    root_value: float           # = value6[hero_seat] (scalar, kept for sanity/back-compat)
    root_policy: np.ndarray     # float32[n_actions]
    root_q_values: np.ndarray   # float32[n_actions]
    legal_mask: np.ndarray      # int8[n_actions]


# Column names <-> per-sample fields. Kept explicit so the on-disk layout is a
# stable contract the merger/loader and any trainer can rely on.
_VECTOR_COLS = ("feat", "value6", "belief", "root_policy",
                "root_q_values", "legal_mask")


class ShardWriter:
    """Buffers RebelSamples and flushes atomic .npz shards to `root_dir`.

    One writer per worker process. The host+worker+timestamp prefix guarantees
    cross-box / cross-worker uniqueness, so every box can point at its own
    /workspace and the union of all shard dirs merges with a glob.
    """

    def __init__(
        self,
        root_dir: str = DEFAULT_ROOT,
        worker: int = 0,
        flush_every: int = 256,
        meta: Optional[dict] = None,
        host: Optional[str] = None,
    ) -> None:
        self.root_dir = root_dir
        self.worker = int(worker)
        self.flush_every = int(flush_every)
        self.host = (host or socket.gethostname()).replace("/", "_").replace(" ", "_")
        # Whole-second start stamp makes shard names sortable and stable per run.
        self._ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self._seq = 0
        self._buf: list[RebelSample] = []
        self._written = 0
        self.meta = dict(meta or {})
        self.meta.update({
            "schema": SCHEMA_VERSION,
            "host": self.host,
            "worker": self.worker,
        })
        os.makedirs(self.root_dir, exist_ok=True)

    def add(self, s: RebelSample) -> None:
        self._buf.append(s)
        if len(self._buf) >= self.flush_every:
            self.flush()

    @property
    def written(self) -> int:
        return self._written

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def flush(self) -> Optional[str]:
        """Write the current buffer as one atomic shard. Returns the path (or None)."""
        if not self._buf:
            return None
        n = len(self._buf)
        cols: dict = {}
        # Stacked vector columns.
        for c in _VECTOR_COLS:
            cols[c] = np.stack([np.asarray(getattr(s, c)) for s in self._buf])
        # Scalar columns (1-D arrays, one entry per sample).
        cols["bucket"] = np.asarray([s.bucket for s in self._buf], dtype=np.int16)
        cols["hero_seat"] = np.asarray([s.hero_seat for s in self._buf], dtype=np.int8)
        cols["street"] = np.asarray([s.street for s in self._buf], dtype=np.int8)
        cols["depth"] = np.asarray([s.depth for s in self._buf], dtype=np.int8)
        cols["n_iterations"] = np.asarray([s.n_iterations for s in self._buf], dtype=np.int32)
        cols["degraded"] = np.asarray([s.degraded for s in self._buf], dtype=np.bool_)
        cols["blind_level"] = np.asarray([s.blind_level for s in self._buf], dtype=np.int8)
        cols["alive_count"] = np.asarray([s.alive_count for s in self._buf], dtype=np.int8)
        cols["dealer_seat"] = np.asarray([s.dealer_seat for s in self._buf], dtype=np.int8)
        cols["root_value"] = np.asarray([s.root_value for s in self._buf], dtype=np.float32)
        cols["_meta_json"] = np.asarray(json.dumps(self.meta))

        name = f"shard-{self.host}-w{self.worker:02d}-{self._ts}-{self._seq:05d}.npz"
        path = os.path.join(self.root_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **cols)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)   # atomic publish
        self._seq += 1
        self._written += n
        self._buf.clear()
        return path

    def close(self) -> None:
        self.flush()


def load_shard(path: str) -> dict:
    """Load one shard .npz into a dict of columns (+ parsed `meta`)."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "_meta_json"}
        if "_meta_json" in z.files:
            out["meta"] = json.loads(str(z["_meta_json"]))
    return out


def iter_shards(root_dirs: Sequence[str]):
    """Yield every shard path under the given root dirs (sorted, dedup)."""
    seen = set()
    for root in root_dirs:
        for p in sorted(_glob.glob(os.path.join(root, "shard-*.npz"))):
            if p not in seen:
                seen.add(p)
                yield p


def merge_shards(root_dirs: Sequence[str]) -> dict:
    """Concatenate all shards under `root_dirs` into one in-memory column dict.

    Validates schema/feat-dim/n-actions consistency across boxes and raises on
    mismatch (a silent dim drift across boxes is exactly the bug we guard).
    """
    paths = list(iter_shards(root_dirs))
    if not paths:
        raise FileNotFoundError(f"no shards under {list(root_dirs)}")
    cols: dict = {}
    metas: list = []
    feat_dim = None
    n_act = None
    total = 0
    for p in paths:
        d = load_shard(p)
        if "feat" in d and d["feat"].ndim == 2:
            fd = d["feat"].shape[1]
            feat_dim = fd if feat_dim is None else feat_dim
            if fd != feat_dim:
                raise ValueError(f"feat_dim drift: {p} has {fd}, expected {feat_dim}")
        if "root_policy" in d and d["root_policy"].ndim == 2:
            na = d["root_policy"].shape[1]
            n_act = na if n_act is None else n_act
            if na != n_act:
                raise ValueError(f"n_actions drift: {p} has {na}, expected {n_act}")
        metas.append(d.pop("meta", {}))
        for k, v in d.items():
            cols.setdefault(k, []).append(v)
        total += len(d.get("root_value", []))
    merged = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    merged["_n"] = total
    merged["_n_shards"] = len(paths)
    merged["_metas"] = metas
    merged["_feat_dim"] = feat_dim
    merged["_n_actions"] = n_act
    return merged


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Merge/inspect ReBeL sample shards.")
    ap.add_argument("roots", nargs="+", help="one or more shard root dirs")
    ap.add_argument("--out", default=None, help="write merged .npz here")
    a = ap.parse_args()
    m = merge_shards(a.roots)
    hosts = sorted({mm.get("host", "?") for mm in m["_metas"]})
    print(f"merged {m['_n']} samples from {m['_n_shards']} shards "
          f"across hosts {hosts}; feat_dim={m['_feat_dim']} n_actions={m['_n_actions']}")
    if a.out:
        save = {k: v for k, v in m.items() if not k.startswith("_")}
        np.savez_compressed(a.out, **save)
        print(f"wrote merged dataset -> {a.out}")


if __name__ == "__main__":
    _cli()
