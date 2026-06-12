"""Run-identity stamping + argmax guard for the live dry-run entry point.

Guards two properties of scripts/run_live_dryrun.py:
  1. The session header carries the full run identity (mode, seed, ckpt
     sha256, abstraction sha256, floors, git head/dirty, start time) and
     the hashes are computed from the actual file bytes.
  2. --mode argmax is refused without the explicit --unsafe-argmax flag
     (sample mode is the deployment invariant — DECISIONS.md iter_500
     entry); the default invocation stays sample.
All tests are arg-parse/unit level: no session is run, no model loaded.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.run_live_dryrun import (
    _sha256_of_file,
    build_session_header,
    parse_args,
)

REQUIRED_HEADER_KEYS = {
    "record_type", "mode", "seed", "ckpt_path", "ckpt_sha256",
    "abstraction_sha256", "floors", "git_head", "git_dirty", "started_utc",
}


def _minimal_argv(tmp_path, extra=()):
    ckpt = tmp_path / "ckpt.pt"
    abstr = tmp_path / "abstraction.pkl"
    ckpt.write_bytes(b"fake checkpoint bytes for hashing\n")
    abstr.write_bytes(b"fake abstraction bytes for hashing\n")
    argv = [
        "--replay-from-jsonl", str(tmp_path / "corpus.jsonl"),
        "--checkpoint", str(ckpt),
        "--abstraction", str(abstr),
        "--out", str(tmp_path / "out.jsonl"),
    ]
    argv.extend(extra)
    return argv, ckpt, abstr


def test_header_parses_with_required_keys_and_matching_hashes(tmp_path):
    argv, ckpt, abstr = _minimal_argv(tmp_path)
    args = parse_args(argv)
    header = build_session_header(args, short_stack_bb=6.0)

    # Round-trips through JSON (what main() writes as the first line).
    parsed = json.loads(json.dumps(header))
    assert parsed["record_type"] == "session_header"
    assert REQUIRED_HEADER_KEYS.issubset(parsed.keys())
    assert set(parsed["floors"].keys()) == {
        "aa_kk", "check_when_free", "short_stack_bb", "tail_floor_tau"}
    assert parsed["floors"]["aa_kk"] is True
    assert parsed["floors"]["check_when_free"] is True
    assert parsed["floors"]["short_stack_bb"] == 6.0
    # H1 tail floor defaults OFF (None) — header must say so explicitly.
    assert parsed["floors"]["tail_floor_tau"] is None
    assert isinstance(parsed["git_dirty"], bool)
    assert parsed["seed"] == 2026

    # Hashes match independently computed sha256 of the same files.
    assert parsed["ckpt_sha256"] == hashlib.sha256(
        ckpt.read_bytes()).hexdigest()
    assert parsed["abstraction_sha256"] == hashlib.sha256(
        abstr.read_bytes()).hexdigest()
    assert parsed["ckpt_sha256"] == _sha256_of_file(ckpt)


def test_argmax_without_unsafe_flag_exits_nonzero(tmp_path):
    argv, _, _ = _minimal_argv(tmp_path, extra=["--mode", "argmax"])
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code != 0


def test_argmax_with_unsafe_flag_accepted(tmp_path):
    argv, _, _ = _minimal_argv(
        tmp_path, extra=["--mode", "argmax", "--unsafe-argmax"])
    args = parse_args(argv)
    assert args.mode == "argmax"
    assert args.unsafe_argmax is True


def test_default_invocation_is_sample_in_header(tmp_path):
    argv, _, _ = _minimal_argv(tmp_path)
    args = parse_args(argv)
    assert args.mode == "sample"
    header = build_session_header(args, short_stack_bb=6.0)
    assert header["mode"] == "sample"
