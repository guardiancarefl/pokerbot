"""Build (or extend) a CheckpointRegistry from one or more training-run directories.

Scans `<run_dir>/checkpoints/` for files matching `ckpt_iter_NNNN.pt`, derives
a stable name per checkpoint (default `<name-prefix>-NNNN`), and registers
each into a `CheckpointRegistry` saved at `--output` (default
`runs/league/registry.json`).

The script is idempotent: running it twice with identical args is a no-op
(CheckpointRegistry.register short-circuits on identical content). Running
it with the same `--run-dir` but a different `--name-prefix` registers
the same checkpoints under different names, which is usually not what
you want — pass `--dry-run` to preview if uncertain.

Checkpoint iter is parsed from the filename. Files that don't match the
expected pattern are skipped with a warning. Checkpoints are registered
in ascending iter order so that LeaguePool's `recency` sampling strategy
sees the newest checkpoints as most recent.

Usage examples:

    # Register every checkpoint from tonight's overnight as dcfr-overnight-NNNN.
    python -m scripts.build_league_registry \\
        --run-dir runs/six_max_20260524_014344_phase4f_dcfr_linear_overnight \\
        --name-prefix dcfr-overnight \\
        --tags dcfr overnight \\
        --output runs/league/registry.json

    # Extend the existing registry with shakedown checkpoints.
    python -m scripts.build_league_registry \\
        --run-dir runs/six_max_20260524_005853_phase4f_dcfr_linear_shakedown \\
        --name-prefix dcfr-shake \\
        --tags dcfr shakedown \\
        --output runs/league/registry.json

    # Dry-run: print what WOULD be registered without writing.
    python -m scripts.build_league_registry \\
        --run-dir runs/... --name-prefix dcfr-overnight --tags dcfr --dry-run

    # Inspect: just show the current registry.
    python -m scripts.build_league_registry --inspect --output runs/league/registry.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

from src.nlhe.checkpoint_registry import (
    CheckpointRegistry,
    DEFAULT_REGISTRY_PATH,
)


CKPT_PATTERN = re.compile(r"^ckpt_iter_(\d+)\.pt$")


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """Return [(iter, path), ...] sorted ascending by iter.

    Looks for `ckpt_iter_NNNN.pt` files under `<run_dir>/checkpoints/`.
    Files matching the pattern with `final` or other non-numeric suffixes
    are skipped (they don't fit the registry's iter-indexed model).
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {ckpt_dir}"
        )

    found: list[tuple[int, Path]] = []
    skipped: list[str] = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file():
            continue
        m = CKPT_PATTERN.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        it = int(m.group(1))
        found.append((it, p))

    if skipped:
        print(
            f"  note: skipped {len(skipped)} non-iter checkpoint(s) in "
            f"{ckpt_dir}: {skipped[:5]}"
            f"{'...' if len(skipped) > 5 else ''}",
            file=sys.stderr,
        )

    return sorted(found, key=lambda t: t[0])


def read_run_config_tag(run_dir: Path) -> str | None:
    """Best-effort: pull the config tag from <run_dir>/checkpoints/config.json.

    Session 2 deferred-cleanup noted that train_leduc writes config into
    checkpoints/. The 6-max trainer follows the same pattern. Returns
    None if no usable tag is found.
    """
    candidates = [
        run_dir / "checkpoints" / "config.json",
        run_dir / "config.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                d = json.loads(c.read_text())
                tag = d.get("tag")
                if isinstance(tag, str) and tag:
                    return tag
            except (json.JSONDecodeError, OSError):
                continue
    return None


def register_run(
    registry: CheckpointRegistry,
    run_dir: Path,
    name_prefix: str,
    tags: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Register every checkpoint in run_dir under <prefix>-NNNN.

    Returns (added_count, idempotent_count). Conflicts raise.
    """
    config_tag = read_run_config_tag(run_dir)
    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        print(f"  warning: no checkpoints found in {run_dir}", file=sys.stderr)
        return 0, 0

    added = 0
    idempotent = 0
    for it, path in checkpoints:
        name = f"{name_prefix}-{it}"
        metadata = {
            "iter": it,
            "run_dir": str(run_dir),
        }
        if config_tag:
            metadata["config_tag"] = config_tag

        if dry_run:
            existing = registry._entries.get(name)
            status = "skip" if existing else "add "
            print(f"  [dry-run] {status} {name:<28}  iter={it:>5}  {path}")
            continue

        existing = registry._entries.get(name)
        try:
            entry = registry.register(
                name=name,
                path=str(path),
                metadata=metadata,
                tags=tags,
            )
        except ValueError as e:
            raise ValueError(f"conflict on {name!r}: {e}") from e
        if existing and existing == entry:
            idempotent += 1
        else:
            added += 1
            print(f"  add  {name:<28}  iter={it:>5}  {path}")

    return added, idempotent


def inspect_registry(registry: CheckpointRegistry) -> None:
    """Print the registry contents in a human-readable summary."""
    if len(registry) == 0:
        print("(registry is empty)")
        return
    print(f"{'name':<32}  {'iter':>6}  tags                         path")
    print("-" * 110)
    for name in registry.names():
        e = registry.get(name)
        it = e.metadata.get("iter", "?")
        tags = ",".join(e.tags)
        print(f"{name:<32}  {str(it):>6}  {tags:<28}  {e.path}")
    print(f"\ntotal: {len(registry)} checkpoints")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Training-run directory containing checkpoints/. "
             "Repeat to register multiple runs in one invocation.",
    )
    p.add_argument(
        "--name-prefix",
        default=None,
        help="Per-run name prefix (required if --run-dir is given). "
             "Checkpoints register as <prefix>-<iter>. If using multiple "
             "--run-dir, also pass multiple --name-prefix in matching "
             "order. (Single --name-prefix with multiple --run-dir uses "
             "the same prefix for all, which usually causes a conflict.)",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        default=[],
        help="Tag(s) to apply to every registered checkpoint in this run.",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_REGISTRY_PATH,
        help=f"Path to registry JSON (default: {DEFAULT_REGISTRY_PATH}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be registered, do not write.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print the contents of the registry at --output and exit. "
             "No registration performed.",
    )
    args = p.parse_args()

    output_path = Path(args.output)
    registry = CheckpointRegistry.load(str(output_path))

    if args.inspect:
        if not output_path.exists():
            print(f"(no registry file at {output_path}; empty registry)")
        inspect_registry(registry)
        return 0

    if not args.run_dir:
        p.error("at least one --run-dir is required (or use --inspect)")

    if not args.name_prefix:
        p.error("--name-prefix is required when --run-dir is given")

    print(f"Output: {output_path}")
    print(f"Starting registry size: {len(registry)}")
    print()

    total_added = 0
    total_idempotent = 0
    for rd in args.run_dir:
        run_dir = Path(rd)
        if not run_dir.is_dir():
            print(f"  ERROR: --run-dir not a directory: {run_dir}", file=sys.stderr)
            return 2
        print(f"Run: {run_dir}")
        added, idem = register_run(
            registry=registry,
            run_dir=run_dir,
            name_prefix=args.name_prefix,
            tags=args.tags,
            dry_run=args.dry_run,
        )
        total_added += added
        total_idempotent += idem
        print(f"  -> added {added}, idempotent {idem}")
        print()

    if args.dry_run:
        print("(dry-run; nothing written)")
        return 0

    saved_path = registry.save(str(output_path))
    print(f"Wrote {saved_path}")
    print(f"Final registry size: {len(registry)} "
          f"(+{total_added} new, {total_idempotent} unchanged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
