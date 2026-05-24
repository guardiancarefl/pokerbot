"""Add Shanky archetype entries to the league registry.

Reads the existing configs/league/registry.json, adds entries for each
.txt profile in --shanky-dir (filtered by --only if provided), and writes
back the updated registry. Skips entries that already exist.

Run AFTER the tournament eval (eval_shanky_vs_dcfr.py) — the eval tells
you which Shanky profiles are competitive enough to be worth including.
Pass those profiles' names via --only to add only the winners.

Example (add 5 selected archetypes):
    python -m scripts.update_registry_shanky \\
        --shanky-dir /workspace/pokerbot/data/shanky_profiles \\
        --registry configs/league/registry.json \\
        --only gushansenmtt modernmikemtt killphilmtt kamakazi shapeshifter

Example (add all 36 profiles — usually NOT what you want):
    python -m scripts.update_registry_shanky \\
        --shanky-dir /workspace/pokerbot/data/shanky_profiles \\
        --registry configs/league/registry.json
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("update_registry_shanky")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shanky-dir",
        required=True,
        help="directory containing .txt Shanky profile files",
    )
    ap.add_argument(
        "--registry",
        default="configs/league/registry.json",
        help="path to league registry JSON",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "if given, only register these profiles (by normalized stem). "
            "default: register all .txt files in --shanky-dir"
        ),
    )
    ap.add_argument(
        "--tags",
        nargs="*",
        default=["shanky", "archetype"],
        help="tags to attach to each Shanky entry (default: shanky, archetype)",
    )
    ap.add_argument(
        "--big-blind-chips",
        type=int,
        default=100,
        help="chips per BB metadata (default 100)",
    )
    ap.add_argument(
        "--name-prefix",
        default="shanky-",
        help="prefix for registry entry names (default 'shanky-')",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned changes without writing",
    )
    args = ap.parse_args()

    from src.nlhe.checkpoint_registry import CheckpointRegistry
    # Validate Shanky parsing works on these files before registering
    from src.nlhe.scripted_bots.parser import parse_profile

    if not os.path.isdir(args.shanky_dir):
        log.error(f"shanky_dir not found: {args.shanky_dir}")
        sys.exit(1)

    # Load existing registry (creates empty if missing)
    log.info(f"loading registry from {args.registry}")
    if os.path.exists(args.registry):
        reg = CheckpointRegistry.load(args.registry)
    else:
        log.info(f"  (no existing registry; will create new)")
        reg = CheckpointRegistry()

    existing_names = {e.name for e in reg}
    log.info(f"  existing entries: {len(existing_names)}")

    # Discover candidate Shanky files
    added = []
    skipped_existing = []
    skipped_filter = []
    failed_parse = []

    for fname in sorted(os.listdir(args.shanky_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0].lower()
        # Normalize suffixes added by some download sources
        normalized = stem.replace("__1_", "").replace("_v_", "_v").strip("_")
        if args.only is not None and normalized not in args.only and stem not in args.only:
            skipped_filter.append(normalized)
            continue

        entry_name = f"{args.name_prefix}{normalized}"
        if entry_name in existing_names:
            skipped_existing.append(entry_name)
            continue

        path = os.path.join(args.shanky_dir, fname)

        # Sanity check: profile parses without exception
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            prof = parse_profile(source, source_name=normalized)
            n_rules = sum(len(s.rules) for s in prof.sections)
        except Exception as e:
            log.warning(f"  parse failed for {fname}: {e}")
            failed_parse.append(fname)
            continue

        if not args.dry_run:
            reg.register(
                entry_name,
                path,
                metadata={
                    "policy_type": "shanky",
                    "source_file": fname,
                    "n_rules": n_rules,
                    "big_blind_chips": args.big_blind_chips,
                },
                tags=list(args.tags),
            )
        added.append((entry_name, fname, n_rules))

    log.info("")
    log.info("=== Summary ===")
    log.info(f"  added           {len(added)}")
    log.info(f"  skipped (already in registry)  {len(skipped_existing)}")
    log.info(f"  skipped (filtered out by --only)  {len(skipped_filter)}")
    log.info(f"  failed to parse {len(failed_parse)}")

    if added:
        log.info("")
        log.info("Added entries:")
        for name, fname, n_rules in added:
            log.info(f"  {name:<40} {fname:<35} ({n_rules} rules)")

    if skipped_existing:
        log.info("")
        log.info(f"Skipped (already registered): {len(skipped_existing)}")
        for n in skipped_existing[:10]:
            log.info(f"  {n}")
        if len(skipped_existing) > 10:
            log.info(f"  ...and {len(skipped_existing) - 10} more")

    if failed_parse:
        log.info("")
        log.warning("Failed to parse:")
        for fname in failed_parse:
            log.warning(f"  {fname}")

    if args.dry_run:
        log.info("")
        log.info("DRY RUN — no changes written")
    elif added:
        reg.save(args.registry)
        log.info("")
        log.info(f"Saved updated registry to {args.registry} ({len(reg)} total entries)")
    else:
        log.info("")
        log.info("No changes to write.")


if __name__ == "__main__":
    main()
