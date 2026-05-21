# Project Status

**Last updated:** 2026-05-21
**Current phase:** Pre-Phase-1 setup

## Done
- Architecture designed (four-layer stack with opponent anonymity as core principle)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Runtime environment decided (WSL2 + Ubuntu 22.04 on Windows host)
- Hardware confirmed (RTX 3060 Laptop GPU, 6GB VRAM — not the 12GB desktop variant; constraint noted for Phase 4+)
- Scope and non-goals defined
- Opponent anonymity principle established (no persistent identity across matches, no real-world hand history data, no pre-collected opponent profiles)
- Within-match adaptation approach decided (Position 2: light-to-medium online reads anchored to blueprint)
- League play role clarified (provides strength diversity; the 42 bought-bot profiles stay frozen to preserve benchmark stability)
- Claude Code installed on Windows host, PATH configured
- Project directory created at C:\Users\Ngior\pokerbot\ with docs/ subfolder
- All four foundational docs created in docs/ (PROJECT_OVERVIEW.md, ARCHITECTURE.md, DECISIONS.md, STATUS.md)

## In progress
- Pre-WSL2 setup (next session)

## Next up
1. Install WSL2 + Ubuntu 22.04 on Windows host
2. Verify CUDA passthrough from WSL2 to the RTX 3060 Laptop GPU
3. Set up Python 3.10 environment with venv inside WSL2
4. Install PyTorch (CUDA build) and OpenSpiel inside the venv
5. Confirm PyTorch sees the GPU and OpenSpiel loads Leduc poker
6. Move project directory to the WSL2 Linux filesystem (better I/O performance than the Windows mount)
7. Install Claude Code inside WSL2 for ongoing project work
8. Phase 1 scaffold: Leduc Deep CFR training script, project directory structure
9. Phase 1 validation: confirm convergence to Nash on Leduc

## Known issues / open questions
- 42 bought-bot profile format still unidentified (text/XML/JSON/binary) — defer until needed in Phase 3
- Workflow note: Claude Code has shown a tendency to apply its own judgment to instructions instead of following them verbatim. The rule for this project is: when exact file contents are provided between BEGIN/END markers, write them exactly; concerns must be raised as questions before writing, not as silent edits announced after.

## Decisions deferred
- Specific card abstraction granularity (decide during Phase 2 based on observed convergence rates)
- Exact league play schedule (decide after Phase 4 blueprint exists)
- Whether to burst cloud GPU for final blueprint training (decide late Phase 4)
- Whether to integrate VPS as parallel self-play worker (decide if 3060 throughput becomes bottleneck)
- Exact mix percentages for training opponent pool composition (self-play vs. archetypes vs. bought-bots vs. league archives) — tune empirically during Phase 4-5

## Session log
- **2026-05-21:** Project bootstrapped. Foundational docs (PROJECT_OVERVIEW, ARCHITECTURE, DECISIONS, STATUS) created on local Windows machine. Major design decisions made: opponent anonymity as core principle, within-match adaptation at Position 2, league play for strength diversity, 42 profiles stay frozen. Claude Code installed and operational. Next session: WSL2 setup.
