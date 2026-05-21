# Project Status

**Last updated:** 2026-05-21 (Session 2)
**Current phase:** Phase 1 — Infrastructure validation (Leduc poker)

## Done
- Architecture designed (four-layer stack with opponent anonymity as core principle)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Scope and non-goals defined
- Opponent anonymity principle established (no persistent identity across matches, no real-world hand history data, no pre-collected opponent profiles)
- Within-match adaptation approach decided (Position 2: light-to-medium online reads anchored to blueprint)
- League play role clarified (provides strength diversity; the 42 bought-bot profiles stay frozen to preserve benchmark stability)
- Foundational docs created (PROJECT_OVERVIEW, ARCHITECTURE, DECISIONS, STATUS, SESSION_LOG)
- Runtime environment migrated from WSL2-on-Windows to Contabo VPS (Ubuntu 24.04, 12 vCPU AMD EPYC, 48GB RAM, ~300GB free) after Windows component store corruption blocked WSL2 install
- GitHub repo created at github.com/guardiancarefl/pokerbot (private), SSH key registered
- Python 3.10 installed alongside system Python 3.12 via deadsnakes PPA
- Project directory established at ~/pokerbot on the Contabo VPS, initial commit pushed

## In progress
- Python venv + PyTorch (CPU) + OpenSpiel install
- Phase 1 scaffold: Leduc Deep CFR training script

## Next up
1. Create Python 3.10 venv at ~/pokerbot/.venv
2. Install PyTorch (CPU build) and OpenSpiel inside the venv
3. Smoke tests: torch tensor op, pyspiel.load_game("leduc_poker")
4. Create project directory structure (src/, tests/, scripts/, configs/, runs/)
5. Write Leduc Deep CFR training script using open_spiel.python.pytorch.deep_cfr
6. Start training run, verify it produces decreasing exploitability
7. Evaluate Phase 1 result in Session 3 (after the run has had time to converge)
8. Phase 2: rent GPU instance (Vast.ai / RunPod / Vultr GPU) when needed, migrate via git clone

## Known issues / open questions
- 42 bought-bot profile format still unidentified (text/XML/JSON/binary) — defer until needed in Phase 3
- Workflow note: Claude Code has shown a tendency to apply its own judgment to instructions instead of following them verbatim. The rule for this project is: when exact file contents are provided between BEGIN/END markers, write them exactly; concerns must be raised as questions before writing, not as silent edits announced after.
- The original Windows host (RTX 3060 Laptop, 6GB VRAM) still exists and could be used later if the WSL2 component store issue is repaired. Not a priority — Contabo + future cloud GPU is cleaner.
- Contabo vCPUs are oversubscribed (shared with neighbors). Adequate for Phase 1–3 algorithm validation; throughput numbers from this box are not benchmarks for Phase 4+ training compute.

## Decisions deferred
- Specific card abstraction granularity (decide during Phase 2 based on observed convergence rates)
- Exact league play schedule (decide after Phase 4 blueprint exists)
- Which GPU provider for Phase 4+ training (Vast.ai / RunPod / Vultr GPU) — decide based on price/availability when Phase 2 begins
- Whether to use VPS as parallel self-play worker (decide if single-instance throughput becomes bottleneck)
- Exact mix percentages for training opponent pool composition (self-play vs. archetypes vs. bought-bots vs. league archives) — tune empirically during Phase 4–5

## Session log
- **2026-05-21 (Session 1):** Project bootstrapped on Windows. Foundational docs created. Major design decisions made: opponent anonymity as core principle, within-match adaptation at Position 2, league play for strength diversity, 42 profiles stay frozen.
- **2026-05-21 (Session 2):** Attempted WSL2 install, blocked by Windows component store corruption (DISM error 14098) that DISM /RestoreHealth could not fix. Switched runtime to existing Contabo VPS (Ubuntu 24.04). Set up GitHub repo, SSH keys, installed Python 3.10 via deadsnakes PPA. Committed initial docs verbatim, then this Session 2 update reflecting the migration. Proceeded to Phase 1 environment install and Leduc scaffold.
