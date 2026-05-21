# Session Log

Append-only record of what happened in each working session and why. STATUS.md tracks current state; this file tracks history. When something in STATUS.md changes, the reason should be findable here.

Format: most recent session at the top. Each session block notes date, what was done, what was decided, what was learned, and what's queued for next time.

---

## Session 1 — 2026-05-21
**Focus:** Project bootstrap. Establish foundational docs and local infrastructure for Claude Code.

### What was done
- Identified that the local machine is Windows 11 with an RTX 3060 Laptop GPU (6GB VRAM), not the 12GB desktop variant originally assumed in project docs. Updated ARCHITECTURE.md and DECISIONS.md to reflect this.
- Confirmed OpenSpiel does not officially support Windows native. Decided to use WSL2 + Ubuntu 22.04 as the runtime environment.
- Installed Claude Code on the Windows host. Resolved PATH issue so `claude` is callable from any PowerShell.
- Created project directory at C:\Users\Ngior\pokerbot\ with docs/ subfolder.
- Created the four foundational docs verbatim from contents provided in the planning chat: PROJECT_OVERVIEW.md, ARCHITECTURE.md, DECISIONS.md, STATUS.md.

### What was decided
- **Runtime environment: WSL2 + Ubuntu 22.04** on the Windows host, preserving every other architectural decision while avoiding Windows-native build friction.
- **Opponent anonymity as a core design principle.** No persistent identity for any opponent across matches. No pre-collected real-world hand history data feeds training. The bot's information state mirrors a competent human at an anonymous online table. This is a values-driven decision — robustness and fairness over peak exploitation EV.
- **Within-match adaptation at Position 2.** Light-to-medium online statistics nudge subgame solver ranges within the current match, anchored to the blueprint as a safety floor. When a match ends, all derived state is wiped. Position 1 (pure GTO) leaves too much EV; Position 3 (full real-time opponent modeling) is fragile and compute-expensive on a 6GB laptop GPU.
- **The 42 bought-bot profiles stay frozen.** Their role is style diversity in training and stable benchmark targets. Evolving them would lose benchmark stability and waste compute. Strength diversity in the training pool comes from league play (PSRO with archived self), which separates style diversity from strength diversity.

### What was learned / surprises
- The original ARCHITECTURE.md mixed two design goals — anonymous Nash-leaning play and identified-opponent exploitation. Removing the exploitation layer made the project cleaner.
- Claude Code defaulted to its own judgment when given file contents to write verbatim — it removed a line on its own and announced the edit after the fact. A hard rule for the project was set: when exact contents are given between BEGIN/END markers, write them exactly; concerns must be raised as questions before writing, not as silent edits. Noted in STATUS.md under Known issues.

### Workflow notes for next time
- Always verify Claude Code's output, not just its self-reports. "Done" doesn't always mean done correctly.
- The relay workflow (Claude planning chat ↔ Claude Code execution) works, but only if outputs are actually read end to end before being passed onward.

### Queued for next session
1. Install WSL2 + Ubuntu 22.04 on Windows host
2. Verify CUDA passthrough from WSL2 to the RTX 3060 Laptop GPU
3. Set up Python 3.10 environment with venv inside WSL2
4. Install PyTorch (CUDA build) and OpenSpiel inside the venv
5. Confirm PyTorch sees the GPU and OpenSpiel loads Leduc poker
6. Move the project directory from C:\Users\Ngior\pokerbot\ to the WSL2 Linux filesystem (better I/O than the Windows mount)
7. Install Claude Code inside WSL2 for ongoing project work
8. Begin Phase 1 scaffold: Leduc Deep CFR training script
