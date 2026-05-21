# Session Log

Append-only record of what happened in each working session and why. STATUS.md tracks current state; this file tracks history. When something in STATUS.md changes, the reason should be findable here.

Format: most recent session at the top. Each session block notes date, what was done, what was decided, what was learned, and what's queued for next time.

---

## Session 2 — 2026-05-21
**Focus:** Move from planning to execution. Get a working Linux environment with PyTorch + OpenSpiel, write the Phase 1 Leduc Deep CFR scaffold, start a training run.

### What was done
- Attempted to install WSL2 + Ubuntu 22.04 on the Windows 11 host per Session 1's queued plan.
- WSL itself (2.7.3) installed successfully, but DISM failed at `Enabling feature VirtualMachinePlatform` with exit code 14098 ("The component store has been corrupted"). Same failure occurred when DISM tried to enable `Microsoft-Windows-Subsystem-Linux`.
- Ran `DISM /Online /Cleanup-Image /StartComponentCleanup` — completed successfully, no effect on the 14098 error.
- Ran `DISM /Online /Cleanup-Image /RestoreHealth` — completed successfully (pulled replacement components from Windows Update), no effect on the 14098 error.
- Ran `sfc /scannow` — reported finding and repairing corrupt files. No effect on the 14098 error after reboot.
- Verified post-reboot that both `VirtualMachinePlatform` and `Microsoft-Windows-Subsystem-Linux` were still `State : Disabled` via `Get-WindowsOptionalFeature`. The repair had not enabled them.
- Considered escalating to an ISO-source DISM repair (downloading the Windows 11 ISO, mounting it, pointing DISM at install.wim as a clean source). Decided against it.
- Reason for the switch: the project does not require the Windows host. The earlier WSL2 decision was a path of least resistance assuming Windows-side work would be quick. With Windows servicing broken, the path of least resistance reversed direction.
- Pivoted to an existing Contabo VPS already owned by the user: Ubuntu 24.04 (Noble), 12 vCPU AMD EPYC, 48GB RAM, ~300GB free on /, no GPU. The trade is: lose the local RTX 3060 (6GB VRAM) for now, gain a clean Linux environment immediately. GPU work happens on rented cloud GPU in Phase 4+ regardless.
- Created GitHub account `guardiancarefl`. Created private repo `pokerbot`. Generated ed25519 SSH key on Contabo, registered with GitHub. Verified `ssh -T git@github.com` returns `Hi guardiancarefl!`.
- Installed system packages on Contabo: `cmake`, `build-essential`, `software-properties-common`. Added `ppa:deadsnakes/ppa`. Installed `python3.10`, `python3.10-venv`, `python3.10-dev` alongside the system's default 3.12. OpenSpiel's officially-tested Python range is 3.7–3.10, so 3.10 is the safer target.
- Created `~/pokerbot/` project directory. Initialized git, set remote, created `.gitignore` and `README.md`.
- Wrote all five foundational docs (PROJECT_OVERVIEW.md, ARCHITECTURE.md, DECISIONS.md, STATUS.md, SESSION_LOG.md) into `~/pokerbot/docs/` verbatim from the Session 1 versions in project knowledge. Committed and pushed as initial commit (`c756764`). That commit captures the pre-Session-2 state — Windows runtime, RTX 3060, WSL2 plan, all intact.
- Then updated STATUS.md, SESSION_LOG.md (this file), DECISIONS.md, and ARCHITECTURE.md to reflect the migration. Committed and pushed as the Session 2 commit.
- Created Python 3.10 venv at `~/pokerbot/.venv`. Installed PyTorch CPU build and OpenSpiel via pip. Verified both import cleanly and `pyspiel.load_game("leduc_poker")` returns a valid game.
- Created project directory structure: `src/`, `tests/`, `scripts/`, `configs/`, `runs/`.
- Wrote a Leduc Deep CFR training script using `open_spiel.python.pytorch.deep_cfr` as the reference implementation. Added logging, checkpointing, and periodic exploitability evaluation.
- Started the training run. Verified the script ran past initialization and was producing iteration output before declaring the session done.

### What was decided
- **Runtime moved from WSL2-on-Windows to Contabo VPS.** Recorded as a new entry in DECISIONS.md that supersedes (but does not delete) the original WSL2 entry. The original decision is preserved for historical reference.
- **GPU work deferred to rented cloud hardware in Phase 4+.** Phases 1–3 don't need a GPU — Deep CFR's bottleneck through Phase 3 is CPU-bound trajectory generation, not network training. The local RTX 3060 was always going to be insufficient for Phase 4 anyway; this just makes the eventual cloud migration explicit instead of optional.
- **OpenSpiel reference implementation, not custom CFR.** Used `open_spiel.python.pytorch.deep_cfr` as the Phase 1 implementation. The goal of Phase 1 is pipeline validation, not algorithmic research — reimplementing CFR adds risk without research payoff.
- **Two-commit migration in git.** First commit captures the verbatim pre-Session-2 state (Windows-era plan). Second commit captures Session 2's changes. This keeps the history readable: anyone reading later can see exactly what changed and when, instead of an opaque "everything was Contabo from the start."

### What was learned / surprises
- The Windows component store corruption was the kind of environmental problem that no amount of project-specific planning would have caught. Worth flagging that operating systems can be broken in ways that look like "your install command failed" but are actually "your system needs an in-place repair." Knowing when to bail vs. when to push through is a real skill.
- The Contabo box being already paid for, already running, and already on Ubuntu turned a multi-hour Windows repair problem into a five-minute SSH session. The pre-existing infrastructure on the user's side was the unsung hero of this session.
- Initial heredoc paste of `.gitignore` and `README.md` produced visually-mangled terminal echo (mid-line text from one block appearing inside another) but the actual file contents were fine. Lesson: always `cat` and `wc -l` after writing files via heredoc — the terminal echo is unreliable, the file contents are what matter.

### Queued for next session
1. Check the Leduc training run's progress. Exploitability should be decreasing toward known Nash (Leduc heads-up Nash exploitability is on the order of 0.01 mbb/g; reaching that takes hundreds of CFR iterations).
2. Document Phase 1 result in SESSION_LOG.md and update STATUS.md.
3. Plan Phase 2: heads-up NLHE prototype with coarse abstraction. Likely needs a GPU at some point. Decide GPU provider (Vast.ai / RunPod / Vultr GPU) and rough budget.
4. Identify the 42 bought-bot profile format (text/XML/JSON/binary) — even though Phase 3 is a ways off, knowing the format unblocks parser work that can happen in parallel.

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
- **Runtime environment: WSL2 + Ubuntu 22.04** on the Windows host, preserving every other architectural decision while avoiding Windows-native build friction. (Superseded in Session 2.)
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
