"""Gate for the raise_to->CHECK realization fix (2026-06-13).

Replays compute_click_target over EVERY decision frame carrying a
raw_record in ALL logs/live_dryrun_*.jsonl, with the committed (HEAD)
click_target vs the working-tree click_target, in BOTH modes:

  flag-off (extended=False): must be byte-identical everywhere
      (the fix lives inside the extended-gated block).
  flag-on  (extended=True):  diffs must be LIMITED to the exhibit
      class — old plan action_kind=="raise_to" AND realized_as=="check";
      new plan must be an unexecutable no-op (empty steps + reason).

compute_click_target is pure in (client_action, controls,
hero_max_commit), so this covers the full live history without
re-running make_decision. hero_max_commit is derived exactly as
live_loop does: stack[hero_seat] + bet[hero_seat] (null bet -> 0).
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/quant/pokerbot")
sys.path.insert(0, str(ROOT))

from src.nlhe.integration.click_target import (   # noqa: E402  (new code)
    compute_click_target as new_cct, plan_as_dict as new_pad,
)

# Load the committed (pre-fix) module from git HEAD.
old_src = subprocess.run(
    ["git", "-C", str(ROOT), "show", "HEAD:src/nlhe/integration/click_target.py"],
    check=True, capture_output=True, text=True).stdout
with tempfile.NamedTemporaryFile("w", suffix="_old_click_target.py",
                                 delete=False) as f:
    f.write(old_src)
    old_path = f.name
spec = importlib.util.spec_from_file_location("old_click_target", old_path)
old_mod = importlib.util.module_from_spec(spec)
sys.modules["old_click_target"] = old_mod  # dataclass needs module registered
spec.loader.exec_module(old_mod)
old_cct, old_pad = old_mod.compute_click_target, old_mod.plan_as_dict

logs = sorted(ROOT.glob("logs/live_dryrun_*.jsonl"))
n_frames = n_planned = 0
off_diffs = []
on_diffs_exhibit = []
on_diffs_other = []

for log in logs:
    for ln, line in enumerate(log.open(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = row.get("raw_record")
        ca = row.get("client_action")
        if not rec or not ca:
            continue
        n_frames += 1
        controls = rec.get("controls") or {}
        hero_seat = row.get("hero_seat", 0)
        key = f"seat{hero_seat + 1}"
        stacks = rec.get("stacks") or {}
        bets = rec.get("bets") or {}
        try:
            hmc = int(stacks.get(key) or 0) + int(bets.get(key) or 0)
        except (TypeError, ValueError):
            hmc = None
        tag = f"{log.name} seq={row.get('seq')}"

        # flag-off
        po = old_pad(old_cct(ca, controls))
        pn = new_pad(new_cct(ca, controls))
        if po != pn:
            off_diffs.append((tag, po, pn))

        # flag-on
        po = old_pad(old_cct(ca, controls, extended=True, hero_max_commit=hmc))
        pn = new_pad(new_cct(ca, controls, extended=True, hero_max_commit=hmc))
        n_planned += 1
        if po != pn:
            is_exhibit_class = (
                po.get("action_kind") == "raise_to"
                and po.get("realized_as") == "check")
            new_is_withheld = (pn.get("steps") == [] and pn.get("reason"))
            if is_exhibit_class and new_is_withheld:
                on_diffs_exhibit.append((tag, ca.get("chip_amount"),
                                         pn.get("reason")))
            else:
                on_diffs_other.append((tag, po, pn))

print(f"logs scanned:                 {len(logs)}")
print(f"frames with raw_record+action:{n_frames}")
print()
print(f"FLAG-OFF diffs (must be 0):   {len(off_diffs)}")
for t, a, b in off_diffs[:10]:
    print(f"  {t}\n    old={a}\n    new={b}")
print()
print(f"FLAG-ON diffs, exhibit class (raise_to realized_as=check -> withheld no-op):"
      f" {len(on_diffs_exhibit)}")
for t, amt, reason in on_diffs_exhibit:
    print(f"  {t}  raise_to {amt} -> WITHHELD ({reason})")
print()
print(f"FLAG-ON diffs OUTSIDE exhibit class (must be 0): {len(on_diffs_other)}")
for t, a, b in on_diffs_other[:10]:
    print(f"  {t}\n    old={a}\n    new={b}")

ok = not off_diffs and not on_diffs_other and on_diffs_exhibit
print()
print("GATE:", "GREEN" if ok else "RED")
sys.exit(0 if ok else 1)
