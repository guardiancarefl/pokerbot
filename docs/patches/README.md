# OpenSpiel ante patch — for the bot-design program

Two patches that add native `ante` parameter support to OpenSpiel's
universal_poker, removing the inflated-BB action-set distortion identified
in `docs/HANDOFF_RETRAIN.md`. Apply both before rebuilding pyspiel.

## What they do

`acpc_ante_master.patch` (60 lines, applies to `project_acpc_server` master)

  - Adds `int32_t ante[MAX_PLAYERS]` field to the ACPC `Game` struct
  - Parses optional `ante = N N N ...` line in GAMEDEF (after `blind`)
  - Validates: ante >= 0 per seat, blind+ante <= stack per seat
  - In `initState`: adds ante to each seat's `state->spent[]`
    (deducting from effective stack and adding to the pot), but does
    NOT include ante in `state->maxSpent` — so min-raise-to stays at
    `2 × max(blind[])` = 2 × BB per NL convention
  - `printGame`: emits `ante = ...` line only if any seat has non-zero
    ante, preserving exact pre-patch output for ante-less games

`openspiel_ante_v1.6.11.patch` (16 lines, applies to open_spiel v1.6.11)

  - Registers `ante` as a string game parameter (default empty)
  - In `parseParameters`: emits `ante = ...` line to GAMEDEF only when
    the parameter is non-empty, preserving identical GAMEDEF output for
    ante-less games

Both patches are **fully backward-compatible**: passing no `ante`
parameter (or an empty string) is bit-identical to pre-patch behavior.

## How to apply

```bash
# 1. Clone open_spiel at the matching tag
git clone --depth 1 --branch v1.6.11 \
    https://github.com/deepmind/open_spiel.git ~/open_spiel-patched
cd ~/open_spiel-patched

# 2. Fetch dependencies (pybind11, abseil-cpp, ACPC, json libs).
#    The official install.sh does this but also tries to apt-get system
#    packages — for non-root environments, run the script and ignore the
#    final apt failure (all git clones complete first), then manually
#    fetch ACPC and json deps if missed:
./install.sh || true  # ignore sudo failure
[[ -d open_spiel/games/universal_poker/acpc ]] || git clone -b master --single-branch --depth 1 \
    https://github.com/jblespiau/project_acpc_server.git \
    open_spiel/games/universal_poker/acpc
[[ -d open_spiel/json ]] || (git clone -b master https://github.com/nlohmann/json.git open_spiel/json && \
    cd open_spiel/json && git checkout 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)
[[ -d open_spiel/pybind11_json ]] || (git clone -b master https://github.com/pybind/pybind11_json.git open_spiel/pybind11_json && \
    cd open_spiel/pybind11_json && git checkout d0bf434be9d287d73a963ff28745542daf02c08f)
[[ -d open_spiel/pybind11_abseil ]] || (git clone -b master https://github.com/pybind/pybind11_abseil.git open_spiel/pybind11_abseil && \
    cd open_spiel/pybind11_abseil && git checkout 73992b5)

# 3. Apply both patches
cd ~/open_spiel-patched
git apply <path-to-pokerbot>/docs/patches/openspiel_ante_v1.6.11.patch
cd open_spiel/games/universal_poker/acpc/project_acpc_server
git apply <path-to-pokerbot>/docs/patches/acpc_ante_master.patch

# 4. Build (cmake from pip if no system cmake; disable optional deps)
cd ~/open_spiel-patched && mkdir -p build && cd build
pip install cmake   # non-root, into venv
export PATH=$(dirname $(which python3))/../bin:$PATH
export OPEN_SPIEL_BUILD_WITH_HANABI=OFF
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF
export OPEN_SPIEL_BUILD_WITH_GO=OFF
export OPEN_SPIEL_BUILD_WITH_JULIA=OFF
export OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF
export OPEN_SPIEL_BUILD_WITH_XINXIN=OFF
export OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF
export OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF
export OPEN_SPIEL_BUILD_WITH_RUST=OFF
cmake -DPython3_EXECUTABLE=$(which python3) \
      -DCMAKE_BUILD_TYPE=Release \
      ../open_spiel
make -j8 pyspiel    # ~30-60 min on a 12-core box

# 5. Back up existing pyspiel.so (don't skip this) + install the new one
cp .venv/lib/python3.10/site-packages/pyspiel.so \
   .venv/lib/python3.10/site-packages/pyspiel.so.bak.$(date +%Y-%m-%d)
cp ~/open_spiel-patched/build/python/pyspiel.so \
   .venv/lib/python3.10/site-packages/pyspiel.so

# 6. Verify
python -m pytest tests/test_openspiel_ante_patch.py -v
```

## Verification tests

`tests/test_openspiel_ante_patch.py` covers both gates separately:

  - **2a — BUILD correctness:** the rebuilt module imports, the existing
    `six_max_sng()` game string still loads + plays through, the old
    inflated-BB convention still produces min-raise = 110 (= 2 × inflated_BB),
    confirming the patch is non-invasive when no ante parameter is passed.

  - **2b — PATCH correctness:** the new `ante` parameter is registered,
    min-raise-to = 50 chips at level 1 (NOT 60, NOT 110), antes are
    deducted from per-seat stacks correctly, starting pot includes
    antes, a 2×BB open (raise-to 50) is a legal preflop action, and
    empty/zero ante parameters give bit-identical pre-patch behavior.

## Using the new ante parameter

After the patch, build the game string with an explicit `ante=` field:

```python
# Level 1 Ignition Double-Up Turbo: SB=15, BB=25, ante=5 per seat
game_str = (
    "universal_poker(betting=nolimit,numPlayers=6,numRounds=4,"
    "blind=15 25 0 0 0 0,"
    "ante=5 5 5 5 5 5,"   # NEW
    "firstPlayer=3 1 1 1,"
    "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,"
    "stack=1500 1500 1500 1500 1500 1500,"
    "bettingAbstraction=fullgame)"
)
```

For the existing codebase, the source-of-truth game string builder is
`src/nlhe/game_strings.py`. Post-retrain cleanup includes updating
`BlindLevel` / `TournamentStructure` to thread the real `ante` through
that builder (see `docs/HANDOFF_RETRAIN.md` section 1 "Library fix"
for the file-by-file change list).
