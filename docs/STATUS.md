# Project Status

**Last updated:** 2026-05-21
**Current phase:** Pre-Phase-1 setup

## Done
- Architecture designed (four-layer stack)
- Format target chosen (6-max NLHE SNG, top-3 equal payout)
- Engine selected (OpenSpiel)
- Hardware confirmed (RTX 3060 local for development; VPS available as parallel worker if needed)
- Scope and non-goals defined
- Project structure planned

## In progress
- Project setup (creating files, custom instructions)
- 42 bought-bot profiles awaiting format identification
- Local Python/PyTorch/CUDA environment confirmation needed

## Next up
1. Confirm Python environment on 3060 box (Python 3.10+, PyTorch with CUDA, OpenSpiel installable)
2. Identify format of the 42 bought-bot profiles (text/XML/JSON/binary)
3. Phase 1 scaffold: OpenSpiel install, Leduc Deep CFR training script, project directory structure
4. Phase 1 validation: confirm convergence to Nash on Leduc

## Known issues / open questions
- None yet

## Decisions deferred
- Specific card abstraction granularity (decide during Phase 2 based on observed convergence rates)
- Exact league play schedule (decide after Phase 4 blueprint exists)
- Whether to burst cloud GPU for final blueprint training (decide late Phase 4)
- Whether to integrate VPS as parallel self-play worker (decide if 3060 throughput becomes bottleneck)
