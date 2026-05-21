# Project Overview

## What we're building
A specialized 6-max No-Limit Hold'em poker bot, purpose-built for the SNG format where top 3 of 6 finishers each receive 33% of the prize pool (equal-payout / "triple-up" structure). The bot trains via a hybrid pipeline combining self-play Deep CFR, population training against bought-bot archetypes and hand-engineered styles, and league play (PSRO-style) where archived versions of the bot enter the opponent pool over time.

The end deliverable is a trained model evaluated on bb/100 against a held-out benchmark set, plus the training infrastructure to continue improving it.

## Why this format specifically
- Equal-payout SNG structure means ICM (Independent Chip Model) dynamics dominate — survival matters more than chip accumulation, which is mathematically derivable and deeply counterintuitive to humans
- Late-game stacks compress to push/fold decisions, which are analytically solvable
- Smaller strategic space than full 6-max cash, meaning a well-trained bot can get closer to optimal play with realistic compute
- Population at these formats is typically softer than equivalent-stakes cash games
- Bounded variance per game (can't lose 200bb in a hand)

## Scope (what this project includes)
- Deep CFR self-play training infrastructure
- Card and action abstraction systems
- ICM-adjusted value functions for SNG payout structures
- Population/archetype training framework
- League play (PSRO) for continuous improvement
- Opponent-conditional exploitation layer
- Evaluation infrastructure (Slumbot API integration, internal tournaments)
- Hand-history collection and analysis pipeline

## Success criteria
- Bot beats Slumbot in heads-up evaluation (positive win rate over 100k+ hands)
- Bot beats blueprint-only version of itself when full pipeline is applied (validates exploitation layer adds value)
- Bot beats the bought-bot archetypes by a meaningful margin in closed tournaments
- League play produces measurable strength gains across iterations
- ICM-adjusted strategy demonstrably differs from chip-EV strategy in expected ways (correct bubble pressure, correct in-the-money tightening)
