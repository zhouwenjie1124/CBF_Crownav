# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIG-CBF CrowdNav is a JAX/Flax-based reinforcement learning framework for training robots to navigate safely in pedestrian crowds using Control Barrier Functions (CBFs) and Graph Neural Networks (GNNs). The core algorithm is PPO with an optional CBF safety filter layer.

## Common Commands

### Installation
```bash
conda create -n higcbf python=3.10 && conda activate higcbf
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
python train.py --algo ppo --env RobotPedEnv -n 1 --area-size 12 --obs 8 --n-rays 16 --steps 1500 --n-env-train 32 --n-env-test 32
```

Key flags: `--debug` (no JIT/saving), `--resume` / `--resume-dir <path>`, `--reward-mode [legacy|proactive|paper]`, `--reward-override kappa_succ=60 kappa_ttc=-0.2`, `--curriculum-config configs/curriculum_robotped_metric.yaml`

### Testing
```bash
JAX_PLATFORM_NAME=cpu python test.py --path logs/RobotPedEnv/ppo/seed0_<timestamp> --epi 100 -n 1 --obs 4 --area-size 12 --no-video
```

Key flags: `--u-ref` (test nominal CBF/SFM controller without learned policy), `--continue-after-collision`, `--nojit-rollout` (large-scale tests)

### Reward Sweeping
```bash
python sweep_reward.py --repo-dir . --steps 300 --eval-epi 50
```
Config at `configs/sweep_reward.yaml` (supports bayesian search via wandb or local parallel runs).

### Tests
```bash
pytest tests/
```

### W&B Sync (for offline runs)
```bash
wandb sync logs/RobotPedEnv/ppo/<run_dir>/wandb/offline-run-*
```

## Architecture

### Entry Points
- `train.py` — parses args, calls `make_env()` + `make_algo()`, instantiates `Trainer`, runs training loop
- `test.py` — loads checkpoint, runs rollouts, optionally renders video
- `sweep_reward.py` — orchestrates reward hyperparameter search (local parallel or wandb sweep)

### Core Modules (`higcbf/`)

**`env/`** — Environment layer (JAX-functional style)
- `robot_ped_env.py` — Main `RobotPedEnv` class; assembles robot dynamics, pedestrian sim, rewards, obstacles, and LiDAR observations into a `MultiAgentEnv`
- `robot_dynamics.py` — Robot kinematics (unicycle/single-integrator); pure functions operating on state arrays
- `rewards.py` — Composable reward components (success, collision, TTC, progress, etc.); assembled via `assemble_reward_fn()`
- `path_planner.py` — Geometric path planning utilities
- `ped_sim.py` — Pedestrian controllers: SFM, CBF, ORCA
- `obstacle.py` — Static rectangle/circle obstacles

**`algo/`** — Algorithm layer
- `ppo.py` — PPO implementation using Flax; wraps `PPOPolicy` + `ValueNet`; `update()` does gradient steps
- `dec_share_cbf.py` — Decentralized CBF baseline controller
- `module/policy.py` — Policy network: GNN encoder → optional GRU → action head
- `module/cbf.py` — CBF module: solves QP to project actions into safe set (uses `jaxproxqp`)
- `module/value.py` — Value network (GNN + MLP)

**`nn/`** — Reusable network primitives
- `gnn.py` — Message-passing GNN (jraph-based); agents and obstacles are graph nodes
- `mlp.py` — Standard MLP layers

**`trainer/`** — Training orchestration
- `trainer.py` — `Trainer` class: rollout collection, evaluation, checkpointing (orbax), W&B logging
- `curriculum.py` — Adjusts env difficulty (num obstacles, num pedestrians) based on success metrics
- `buffer.py` / `data.py` — Rollout storage and data structures

**`utils/`** — Shared utilities (JAX helpers, graph types, W&B compat layer)

### Key Design Patterns

- **JAX-functional**: Environments are pure functions; state is explicit. `jit`/`vmap` used throughout. Use `--debug` to disable JIT when debugging.
- **Graph observations**: Agents and obstacles are graph nodes. GNN processes interaction edges. See `utils/graph.py` for the `Graph` type.
- **Modular rewards**: Each reward term is a standalone function in `rewards.py`. New rewards are added there and registered in `assemble_reward_fn()`.
- **Checkpointing**: Saved as Flax `TrainState` via orbax at `logs/{env}/{algo}/seed{seed}_{timestamp}/models/{step}/`.

### Log Directory Structure
```
logs/{env}/{algo}/seed{seed}_{timestamp}/
├── config.yaml          # full training config snapshot
├── models/{step}/       # orbax checkpoints
├── curriculum_state.yaml
└── wandb/               # W&B artifacts (use wandb sync for offline)
```
