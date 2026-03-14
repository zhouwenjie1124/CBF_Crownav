<div align="center">

# HIG-CBF CrowdNav

[Dependencies](#Dependencies) •
[Installation](#Installation) •
[Run](#Run)

</div>

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n higcbf python=3.10
conda activate higcbf
cd higcbf
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

## Installation

Install HIG-CBF:

```bash
pip install -e .
```

## Run

### Hyper-parameters

Use `python train.py --help` to view the current supported training flags.

### Train

To train the robot pedestrian environment with PPO, use:

```bash
python train.py \
  --algo ppo \
  --env RobotPedEnv \
  -n 1 \
  --area-size 12 \
  --obs 8 \
  --n-rays 16 \
  --steps 1500 \
  --n-env-train 32 \
  --n-env-test 32 \
  --lr-actor 2e-4 \
  --lr-value 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --ppo-epochs 5 \
  --minibatch-size 512 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --gnn-layers 1 \
  --train-cbf-filter \
  --anti-stuck-enable \
  --curriculum-config configs/curriculum_robotped_metric.yaml \
  --use-gru
```

Reward tuning (RobotPedEnv):

```bash
python train.py \
  --algo ppo \
  --env RobotPedEnv \
  -n 1 \
  --area-size 12 \
  --obs 8 \
  --reward-mode proactive \
  --reward-override kappa_succ=60 kappa_ttc=-0.2 kappa_inv=-0.05 kappa_timeout=-10
```

Quick reward sweep (train + test + rank):

```bash
python sweep_reward.py --repo-dir . --steps 300 --eval-epi 50
```

Sync wandb data

```bash
wandb: wandb sync logs/RobotPedEnv/ppo/seed0_20260224003826/wandb/offline-run-20260224_003826-xw8pf8m4
```

Curriculum learning (metric-driven, with promotion/demotion):

```bash
python train.py \
  --algo ppo \
  --env RobotPedEnv \
  -n 1 \
  --area-size 12 \
  --steps 2000 \
  --curriculum-config configs/curriculum_robotped_metric.yaml
```

Optional: set a start stage manually (resume uses saved curriculum state if available):

```bash
python train.py \
  --algo ppo \
  --env RobotPedEnv \
  -n 1 \
  --area-size 12 \
  --steps 2000 \
  --curriculum-config configs/curriculum_robotped_metric.yaml \
  --curriculum-start-stage 1
```

Note: each curriculum stage switch updates env parameters and triggers a one-time JIT recompilation of rollout/eval functions.

### Resume Training

Training automatically saves checkpoints under:

```
./logs/<env>/<algo>/seed<seed>_<timestamp>/models/<step>/
```

To resume from the latest checkpoint of the latest run for the same `env/algo/seed`:

```bash
python train.py \
  --algo ppo \
  --env RobotPedEnv \
  -n 1 \
  --area-size 12 \
  --obs 8 \
  --n-rays 16 \
  --steps 1500 \
  --resume \
  --resume-dir logs/RobotPedEnv/ppo/seed0_20260223001736 \
  --lr-actor 2e-4 \
  --lr-value 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --ppo-epochs 5 \
  --minibatch-size 256 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --gnn-layers 1 \
  --n-env-train 32 \
  --n-env-test 32
```

Note: `--steps` is still the total training steps, so it must be **greater** than the resume step.

In our paper, we use 8 agents with 1000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `-n`: number of agents
- `--env`: environment, currently `RobotPedEnv`
- `--algo`: algorithm, `ppo` or `dec_share_cbf`
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: name of the experiment
- `--debug`: debug mode: no recording, no saving
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of environments for training
- `--n-env-test`: number of environments for testing
- `--log-dir`: path to save the training logs
- `--eval-interval`: interval of evaluation
- `--eval-epi`: number of episodes for evaluation
- `--save-interval`: interval of saving the model
- `--resume`: auto-resume from the latest run and latest checkpoint
- `--resume-dir`: resume from a specific run directory
- `--resume-step`: resume from a specific checkpoint step

In addition, use the following flags to specify hyper-parameters:

- `--alpha`: CBF alpha (for `dec_share_cbf`)
- `--lr-actor`: learning rate of the policy network
- `--lr-value`: learning rate of the value network
- `--gamma`: PPO discount factor
- `--gae-lambda`: PPO GAE lambda
- `--clip-ratio`: PPO clip ratio
- `--ppo-epochs`: PPO update epochs per rollout
- `--minibatch-size`: PPO minibatch size
- `--max-grad-norm`: PPO gradient clipping norm

### Test

To test the PPO model, use:

```bash
JAX_PLATFORM_NAME=cpu python test.py --path logs/RobotPedEnv/ppo/seed0_20260221165114 --epi 100 --area-size 12 -n 1 --obs 4  --max-step 256 --no-video --continue-after-collision
```

To test CBF and sfm, use:

```bash
python test.py --env RobotPedEnv -n 1 --area-size 10 --obs 4 --u-ref --epi 5
```

This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--n-rays`: number of LiDAR rays
- `--alpha`: CBF alpha (used by `dec_share_cbf`)
- `--max-travel`: maximum travel distance of agents
- `--cbf`: plot the CBF contour of this agent, only support 2D environments
- `--seed`: random seed
- `--debug`: debug mode
- `--cpu`: use CPU
- `--u-ref`: test the nominal controller
- `--env`: test environment (not needed if the log folder is specified)
- `--algo`: test algorithm (not needed if the log folder is specified)
- `--step`: test step (not needed if testing the last saved model)
- `--epi`: number of episodes to test
- `--offset`: offset of the random seeds
- `--no-video`: do not generate videos
- `--log`: log the results to a file
- `--dpi`: dpi of the video
- `--nojit-rollout`: do not use jit to speed up the rollout, used for large-scale tests
