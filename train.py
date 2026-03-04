import argparse
import datetime
import os
import numpy as np
import yaml
from typing import Any

try:
    import ipdb
except ImportError:  # pragma: no cover - optional debug dependency
    ipdb = None

from higcbf.algo import make_algo
from higcbf.env import make_env
from higcbf.trainer.curriculum import load_curriculum_yaml
from higcbf.trainer.trainer import Trainer
from higcbf.trainer.utils import is_connected
from higcbf.utils.wandb_compat import wandb


def _parse_override_value(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_reward_overrides(items: list[str] | None) -> dict:
    if items is None:
        return {}
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --reward-override entry '{item}'. Expected format key=value."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if len(key) == 0:
            raise ValueError(f"Invalid --reward-override entry '{item}': empty key.")
        overrides[key] = _parse_override_value(value)
    return overrides


def _load_resume_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.UnsafeLoader)
    if isinstance(loaded, argparse.Namespace):
        return dict(vars(loaded))
    if isinstance(loaded, dict):
        return dict(loaded)
    return {}


def train(args):
    print(f"> Running train.py {args}")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # set up logger and resolve resume target before constructing env/algo
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    base_dir = f"{args.log_dir}/{args.env}/{args.algo}"
    if not os.path.exists(f"{args.log_dir}/{args.env}"):
        os.makedirs(f"{args.log_dir}/{args.env}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    resume_dir = None
    resume_step = None
    resume_env_params = None
    if args.resume or args.resume_dir or args.resume_step is not None:
        if args.resume_dir is not None:
            resume_dir = args.resume_dir
        else:
            seed_prefix = f"seed{args.seed}_"
            candidates = [
                d for d in os.listdir(base_dir)
                if d.startswith(seed_prefix) and os.path.isdir(os.path.join(base_dir, d))
            ]
            if len(candidates) == 0:
                raise ValueError(f"No runs found under {base_dir} for seed={args.seed}.")
            candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
            resume_dir = os.path.join(base_dir, candidates[-1])

        models_dir = os.path.join(resume_dir, "models")
        if args.resume_step is not None:
            resume_step = args.resume_step
        else:
            if not os.path.isdir(models_dir):
                raise ValueError(f"No models directory found under {resume_dir}.")
            steps = [
                int(d) for d in os.listdir(models_dir)
                if d.isdigit() and os.path.isdir(os.path.join(models_dir, d))
            ]
            if len(steps) == 0:
                raise ValueError(f"No saved steps found under {models_dir}.")
            resume_step = max(steps)

        resume_cfg = _load_resume_config(os.path.join(resume_dir, "config.yaml"))
        if resume_cfg:
            saved_env = resume_cfg.get("env")
            saved_algo = resume_cfg.get("algo")
            if saved_env is not None and str(saved_env) != str(args.env):
                raise ValueError(
                    f"Resume config env='{saved_env}' does not match CLI env='{args.env}'."
                )
            if saved_algo is not None and str(saved_algo) != str(args.algo):
                raise ValueError(
                    f"Resume config algo='{saved_algo}' does not match CLI algo='{args.algo}'."
                )

            restore_keys = (
                "num_agents",
                "gnn_layers",
                "concat_robot_state",
                "use_gru",
                "rnn_hidden_dim",
            )
            for key in restore_keys:
                if key not in resume_cfg:
                    continue
                prev_v = getattr(args, key)
                new_v = resume_cfg[key]
                if prev_v != new_v:
                    print(f"[resume] Overriding {key}: {prev_v} -> {new_v}")
                    setattr(args, key, new_v)

            resume_env_params = resume_cfg.get("env_params")
            if resume_env_params is not None and not isinstance(resume_env_params, dict):
                raise ValueError(
                    f"Invalid env_params in resume config ({type(resume_env_params)}); expected dict."
                )

        if args.steps <= resume_step:
            raise ValueError(
                f"--steps ({args.steps}) must be > resume_step ({resume_step}) to continue training."
            )

        log_dir = resume_dir
        run_name = args.name if args.name is not None else f"resume_{os.path.basename(resume_dir)}"
    else:
        log_dir = f"{base_dir}/seed{args.seed}_{start_time}"
        run_name = f"{args.algo}_{args.env}_{start_time}" if args.name is None else args.name

    # create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        ped_ignore_robot_frac=args.ped_ignore_robot_frac,
        env_params=resume_env_params,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
        ped_ignore_robot_frac=args.ped_ignore_robot_frac,
        env_params=resume_env_params,
    )
    # Persist the effective env value in config even when CLI arg is omitted.
    if hasattr(env, "params") and "ped_ignore_robot_frac" in env.params:
        args.ped_ignore_robot_frac = float(env.params["ped_ignore_robot_frac"])
    if args.env == "RobotPedEnv":
        reward_overrides = _parse_reward_overrides(args.reward_override)
        args.reward_override = reward_overrides
        for this_env in (env, env_test):
            if args.reward_mode is not None:
                this_env.params["reward_mode"] = str(args.reward_mode)
            for key, value in reward_overrides.items():
                if key not in this_env.params:
                    raise ValueError(
                        f"Unknown reward override key '{key}'. "
                        f"Available keys include: {sorted(this_env.params.keys())}"
                    )
                this_env.params[key] = value
            this_env.params["train_cbf_filter"] = bool(args.train_cbf_filter)
            this_env.params["train_cbf_alpha"] = float(args.train_cbf_alpha)
            this_env.params["train_cbf_sigma"] = float(args.train_cbf_sigma)
            this_env.params["train_cbf_weight"] = float(args.train_cbf_weight)
            this_env.params["train_cbf_use_ped"] = bool(args.train_cbf_use_ped)
            this_env.params["train_cbf_use_lidar"] = bool(args.train_cbf_use_lidar)
            this_env.params["train_cbf_use_walls"] = bool(args.train_cbf_use_walls)
            this_env.params["anti_stuck_enable"] = bool(args.anti_stuck_enable)
            if args.anti_stuck_k is not None:
                this_env.params["anti_stuck_k"] = float(args.anti_stuck_k)
            if args.anti_stuck_progress_eps is not None:
                this_env.params["anti_stuck_progress_eps"] = float(args.anti_stuck_progress_eps)
            if args.anti_stuck_speed_eps is not None:
                this_env.params["anti_stuck_speed_eps"] = float(args.anti_stuck_speed_eps)
            if args.anti_stuck_free_steps is not None:
                this_env.params["anti_stuck_free_steps"] = int(args.anti_stuck_free_steps)
            if args.anti_stuck_use_obs_blocked is not None:
                this_env.params["anti_stuck_use_obs_blocked"] = bool(args.anti_stuck_use_obs_blocked)
            if args.anti_stuck_obs_clear_thresh is not None:
                this_env.params["anti_stuck_obs_clear_thresh"] = float(args.anti_stuck_obs_clear_thresh)

    curriculum_config = None
    if args.curriculum_start_stage is not None and args.curriculum_config is None:
        raise ValueError("--curriculum-start-stage requires --curriculum-config.")
    if args.curriculum_config is not None:
        curriculum_config = load_curriculum_yaml(args.curriculum_config)
        if args.curriculum_start_stage is not None:
            curriculum_config.start_stage = int(args.curriculum_start_stage)
        args.curriculum = curriculum_config.to_dict()
    else:
        args.curriculum = None

    # create low level controller
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=args.gnn_layers,
        batch_size=256,
        buffer_size=args.buffer_size,
        horizon=args.horizon,
        lr_actor=args.lr_actor,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=args.loss_action_coef,
        loss_unsafe_coef=args.loss_unsafe_coef,
        loss_safe_coef=args.loss_safe_coef,
        loss_h_dot_coef=args.loss_h_dot_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr_value=args.lr_value,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        vf_clip_range=args.vf_clip_range,
        concat_robot_state=args.concat_robot_state,
        use_gru=args.use_gru,
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_seq_len=args.rnn_seq_len,
        rnn_minibatch_chunks=args.rnn_minibatch_chunks,
        seed=args.seed,
    )

    # get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval,
        "curriculum_config": curriculum_config,
        "curriculum_start_stage": args.curriculum_start_stage,
    }

    # create trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
        start_step=0 if resume_step is None else resume_step,
    )

    # record effective environment parameters in config for reproducibility
    env_params = dict(env.params) if hasattr(env, "params") else {}
    reward_key_allowlist = {
        "safety_margin",
        "ttc_threshold",
        "ttc_eps",
        "lambda_omega",
        "safe_penalty_mode",
        "reward_mode",
        "g_m",
        "d_r",
        "d_m",
        "omega_m",
        "heading_reward",
        "heading_n_samples",
        "heading_theta_m",
        "heading_r_angle",
        "path_reward_enable",
        "astar_grid_size",
        "astar_max_expand",
        "astar_allow_diag",
        "path_max_waypoints",
        "lookahead_enable",
        "lookahead_l0",
        "lookahead_kv",
        "lookahead_min",
        "lookahead_max",
        "lookahead_eps",
        "astar_fallback_to_goal",
    }
    reward_params = {
        k: v for k, v in env_params.items()
        if k.startswith("kappa_") or k.startswith("r_") or k in reward_key_allowlist
    }
    args.env_params = env_params
    args.reward_params = reward_params

    # save config
    wandb.config.update(args)
    wandb.config.update(algo.config)
    if not args.debug and resume_step is None:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(args, f)
            yaml.dump(algo.config, f)
    if resume_step is not None:
        algo.load(os.path.join(log_dir, "models"), resume_step)

    # start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    # custom arguments
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--algo", type=str, default="gcbf+")
    parser.add_argument("--env", type=str, default="SimpleCar")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000) # maximum training steps
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--ped-ignore-robot-frac", type=float, default=None)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=512)
    parser.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        choices=["legacy", "proactive", "paper"],
        help="RobotPedEnv reward mode. None keeps env default.",
    )
    parser.add_argument(
        "--reward-override",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Override RobotPedEnv reward params with key=value entries, "
            "e.g. --reward-override kappa_succ=60 kappa_ttc=-0.2 kappa_inv=-0.05."
        ),
    )
    parser.add_argument("--train-cbf-filter", action="store_true", default=False)
    parser.add_argument("--train-cbf-alpha", type=float, default=1.0)
    parser.add_argument("--train-cbf-sigma", type=float, default=0.3)
    parser.add_argument("--train-cbf-weight", type=float, default=0.1)
    parser.add_argument("--train-cbf-use-ped", dest="train_cbf_use_ped", action="store_true")
    parser.add_argument("--train-cbf-no-ped", dest="train_cbf_use_ped", action="store_false")
    parser.add_argument("--train-cbf-use-lidar", dest="train_cbf_use_lidar", action="store_true")
    parser.add_argument("--train-cbf-no-lidar", dest="train_cbf_use_lidar", action="store_false")
    parser.add_argument("--train-cbf-use-walls", dest="train_cbf_use_walls", action="store_true")
    parser.add_argument("--train-cbf-no-walls", dest="train_cbf_use_walls", action="store_false")
    parser.add_argument("--anti-stuck-enable", action="store_true", default=False)
    parser.add_argument("--anti-stuck-k", type=float, default=None)
    parser.add_argument("--anti-stuck-progress-eps", type=float, default=None)
    parser.add_argument("--anti-stuck-speed-eps", type=float, default=None)
    parser.add_argument("--anti-stuck-free-steps", type=int, default=None)
    parser.add_argument("--anti-stuck-use-obs-blocked", dest="anti_stuck_use_obs_blocked", action="store_true")
    parser.add_argument("--anti-stuck-no-obs-blocked", dest="anti_stuck_use_obs_blocked", action="store_false")
    parser.add_argument("--anti-stuck-obs-clear-thresh", type=float, default=None)
    parser.set_defaults(
        train_cbf_use_ped=True,
        train_cbf_use_lidar=True,
        train_cbf_use_walls=True,
        anti_stuck_use_obs_blocked=None,
    )

    # gcbf / gcbf+ arguments
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--loss-action-coef", type=float, default=0.0001)
    parser.add_argument("--loss-unsafe-coef", type=float, default=1.0)
    parser.add_argument("--loss-safe-coef", type=float, default=1.0)
    parser.add_argument("--loss-h-dot-coef", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)

    # PPO arguments
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--vf-clip-range", type=float, default=0.2)
    parser.add_argument("--concat-robot-state", action="store_true", default=False)
    parser.add_argument("--use-gru", action="store_true", default=False)
    parser.add_argument("--rnn-hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-seq-len", type=int, default=32)
    parser.add_argument("--rnn-minibatch-chunks", type=int, default=64)

    # default arguments
    parser.add_argument("--n-env-train", type=int, default=16) # number of parallel training environments
    parser.add_argument("--n-env-test", type=int, default=32) # number of parallel test environments
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume-dir", type=str, default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument(
        "--curriculum-config",
        type=str,
        default=None,
        help="Path to curriculum YAML config. When omitted, curriculum learning is disabled.",
    )
    parser.add_argument(
        "--curriculum-start-stage",
        type=int,
        default=None,
        help="Optional curriculum start stage. If resuming and state exists, saved stage takes precedence.",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    if ipdb is not None:
        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
