"""
Reward Component Diagnostics
=============================
Runs N episodes with random / u_ref / checkpoint policy and reports the mean,
std, min, max and share-of-total for every reward component returned by
``env.step(..., get_eval_info=True)``.

The goal is to verify that the reward structure is *balanced*:

  Healthy signal hierarchy
  ─────────────────────────
  r_task (succ / col / progress)  ▶  dominant, typically > 60 % of total |reward|
  r_safe / r_risk                 ▶  shaping,  10-30 %
  r_smooth / r_time               ▶  small penalty, < 15 %
  r_heading                       ▶  auxiliary,  5-20 %

If any shaping term exceeds r_task in magnitude the agent often learns to
"freeze" (avoid all shaping penalties) rather than reach the goal.

Usage
-----
# Random policy, default env params:
  python scripts/diagnose_reward.py --area-size 12 --obs 8 --n-ped 5

# u_ref (nominal controller):
  python scripts/diagnose_reward.py --area-size 12 --obs 8 --policy u_ref

# Load a trained checkpoint:
  python scripts/diagnose_reward.py --path logs/RobotPedEnv/ppo/seed0_xxx --policy checkpoint

# Override reward params:
  python scripts/diagnose_reward.py --area-size 12 --obs 8 \\
      --reward-override kappa_succ=30 kappa_prog=2.0
"""
import argparse
import os
import sys
import numpy as np
import yaml

# allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax.random as jr

from higcbf.env import make_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_override_value(raw: str):
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


def _parse_overrides(items):
    if not items:
        return {}
    result = {}
    for item in items:
        k, v = item.split("=", 1)
        result[k.strip()] = _parse_override_value(v)
    return result


def _load_config_yaml(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
    if isinstance(cfg, argparse.Namespace):
        return vars(cfg)
    return dict(cfg) if isinstance(cfg, dict) else {}


def _load_algo(run_dir: str, cfg: dict, env):
    """Load the latest checkpoint algo from a run directory."""
    from higcbf.algo import make_algo

    models_dir = os.path.join(run_dir, "models")
    steps = [
        int(d) for d in os.listdir(models_dir)
        if d.isdigit() and os.path.isdir(os.path.join(models_dir, d))
    ]
    if not steps:
        raise ValueError(f"No saved model steps found in {models_dir}")
    step = max(steps)

    algo = make_algo(
        algo_id=cfg.get("algo", "ppo"),
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=cfg.get("gnn_layers", 1),
        concat_robot_state=cfg.get("concat_robot_state", False),
        use_gru=cfg.get("use_gru", False),
        rnn_hidden_dim=cfg.get("rnn_hidden_dim", 64),
    )
    algo.load(os.path.join(models_dir, str(step)), step)
    print(f"[diagnose] Loaded checkpoint step={step} from {models_dir}")
    return algo


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(env, policy_fn, n_episodes: int, seed: int = 0) -> list[dict]:
    """
    Run n_episodes.  policy_fn(graph) -> action (JAX array, shape (n_agents, action_dim)).
    Returns a list of dicts: each dict maps component_name -> list[float] of per-step values.
    """
    rng = jr.PRNGKey(seed)
    all_episodes = []

    for ep in range(n_episodes):
        rng, key = jr.split(rng)
        graph = env.reset(key)
        ep_data: dict[str, list] = {}

        for _ in range(env.max_episode_steps):
            action = policy_fn(graph)
            next_graph, reward, cost, done, info = env.step(graph, action, get_eval_info=True)

            for k, v in info.items():
                try:
                    ep_data.setdefault(k, []).append(float(np.asarray(v)))
                except Exception:
                    pass
            # also track the scalar reward
            ep_data.setdefault("reward_total", []).append(float(np.asarray(reward)))

            graph = next_graph
            if bool(done):
                break

        all_episodes.append(ep_data)

    return all_episodes


# ---------------------------------------------------------------------------
# Stats and table printer
# ---------------------------------------------------------------------------

def aggregate(episodes: list[dict]) -> dict[str, dict]:
    """Flatten all episodes, compute per-component statistics."""
    flat: dict[str, list] = {}
    for ep in episodes:
        for k, vs in ep.items():
            flat.setdefault(k, []).extend(vs)

    stats = {}
    for k, vs in flat.items():
        arr = np.array(vs, dtype=np.float64)
        stats[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }
    return stats


def _share_of_total(stats: dict) -> dict[str, float]:
    """Compute each component's |mean| as a fraction of sum of all |mean| values
    (excluding 'reward_total' which is the sum itself)."""
    component_keys = [k for k in stats if k != "reward_total"]
    total_abs = sum(abs(stats[k]["mean"]) for k in component_keys)
    if total_abs < 1e-9:
        return {k: 0.0 for k in component_keys}
    return {k: abs(stats[k]["mean"]) / total_abs for k in component_keys}


_HEALTH_RULES = {
    # (component substring, healthy_share_range, advice_if_too_large, advice_if_too_small)
    "r_task":    (0.60, 1.01, "", "↑ kappa_succ or ↓ kappa_safe / kappa_time"),
    "r_progress":(0.00, 0.40, "↓ kappa_prog (may mask safety)", "↑ kappa_prog for denser shaping"),
    "r_safe":    (0.00, 0.30, "↓ kappa_safe (agent may freeze)", "OK if unsafe_frac is low"),
    "r_risk":    (0.00, 0.25, "↓ kappa_ttc (dominates task signal)", "OK"),
    "r_smooth":  (0.00, 0.15, "↓ kappa_delta / kappa_omega (too cautious)", "OK"),
    "r_time":    (0.00, 0.12, "↓ kappa_time (time pressure too high)", "OK"),
    "r_heading": (0.00, 0.20, "↓ heading_r_angle (agent only aligns, not moves)", "OK"),
}


def _health_hint(key: str, share: float) -> str:
    for substr, (lo, hi, too_large, too_small) in _HEALTH_RULES.items():
        if substr in key:
            if share > hi:
                return f"⚠  TOO LARGE → {too_large}"
            if share < lo:
                return f"⚠  TOO SMALL → {too_small}"
            return "✓"
    return ""


def print_table(stats: dict, policy_label: str):
    shares = _share_of_total(stats)
    all_keys = sorted(stats.keys())

    print(f"\n{'='*80}")
    print(f"  Policy: {policy_label}   |   Total steps sampled: {stats.get('reward_total', {}).get('n', '?')}")
    print(f"{'='*80}")
    print(f"{'Component':<22} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'Share%':>7}  Note")
    print(f"{'-'*80}")

    # Print total first
    if "reward_total" in stats:
        s = stats["reward_total"]
        print(f"{'reward_total':<22} {s['mean']:>9.3f} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {'':>7}")
        print(f"{'-'*80}")

    for k in all_keys:
        if k == "reward_total":
            continue
        s = stats[k]
        share = shares.get(k, 0.0)
        hint = _health_hint(k, share)
        print(f"{k:<22} {s['mean']:>9.3f} {s['std']:>9.3f} {s['min']:>9.3f} {s['max']:>9.3f} {share*100:>6.1f}%  {hint}")

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose reward component magnitudes.")

    # env construction (either via --path or manual flags)
    parser.add_argument("--path", type=str, default=None,
                        help="Path to a saved run directory (loads config + optional checkpoint).")
    parser.add_argument("--env", type=str, default="RobotPedEnv")
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--area-size", type=float, default=12.0)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--n-ped", type=int, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--max-step", type=int, default=256)
    parser.add_argument("--reward-mode", type=str, default=None,
                        choices=["legacy", "proactive", "paper"])
    parser.add_argument("--reward-override", type=str, nargs="*", default=None,
                        help="key=value reward param overrides, e.g. kappa_succ=30")

    # policy
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "u_ref", "checkpoint"],
                        help="Which policy to evaluate.")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # ---- build env ----
    env_params = None
    cfg = {}
    if args.path is not None:
        cfg = _load_config_yaml(args.path)
        env_params = cfg.get("env_params")

    env = make_env(
        env_id=args.env if args.path is None else cfg.get("env", args.env),
        num_agents=args.num_agents if args.path is None else cfg.get("num_agents", args.num_agents),
        area_size=args.area_size if args.path is None else cfg.get("area_size", args.area_size),
        max_step=args.max_step if args.path is None else cfg.get("max_step", args.max_step),
        num_obs=args.obs if args.path is None else cfg.get("obs", args.obs),
        num_ped=args.n_ped,
        n_rays=args.n_rays if args.path is None else cfg.get("n_rays", args.n_rays),
        env_params=env_params,
    )

    # apply reward overrides
    overrides = _parse_overrides(args.reward_override)
    if args.reward_mode is not None:
        env.params["reward_mode"] = args.reward_mode
    for k, v in overrides.items():
        env.params[k] = v

    print(f"[diagnose] Env: {env.__class__.__name__}, area={env.area_size}, "
          f"max_step={env.max_episode_steps}, n_ped={env.params.get('n_ped', 0)}, "
          f"reward_mode={env.params.get('reward_mode', 'proactive')}")
    if overrides:
        print(f"[diagnose] Reward overrides: {overrides}")

    # ---- build policy ----
    rng = jr.PRNGKey(args.seed)

    if args.policy == "random":
        lo, hi = env.action_lim()
        lo_np = np.asarray(lo)
        hi_np = np.asarray(hi)
        def policy_fn(graph):
            nonlocal rng
            rng, k = jr.split(rng)
            shape = (env.num_agents, env.action_dim)
            u = jr.uniform(k, shape=shape, minval=lo, maxval=hi)
            return u
        label = "random"

    elif args.policy == "u_ref":
        def policy_fn(graph):
            return env.u_ref(graph)
        label = "u_ref (nominal controller)"

    elif args.policy == "checkpoint":
        if args.path is None:
            parser.error("--path is required for --policy checkpoint")
        algo = _load_algo(args.path, cfg, env)
        def policy_fn(graph):
            return algo.act(graph, algo.actor_params)
        label = f"checkpoint ({args.path})"

    else:
        raise ValueError(args.policy)

    # ---- run ----
    print(f"[diagnose] Running {args.n_episodes} episodes with policy '{label}' ...")
    episodes = run_episodes(env, policy_fn, args.n_episodes, seed=args.seed)
    stats = aggregate(episodes)
    print_table(stats, label)

    # ---- summary advice ----
    shares = _share_of_total(stats)
    task_key = next((k for k in shares if "r_task" in k), None)
    if task_key is not None and shares[task_key] < 0.60:
        print(">>> ACTION NEEDED: r_task share < 60%.  The task signal is weak.  Consider:")
        print("    • Increasing kappa_succ (success reward)")
        print("    • Reducing kappa_safe, kappa_time, or kappa_ttc")
        print()


if __name__ == "__main__":
    main()
