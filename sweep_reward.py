"""
Local Reward Parameter Sweep Runner
=====================================
Generates N trial parameter sets (random or grid), runs each as a subprocess
calling train.py, collects eval/success and eval/unsafe_frac from the console
output, ranks results, and writes sweep_results.csv.

This does NOT require wandb — it parses the last logged eval line from stdout.

Usage
-----
# Random search, 4 trials, 1 at a time:
  python sweep_reward.py --config configs/sweep_reward.yaml --n-trials 4 --jobs 1

# Parallel (2 jobs):
  python sweep_reward.py --config configs/sweep_reward.yaml --n-trials 8 --jobs 2

# Dry run (print commands, don't execute):
  python sweep_reward.py --config configs/sweep_reward.yaml --n-trials 4 --dry-run

Output
------
  sweep_results.csv    — all trial results sorted by composite score
  Console              — live progress + final top-5 table
"""
import argparse
import concurrent.futures
import csv
import os
import re
import subprocess
import sys
import time
import random
from copy import deepcopy
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _sample_param(spec: dict, rng: random.Random) -> Any:
    dist = spec.get("distribution", "uniform")
    if dist == "uniform":
        return rng.uniform(spec["min"], spec["max"])
    if dist == "int_uniform":
        return rng.randint(int(spec["min"]), int(spec["max"]))
    if dist == "log_uniform":
        import math
        lo, hi = math.log(spec["min"]), math.log(spec["max"])
        return math.exp(rng.uniform(lo, hi))
    if dist == "categorical":
        return rng.choice(spec["values"])
    if dist == "constant":
        return spec["value"]
    raise ValueError(f"Unknown distribution: {dist}")


def generate_trials(cfg: dict, n_trials: int, seed: int = 42) -> list[dict]:
    """Generate n_trials random parameter sets from the search space."""
    rng = random.Random(seed)
    params_spec = cfg.get("parameters", {})
    fixed = cfg.get("fixed", {})
    trials = []
    for _ in range(n_trials):
        trial = deepcopy(fixed)
        for key, spec in params_spec.items():
            trial[key] = _sample_param(spec, rng)
        trials.append(trial)
    return trials


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def build_command(cfg: dict, trial_params: dict, trial_idx: int, log_dir: str,
                  extra_args: list[str]) -> list[str]:
    base_cmd = cfg.get("base_command", "")
    base_parts = base_cmd.split()

    fixed = cfg.get("fixed", {})
    override_parts = [
        f"{k}={v}" for k, v in trial_params.items()
        if k not in fixed
    ]
    # Add fixed params that aren't already in base_command
    for k, v in fixed.items():
        override_parts.append(f"{k}={v}")

    cmd = (
        [sys.executable, "train.py"]
        + base_parts
        + ["--name", f"sweep_trial_{trial_idx:03d}"]
        + ["--log-dir", log_dir]
        + ["--seed", str(trial_idx)]
        + extra_args
    )
    if override_parts:
        cmd += ["--reward-override"] + override_parts

    return cmd


# ---------------------------------------------------------------------------
# Subprocess runner + result parsing
# ---------------------------------------------------------------------------

_EVAL_PATTERN = re.compile(
    r"step:\s*(\d+).*?reward:\s*([\-\d.]+).*?unsafe_frac:\s*([\-\d.]+).*?success:\s*([\-\d.]+)"
)


def _parse_last_eval(output: str) -> dict:
    """Extract the LAST eval line from train.py stdout."""
    matches = list(_EVAL_PATTERN.finditer(output))
    if not matches:
        return {"step": None, "reward": None, "unsafe_frac": None, "success": None}
    m = matches[-1]
    return {
        "step": int(m.group(1)),
        "reward": float(m.group(2)),
        "unsafe_frac": float(m.group(3)),
        "success": float(m.group(4)),
    }


def run_trial(cmd: list[str], trial_idx: int, params: dict, timeout: int,
              sub_env: dict | None = None) -> dict:
    """Run one trial subprocess and return its result dict."""
    t0 = time.time()
    label = f"trial_{trial_idx:03d}"
    print(f"[sweep] START {label}  cmd: {' '.join(cmd[:6])} ...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=sub_env,
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        print(f"[sweep] TIMEOUT {label} after {timeout}s")
        return {"trial": trial_idx, "params": params, "success": None,
                "unsafe_frac": None, "reward": None, "score": -999.0,
                "status": "timeout", "elapsed_s": timeout}
    except Exception as e:
        print(f"[sweep] ERROR {label}: {e}")
        return {"trial": trial_idx, "params": params, "success": None,
                "unsafe_frac": None, "reward": None, "score": -999.0,
                "status": f"error: {e}", "elapsed_s": time.time() - t0}

    # ---- print GPU device being used ----
    device_str = (sub_env or {}).get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "default"))
    print(f"[sweep] {label}  GPU: CUDA_VISIBLE_DEVICES={device_str}")

    elapsed = time.time() - t0
    metrics = _parse_last_eval(stdout)

    success = metrics["success"]
    unsafe_frac = metrics["unsafe_frac"]
    reward = metrics["reward"]

    if success is not None and unsafe_frac is not None:
        score = float(success) - 0.5 * float(unsafe_frac)
        status = "ok" if returncode == 0 else f"rc={returncode}"
    else:
        score = -999.0
        status = f"no_eval_found rc={returncode}"
        if returncode != 0:
            # print last few lines of stderr for debugging
            err_tail = "\n".join(stderr.strip().splitlines()[-5:])
            print(f"[sweep] STDERR {label}:\n{err_tail}")

    print(f"[sweep] DONE  {label}  success={success}  unsafe={unsafe_frac}  "
          f"score={score:.3f}  elapsed={elapsed:.0f}s  [{status}]")

    return {
        "trial": trial_idx,
        "params": params,
        "success": success,
        "unsafe_frac": unsafe_frac,
        "reward": reward,
        "score": score,
        "status": status,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _flat_row(result: dict) -> dict:
    row = {
        "trial": result["trial"],
        "success": result["success"],
        "unsafe_frac": result["unsafe_frac"],
        "reward": result["reward"],
        "score": result["score"],
        "status": result["status"],
        "elapsed_s": result["elapsed_s"],
    }
    for k, v in result["params"].items():
        row[f"p_{k}"] = v
    return row


def write_csv(results: list[dict], path: str):
    if not results:
        return
    rows = [_flat_row(r) for r in results]
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] Results written to {path}")


def print_ranking(results: list[dict], top_n: int = 5):
    ranked = sorted(results, key=lambda r: r.get("score") or -999, reverse=True)
    print(f"\n{'='*70}")
    print(f"  TOP {min(top_n, len(ranked))} TRIALS  (score = success - 0.5 * unsafe_frac)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Trial':<8} {'Success':>8} {'Unsafe':>8} {'Score':>8}  Key Params")
    print(f"{'-'*70}")
    for rank, r in enumerate(ranked[:top_n], 1):
        params_str = "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(r["params"].items())[:4]
        )
        s = r.get("success")
        u = r.get("unsafe_frac")
        sc = r.get("score")
        print(f"{rank:<5} {r['trial']:<8} {(s or 0):>8.3f} {(u or 0):>8.3f} "
              f"{(sc or 0):>8.3f}  {params_str}")
    print(f"{'='*70}\n")
    if ranked:
        best = ranked[0]
        print("Best parameter set:")
        for k, v in best["params"].items():
            print(f"  {k} = {v}")
        print()
        override_str = " ".join(
            f"{k}={'true' if v is True else 'false' if v is False else v}"
            for k, v in best["params"].items()
        )
        print(f"To reproduce:\n  python train.py <base_args> --reward-override {override_str}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local reward parameter sweep runner.")
    parser.add_argument("--config", type=str, default="configs/sweep_reward.yaml",
                        help="Path to sweep YAML config.")
    parser.add_argument("--n-trials", type=int, default=8,
                        help="Number of random trials to run.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of parallel train.py processes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for parameter sampling.")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-trial timeout in seconds.")
    parser.add_argument("--output", type=str, default="sweep_results.csv",
                        help="CSV output path.")
    parser.add_argument("--log-dir", type=str, default="./logs/sweep",
                        help="Log directory for sweep runs.")
    parser.add_argument("--extra-args", type=str, default="",
                        help="Extra args appended to every train.py call.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES for subprocesses (e.g. '0', '0,1'). "
                             "Pass '' to inherit from parent without override.")
    parser.add_argument("--mem-fraction", type=float, default=None,
                        help="XLA_PYTHON_CLIENT_MEM_FRACTION per subprocess. "
                             "Defaults to 1/jobs when --jobs > 1 to avoid OOM.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trials = generate_trials(cfg, args.n_trials, seed=args.seed)
    extra = args.extra_args.split() if args.extra_args else []

    os.makedirs(args.log_dir, exist_ok=True)

    # ---- Build subprocess environment with GPU settings ----
    sub_env = os.environ.copy()
    sub_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Disable CUDA command buffers to avoid OOM on CUDA graph instantiation
    # (XLA creates too many graphs when JIT-compiling parallel envs).
    existing_xla_flags = sub_env.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_command_buffer=" not in existing_xla_flags:
        sub_env["XLA_FLAGS"] = (existing_xla_flags + " --xla_gpu_enable_command_buffer=").strip()
    if args.device != "":
        sub_env["CUDA_VISIBLE_DEVICES"] = args.device
    mem_frac = args.mem_fraction
    if mem_frac is None and args.jobs > 1:
        mem_frac = 1.0 / args.jobs
    if mem_frac is not None:
        sub_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_frac:.3f}"

    print(f"[sweep] Config: {args.config}")
    print(f"[sweep] Trials: {args.n_trials}  Jobs: {args.jobs}  Timeout: {args.timeout}s")
    print(f"[sweep] Log dir: {args.log_dir}")
    print(f"[sweep] GPU: CUDA_VISIBLE_DEVICES={sub_env.get('CUDA_VISIBLE_DEVICES', '(inherited)')}"
          f"  MEM_FRACTION={sub_env.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '(not set)')}"
          f"  XLA_FLAGS={sub_env.get('XLA_FLAGS', '(not set)')}")
    print()

    cmds = []
    for i, params in enumerate(trials):
        cmd = build_command(cfg, params, i, args.log_dir, extra)
        cmds.append((i, params, cmd))
        if args.dry_run:
            print(f"Trial {i:03d}: {' '.join(cmd)}")
            print(f"  params: {params}\n")

    if args.dry_run:
        print("[sweep] Dry run — no processes launched.")
        return

    results = []
    if args.jobs <= 1:
        for i, params, cmd in cmds:
            r = run_trial(cmd, i, params, args.timeout, sub_env)
            results.append(r)
            write_csv(results, args.output)  # save incrementally
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_trial, cmd, i, params, args.timeout, sub_env): i
                for i, params, cmd in cmds
            }
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                results.append(r)
                write_csv(results, args.output)  # save incrementally

    print_ranking(results, top_n=5)
    write_csv(results, args.output)


if __name__ == "__main__":
    main()
