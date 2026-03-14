"""
Standalone, composable reward component functions for RobotPedEnv.

Each function takes explicit JAX arrays and a params dict, returns a scalar
JAX array.  The env's step() calls these and assembles the final reward dict
so that every component is visible to the wandb logger.

Design notes
------------
- All functions are pure (no side-effects, no self).
- Each function corresponds to a single reward concept, named "r_*".
- compute_reward_components() is the single assembly point:
    it calls the relevant functions, applies enable/disable switches
    from params, and returns a flat dict whose keys are used directly
    as wandb metric names (e.g. "r/progress", "r/ttc").
- "reward" key in the returned dict = sum of all active components.

Enable/disable switches in params
----------------------------------
  use_reward_ttc      : bool  (default True  – TTC risk penalty)
  use_reward_heading  : bool  (default True  – heading alignment reward)
  use_reward_smooth   : bool  (default True  – control smoothness penalty)
  use_reward_time     : bool  (default True  – per-step time cost)
  use_reward_safe     : bool  (default True  – safety-margin shaping)
  use_reward_risk_in_main : bool
      (default True in proactive mode – adds TTC risk into r_main before
       overriding with success/collision signals)

Mode mapping (for backward compatibility)
-----------------------------------------
  reward_mode == "legacy"    : use_reward_ttc=False, heading_proactive_gate=False
  reward_mode == "proactive" : use_reward_ttc=True,  heading_proactive_gate=True
  reward_mode == "paper"     : use paper-mode assembler (r_g / r_c / r_w / r_d)
"""
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ..utils.typing import Action, Array, AgentState, State


# ---------------------------------------------------------------------------
# Shared state preparation
# ---------------------------------------------------------------------------

def compute_clearances(
    agent_pos: Array,       # (1, 2)
    ped_pos: Array,         # (n_ped, 2)
    lidar_dist: Array,      # (n_rays,) – distance from agent to lidar hit
    car_radius: float,
    ped_radius: float,
) -> tuple[Array, Array]:
    """
    Compute per-pedestrian and per-lidar-ray clearance to the agent.

    Returns
    -------
    ped_clear_all : (1, n_ped) – distance minus combined radii (neg = collision)
    obs_clear_all : (1, n_rays) – lidar clearance
    """
    if ped_pos.shape[0] > 0:
        ped_dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        ped_clear_all = ped_dist - (car_radius + ped_radius)
    else:
        ped_clear_all = jnp.full((1, 0), jnp.inf)
    obs_clear_all = (lidar_dist - car_radius)[None, :]
    return ped_clear_all, obs_clear_all


def compute_collision_flags(
    ped_clear_all: Array,                        # (1, n_ped)
    obs_clear_all: Array,                        # (1, n_rays)
    collision_obs_attempt: Optional[Array],      # (1,) bool or None
    safety_margin: float,
    safe_penalty_mode: str = "min",
) -> tuple[Array, Array, Array]:
    """
    Compute collision flags and safety-margin penalty.

    Returns
    -------
    any_collision  : (1,) bool
    any_close      : (1,) bool  – within safety_margin
    safe_penalty   : (1,) float – negative when too close
    """
    any_ped_collision = (ped_clear_all <= 0.0).any(axis=1)
    any_obs_collision = (obs_clear_all <= 0.0).any(axis=1)
    if collision_obs_attempt is not None:
        any_obs_collision = jnp.logical_or(any_obs_collision, collision_obs_attempt)
    any_collision = jnp.logical_or(any_ped_collision, any_obs_collision)

    ped_close = ped_clear_all < safety_margin
    obs_close = obs_clear_all < safety_margin
    any_close = jnp.logical_or(ped_close.any(axis=1), obs_close.any(axis=1))

    if safe_penalty_mode == "min":
        all_clear = jnp.concatenate([ped_clear_all, obs_clear_all], axis=1)
        min_clear = jnp.min(all_clear, axis=1)
        safe_penalty = jnp.where(any_close, min_clear - safety_margin, 0.0)
    else:
        safe_penalty = (
            jnp.where(ped_close, ped_clear_all - safety_margin, 0.0).sum(axis=1)
            + jnp.where(obs_close, obs_clear_all - safety_margin, 0.0).sum(axis=1)
        )
    return any_collision, any_close, safe_penalty


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def r_progress(d_prev: Array, d_next: Array, params: dict) -> Array:
    """Dense progress toward goal."""
    kappa = float(params.get("kappa_prog", 4.0))
    return kappa * (d_prev - d_next)


def r_safe_margin(safe_penalty: Array, params: dict) -> Array:
    """Penalty for being too close to pedestrians / obstacles."""
    kappa = float(params.get("kappa_safe", 1.5))
    return kappa * safe_penalty


def r_task_legacy_proactive(
    d_next: Array,
    any_collision: Array,
    progress: Array,        # r_progress already computed
    safe: Array,            # r_safe_margin already computed
    params: dict,
) -> Array:
    """
    Main task reward: success / collision / (progress + safe).
    Used by both legacy and proactive modes.
    """
    goal_tol = float(params.get("goal_tolerance", 0.1))
    kappa_succ = float(params.get("kappa_succ", 20.0))
    kappa_col = float(params.get("kappa_col", -20.0))
    return jnp.where(
        d_next <= goal_tol,
        kappa_succ,
        jnp.where(any_collision, kappa_col, progress + safe),
    )


def r_ttc_risk(
    agent_pos: Array,       # (1, 2)
    agent_states: Array,    # (1, 5)
    ped_pos: Array,         # (n_ped, 2)
    ped_states: Array,      # (n_ped, 5)  [:, 2:4] = velocity
    inv_ttc_fn,             # callable: (rel_pos, rel_vel, r_sum) -> inv_ttc
    params: dict,
) -> Array:
    """TTC-based short-horizon collision risk penalty."""
    if ped_pos.shape[0] == 0:
        return jnp.zeros((1,))
    agent_vel = jnp.concatenate(
        [
            (agent_states[:, 3] * jnp.cos(agent_states[:, 2]))[:, None],
            (agent_states[:, 3] * jnp.sin(agent_states[:, 2]))[:, None],
        ],
        axis=-1,
    )
    ped_vel = ped_states[:, 2:4]
    rel_pos = ped_pos[None, :, :] - agent_pos[:, None, :]
    rel_vel = ped_vel[None, :, :] - agent_vel[:, None, :]
    r_sum = float(params["car_radius"]) + float(params["ped_radius"])
    inv_ttc = inv_ttc_fn(rel_pos, rel_vel, r_sum)
    ttc_eps = float(params.get("ttc_eps", 1e-5))
    ttc = jnp.where(inv_ttc > 0.0, 1.0 / (inv_ttc + ttc_eps), jnp.inf)
    ttc_threshold = float(params.get("ttc_threshold", 1.0))
    kappa_ttc = float(params.get("kappa_ttc", -0.5))
    kappa_inv = float(params.get("kappa_inv", -0.2))
    ttc_mask = ttc < ttc_threshold
    return (ttc_mask * (kappa_ttc + kappa_inv / (ttc + ttc_eps))).sum(axis=1)


def r_heading(
    theta_d: Array,
    speed_gate: Optional[Array],    # None → no gating (legacy mode)
    progress_gate: Optional[Array], # None → no gating
    params: dict,
) -> Array:
    """Heading alignment reward (VO-based)."""
    theta_m = float(params.get("heading_theta_m", np.pi / 6))
    r_angle = float(params.get("heading_r_angle", 0.6))
    base = r_angle * jnp.clip(theta_m - jnp.abs(theta_d), a_min=0.0, a_max=None)
    if speed_gate is not None:
        base = base * speed_gate
    if progress_gate is not None:
        base = base * progress_gate
    return base


def r_smooth_control(action: Array, agent_states: Array, dt: float, params: dict) -> Array:
    """Penalise large accelerations and high angular velocity."""
    kappa_delta = float(params.get("kappa_delta", 0.5))
    lambda_omega = float(params.get("lambda_omega", 1.0))
    kappa_omega = float(params.get("kappa_omega", 0.000125))
    delta_v = action[:, 1] * dt
    delta_w = action[:, 0] * dt
    omega = agent_states[:, 4]
    return -(kappa_delta * (delta_v ** 2 + lambda_omega * delta_w ** 2)
             + kappa_omega * omega ** 2)


def r_time_step(params: dict) -> float:
    """Constant per-step time cost."""
    return -float(params.get("kappa_time", 0.025))


def r_timeout(is_timeout: Array, params: dict) -> Array:
    """One-shot timeout penalty."""
    kappa = float(params.get("kappa_timeout", -15.0))
    return jnp.where(is_timeout, kappa, 0.0)


# ---------------------------------------------------------------------------
# Paper-mode components
# ---------------------------------------------------------------------------

def r_paper_goal_path(d_prev, d_next, is_reached, is_timeout, params: dict) -> Array:
    """Goal / path progress reward from paper formulation."""
    r_goal = float(params.get("r_goal", 20.0))
    r_path = float(params.get("r_path", 3.2))
    return jnp.where(
        is_reached, r_goal,
        jnp.where(is_timeout, -r_goal, r_path * (d_prev - d_next))
    )


def r_paper_collision(d_min, collision, collision_obs_attempt, params: dict) -> Array:
    """Collision / obstacle-proximity penalty from paper formulation."""
    d_m = float(params.get("d_m", 1.2))
    r_collision_val = float(params.get("r_collision", -20.0))
    r_obstacle_val = float(params.get("r_obstacle", -0.2))
    coll = collision
    if collision_obs_attempt is not None:
        coll = jnp.logical_or(coll, collision_obs_attempt)
    return jnp.where(
        coll, r_collision_val,
        jnp.where(d_min <= d_m, r_obstacle_val * (d_m - d_min), 0.0)
    )


def r_paper_rotation(omega, params: dict) -> Array:
    """Angular velocity penalty from paper formulation."""
    r_rot = float(params.get("r_rotation", -0.1))
    omega_m = float(params.get("omega_m", 1.0))
    return jnp.where(jnp.abs(omega) > omega_m, r_rot * jnp.abs(omega), 0.0)


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------

def assemble_legacy_proactive(
    *,
    agent_pos: Array,
    agent_states: Array,
    next_agent_states: Array,
    goal_pos: Array,
    ped_pos: Array,
    ped_states: Array,
    lidar_dist: Array,
    action: Array,
    dt: float,
    is_timeout: Array,
    collision_obs_attempt: Optional[Array],
    theta_d: Optional[Array],      # pre-computed desired heading angle
    inv_ttc_fn,                    # callable from env
    params: dict,
    proactive: bool = True,
) -> dict:
    """
    Assemble legacy or proactive reward and return all components as a dict.

    Parameters
    ----------
    proactive : bool
        If True → TTC risk added to r_main, heading is speed/progress-gated.
        If False → legacy mode (no TTC, simple heading gate).
    """
    car_radius = float(params["car_radius"])
    ped_radius = float(params["ped_radius"])
    safety_margin = float(params.get("safety_margin", 0.2))
    safe_mode = str(params.get("safe_penalty_mode", "min"))

    d_prev = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
    d_next = jnp.linalg.norm(next_agent_states[:, :2] - goal_pos, axis=-1)

    ped_clear_all, obs_clear_all = compute_clearances(
        agent_pos, ped_pos, lidar_dist, car_radius, ped_radius
    )
    any_collision, _any_close, safe_penalty = compute_collision_flags(
        ped_clear_all, obs_clear_all, collision_obs_attempt, safety_margin, safe_mode
    )

    prog = r_progress(d_prev, d_next, params)
    safe = r_safe_margin(safe_penalty, params)
    task = r_task_legacy_proactive(d_next, any_collision, prog, safe, params)

    # TTC risk
    use_ttc = bool(params.get("use_reward_ttc", proactive))
    risk = jnp.zeros_like(task)
    if use_ttc:
        risk = r_ttc_risk(agent_pos, agent_states, ped_pos, ped_states, inv_ttc_fn, params)

    r_main = task + risk if proactive else task

    # Heading reward
    use_heading = bool(params.get("use_reward_heading", True))
    r_head = jnp.zeros_like(r_main)
    if use_heading and theta_d is not None:
        if proactive:
            speed_eps = float(params.get("anti_stuck_speed_eps", 0.05))
            v_max = float(params.get("v_max", 1.5))
            progress_eps = float(params.get("anti_stuck_progress_eps", 0.01))
            speed_gate = jnp.clip(
                (jnp.abs(next_agent_states[:, 3]) - speed_eps) / (v_max - speed_eps + 1e-6),
                a_min=0.0, a_max=1.0,
            )
            progress_gate = (d_prev - d_next > progress_eps).astype(jnp.float32)
        else:
            speed_gate, progress_gate = None, None
        r_head = r_heading(theta_d, speed_gate, progress_gate, params)
        r_main = r_main + r_head

    # Smoothness, time, timeout
    use_smooth = bool(params.get("use_reward_smooth", True))
    smooth = r_smooth_control(action, agent_states, dt, params) if use_smooth else jnp.zeros_like(r_main)

    use_time = bool(params.get("use_reward_time", True))
    r_t = r_time_step(params) * jnp.ones_like(r_main) if use_time else jnp.zeros_like(r_main)

    r_tout = r_timeout(is_timeout, params) * jnp.ones_like(r_main)

    reward = r_main + smooth + r_t + r_tout

    return {
        "reward": reward.squeeze(),
        "r_main": r_main.squeeze(),
        "r_task": task.squeeze(),
        "r_progress": prog.squeeze(),
        "r_safe": safe.squeeze(),
        "r_heading": r_head.squeeze(),
        "r_risk": risk.squeeze(),
        "r_smooth": smooth.squeeze(),
        "r_time": r_t.squeeze(),
        "r_timeout": r_tout.squeeze(),
        "safe_penalty": safe_penalty.squeeze(),
        "d_prev": d_prev.squeeze(),
        "d_next": d_next.squeeze(),
        "timeout": is_timeout.astype(jnp.float32).squeeze(),
        "any_collision": any_collision.astype(jnp.float32).squeeze(),
    }


def assemble_paper(
    *,
    agent_pos: Array,
    agent_states: Array,
    next_agent_states: Array,
    goal_pos: Array,
    ped_pos: Array,
    lidar_dist: Array,
    is_timeout: Array,
    collision_obs_attempt: Optional[Array],
    theta_d: Optional[Array],
    params: dict,
) -> dict:
    """
    Assemble paper-mode reward (r_g + r_c + r_w + r_d).
    """
    car_radius = float(params["car_radius"])
    ped_radius = float(params["ped_radius"])

    d_prev = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
    d_next = jnp.linalg.norm(next_agent_states[:, :2] - goal_pos, axis=-1)

    g_m = float(params.get("g_m", car_radius))
    is_reached = d_next < g_m

    r_g = r_paper_goal_path(d_prev, d_next, is_reached, is_timeout, params)

    # clearance for paper collision term
    if ped_pos.shape[0] > 0:
        ped_dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        ped_clear_all = ped_dist - (car_radius + ped_radius)
    else:
        ped_clear_all = jnp.full((1, 0), jnp.inf)
    obs_clear_all = (lidar_dist - car_radius)[None, :]
    all_clear = jnp.concatenate([ped_clear_all, obs_clear_all], axis=1)
    d_min = jnp.min(all_clear, axis=1)
    collision = d_min <= 0.0

    r_c = r_paper_collision(d_min, collision, collision_obs_attempt, params)
    r_w = r_paper_rotation(next_agent_states[:, 4], params)

    r_d = jnp.zeros_like(r_g)
    if theta_d is not None:
        r_d = r_heading(theta_d, speed_gate=None, progress_gate=None, params=params)

    reward = r_g + r_c + r_w + r_d

    return {
        "reward": reward.squeeze(),
        "r_g": r_g.squeeze(),
        "r_c": r_c.squeeze(),
        "r_w": r_w.squeeze(),
        "r_d": r_d.squeeze(),
        "d_prev": d_prev.squeeze(),
        "d_next": d_next.squeeze(),
        "d_min": d_min.squeeze(),
        "theta_d": theta_d.squeeze() if theta_d is not None else jnp.array(0.0),
        "is_reached": is_reached.astype(jnp.float32).squeeze(),
        "timeout": is_timeout.astype(jnp.float32).squeeze(),
        "collision": collision.astype(jnp.float32).squeeze(),
    }
