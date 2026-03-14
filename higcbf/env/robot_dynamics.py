"""
Robot kinematic/dynamic model for the unicycle (differential drive) robot.

Extracted from robot_ped_env.py so that dynamics can be tested independently
and reused by controllers (e.g. DecShareCBF) without importing the full env.
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from ..utils.typing import Action, Array, State


class RobotDynamics:
    """
    Encapsulates the unicycle robot dynamics and action-space logic.

    Supports three action modes (set via params["action_mode"]):
      - "discrete_delta"  : action = (Δω, Δv) acceleration increments  [default]
      - "continuous_vw"   : action = (ω_cmd, v_cmd) direct velocity commands
      - "discrete_vw_grid": action = index into a (v, w) command grid
    """

    def __init__(self, params: dict) -> None:
        self._params = params
        self._sync_action_accel_params()
        self._validate_action_mode_params()

    # ------------------------------------------------------------------
    # Param helpers
    # ------------------------------------------------------------------

    def _sync_action_accel_params(self) -> None:
        """Keep legacy delta_* param names and new a_* names in sync."""
        p = self._params
        if "a_w_max" not in p and "delta_omega_max" in p:
            p["a_w_max"] = p["delta_omega_max"]
        if "a_v_max" not in p and "delta_v_max" in p:
            p["a_v_max"] = p["delta_v_max"]
        if "a_w_step" not in p and "delta_w_step" in p:
            p["a_w_step"] = p["delta_w_step"]
        if "a_v_step" not in p and "delta_v_step" in p:
            p["a_v_step"] = p["delta_v_step"]
        if "delta_omega_max" not in p and "a_w_max" in p:
            p["delta_omega_max"] = p["a_w_max"]
        if "delta_v_max" not in p and "a_v_max" in p:
            p["delta_v_max"] = p["a_v_max"]
        if "delta_w_step" not in p and "a_w_step" in p:
            p["delta_w_step"] = p["a_w_step"]
        if "delta_v_step" not in p and "a_v_step" in p:
            p["delta_v_step"] = p["a_v_step"]

    def _validate_action_mode_params(self) -> None:
        """Validate action_mode and canonicalize vw_grid bins if needed."""
        p = self._params
        mode = p.get("action_mode", "discrete_delta")
        valid_modes = {"discrete_delta", "continuous_vw", "discrete_vw_grid"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported action_mode '{mode}'. Expected one of {sorted(valid_modes)}."
            )
        if mode != "discrete_vw_grid":
            return

        if "vw_grid_v" not in p or "vw_grid_w" not in p:
            raise ValueError("discrete_vw_grid requires params 'vw_grid_v' and 'vw_grid_w'.")

        v_bins = np.asarray(p["vw_grid_v"], dtype=np.float32).reshape(-1)
        w_bins = np.asarray(p["vw_grid_w"], dtype=np.float32).reshape(-1)
        if v_bins.size == 0 or w_bins.size == 0:
            raise ValueError("vw_grid_v and vw_grid_w must be non-empty.")
        if not np.all(np.isfinite(v_bins)) or not np.all(np.isfinite(w_bins)):
            raise ValueError("vw_grid_v and vw_grid_w must contain finite values.")

        # Canonicalize to sorted unique bins so index ordering is deterministic.
        v_bins = np.unique(v_bins)
        w_bins = np.unique(w_bins)

        v_min, v_max = float(p["v_min"]), float(p["v_max"])
        w_min, w_max = float(p["w_min"]), float(p["w_max"])
        if np.any(v_bins < v_min) or np.any(v_bins > v_max):
            raise ValueError(f"vw_grid_v values must be within [{v_min}, {v_max}].")
        if np.any(w_bins < w_min) or np.any(w_bins > w_max):
            raise ValueError(f"vw_grid_w values must be within [{w_min}, {w_max}].")

        p["vw_grid_v"] = v_bins.tolist()
        p["vw_grid_w"] = w_bins.tolist()
        p["action_discrete"] = True

    # ------------------------------------------------------------------
    # Action / state limits
    # ------------------------------------------------------------------

    def is_vw_command_mode(self) -> bool:
        """True when the action is a direct (ω, v) velocity command."""
        mode = self._params.get("action_mode", "discrete_delta")
        return mode in {"continuous_vw", "discrete_vw_grid"}

    def get_vw_grid_bins(self) -> Tuple[Array, Array]:
        """Return (v_bins, w_bins) JAX arrays for discrete_vw_grid mode."""
        v_bins = jnp.asarray(self._params["vw_grid_v"], dtype=jnp.float32).reshape(-1)
        w_bins = jnp.asarray(self._params["vw_grid_w"], dtype=jnp.float32).reshape(-1)
        return v_bins, w_bins

    def action_lim(self) -> Tuple[Action, Action]:
        """Return (lower, upper) action bounds in internal representation."""
        p = self._params
        if self.is_vw_command_mode():
            if p.get("action_mode") == "discrete_vw_grid":
                v_bins, w_bins = self.get_vw_grid_bins()
                lower = jnp.array([w_bins.min(), v_bins.min()], dtype=jnp.float32)
                upper = jnp.array([w_bins.max(), v_bins.max()], dtype=jnp.float32)
            else:
                lower = jnp.array([p["w_min"], p["v_min"]], dtype=jnp.float32)
                upper = jnp.array([p["w_max"], p["v_max"]], dtype=jnp.float32)
        else:
            a_w_max = p.get("a_w_max", p["delta_omega_max"])
            a_v_max = p.get("a_v_max", p["delta_v_max"])
            lower = jnp.array([-a_w_max, -a_v_max], dtype=jnp.float32)
            upper = jnp.array([a_w_max, a_v_max], dtype=jnp.float32)
        return lower, upper

    def state_lim(self) -> Tuple[State, State]:
        """Return (lower, upper) state bounds for the robot state vector."""
        p = self._params
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, p["v_min"], p["w_min"]])
        upper = jnp.array([jnp.inf, jnp.inf, jnp.inf, p["v_max"], p["w_max"]])
        return lower, upper

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def discretize_action(self, action: Action) -> Action:
        """Snap a continuous discrete_delta action to the allowed step grid."""
        p = self._params
        step_w = p.get("a_w_step", p["delta_w_step"])
        step_v = p.get("a_v_step", p["delta_v_step"])
        w = action[:, 0]
        v = action[:, 1]
        w_disc = jnp.where(w > step_w / 2, step_w, jnp.where(w < -step_w / 2, -step_w, 0.0))
        v_disc = jnp.where(v > step_v / 2, step_v, jnp.where(v < -step_v / 2, -step_v, 0.0))
        return jnp.stack([w_disc, v_disc], axis=-1)

    def apply_action_mode(self, agent_states: State, action: Action) -> Action:
        """
        Convert action from external representation to (Δω, Δv) increments.

        For discrete_delta mode the action is already an increment — returned as-is.
        For vw-command modes the commanded (ω, v) is converted to a bounded increment.
        """
        if not self.is_vw_command_mode():
            return action
        p = self._params
        a_w_max = p.get("a_w_max", p["delta_omega_max"])
        a_v_max = p.get("a_v_max", p["delta_v_max"])
        w_cmd = action[:, 0]
        v_cmd = action[:, 1]
        w = agent_states[:, 4]
        v = agent_states[:, 3]
        delta_w = jnp.clip(w_cmd - w, a_min=-a_w_max, a_max=a_w_max)
        delta_v = jnp.clip(v_cmd - v, a_min=-a_v_max, a_max=a_v_max)
        return jnp.stack([delta_w, delta_v], axis=-1)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def control_affine_dyn(self, state: State) -> Tuple[Array, Array]:
        """
        Unicycle control-affine dynamics:  ẋ = f(x) + g(x) u

        State layout: [x, y, θ, v, ω]
        Control:      [Δω, Δv]  (acceleration increments)

        Returns
        -------
        f : (n, 5)   drift term
        g : (n, 5, 2) control matrix (same g for all agents)
        """
        assert state.ndim == 2, "state must be 2-D: (n_agents, state_dim)"
        f = jnp.concatenate(
            [
                (jnp.cos(state[:, 2]) * state[:, 3])[:, None],  # ẋ  = v cos θ
                (jnp.sin(state[:, 2]) * state[:, 3])[:, None],  # ẏ  = v sin θ
                state[:, 4:5],                                   # θ̇  = ω
                jnp.zeros((state.shape[0], 2)),                  # v̇  = 0, ω̇ = 0 (drift)
            ],
            axis=1,
        )
        g = jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],  # Δv enters v channel
                [1.0, 0.0],  # Δω enters ω channel
            ]
        )
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        return f, g
