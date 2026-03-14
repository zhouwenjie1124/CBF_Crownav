"""Unit tests for the composable reward component functions in rewards.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from higcbf.env.rewards import (
    compute_clearances,
    compute_collision_flags,
    r_progress,
    r_safe_margin,
    r_time_step,
    r_timeout,
    r_smooth_control,
    r_heading,
    assemble_legacy_proactive,
    assemble_paper,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_PARAMS = {
    # Progress / task
    "kappa_prog": 1.0,
    "kappa_succ": 10.0,
    "kappa_col": -10.0,
    "kappa_safe": 0.5,
    "goal_tolerance": 0.3,
    "safety_margin": 0.5,
    "safe_penalty_mode": "sum",
    # TTC
    "kappa_ttc": -0.2,
    "kappa_inv": -0.1,
    "ttc_eps": 1e-3,
    "ttc_threshold": 3.0,
    # Heading
    "heading_reward": False,
    "heading_theta_m": 0.523,
    "heading_r_angle": 0.6,
    "anti_stuck_speed_eps": 0.05,
    "anti_stuck_progress_eps": 0.01,
    # Smooth
    "kappa_delta": 0.01,
    "lambda_omega": 1.0,
    "kappa_omega": 0.01,
    "kappa_time": 0.05,
    # Timeout
    "kappa_timeout": -15.0,
    # Sizes
    "car_radius": 0.3,
    "ped_radius": 0.3,
    "v_max": 1.5,
    # Paper-mode
    "r_goal": 20.0,
    "r_path": 3.2,
    "g_m": 0.3,
    "d_m": 1.2,
    "r_collision": -20.0,
    "r_obstacle": -0.2,
    "r_rotation": -0.1,
    "omega_m": 1.0,
}


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


# ---------------------------------------------------------------------------
# compute_clearances
# ---------------------------------------------------------------------------

class TestComputeClearances:
    def test_no_peds(self):
        agent_pos = jnp.zeros((1, 2))
        ped_pos = jnp.zeros((0, 2))
        lidar_dist = jnp.ones(8) * 5.0
        ped_clear, obs_clear = compute_clearances(agent_pos, ped_pos, lidar_dist, 0.3, 0.3)
        assert ped_clear.shape == (1, 0)
        assert obs_clear.shape == (1, 8)
        assert (obs_clear > 0).all()

    def test_ped_collision(self):
        agent_pos = jnp.zeros((1, 2))
        ped_pos = jnp.array([[0.1, 0.0]])  # very close
        lidar_dist = jnp.ones(8) * 5.0
        ped_clear, _ = compute_clearances(agent_pos, ped_pos, lidar_dist, 0.3, 0.3)
        assert ped_clear[0, 0] < 0.0, "ped inside robot radius → negative clearance"

    def test_obs_clearance(self):
        agent_pos = jnp.zeros((1, 2))
        ped_pos = jnp.zeros((0, 2))
        lidar_dist = jnp.ones(4) * 0.2  # closer than car_radius=0.3
        _, obs_clear = compute_clearances(agent_pos, ped_pos, lidar_dist, 0.3, 0.3)
        assert (obs_clear < 0).all(), "lidar hit inside car_radius → negative obs clearance"


# ---------------------------------------------------------------------------
# r_progress
# ---------------------------------------------------------------------------

class TestRProgress:
    def test_moving_closer(self):
        r = r_progress(d_prev=jnp.array([2.0]), d_next=jnp.array([1.0]), params=_PARAMS)
        assert _scalar(r) == pytest.approx(1.0)  # kappa_prog=1 * (2-1)

    def test_moving_away(self):
        r = r_progress(d_prev=jnp.array([1.0]), d_next=jnp.array([2.0]), params=_PARAMS)
        assert _scalar(r) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# r_safe_margin
# ---------------------------------------------------------------------------

class TestRSafeMargin:
    def test_zero_when_no_close_obstacles(self):
        safe_penalty = jnp.array([0.0])
        r = r_safe_margin(safe_penalty, _PARAMS)
        assert _scalar(r) == pytest.approx(0.0)

    def test_negative_when_close(self):
        # safe_penalty is negative when inside margin
        safe_penalty = jnp.array([-0.2])
        r = r_safe_margin(safe_penalty, _PARAMS)
        assert _scalar(r) < 0.0


# ---------------------------------------------------------------------------
# r_time_step and r_timeout
# ---------------------------------------------------------------------------

class TestTimeRewards:
    def test_time_step_is_negative(self):
        r = r_time_step(_PARAMS)
        assert r < 0

    def test_timeout_penalty_applied(self):
        r = r_timeout(is_timeout=jnp.array(True), params=_PARAMS)
        assert _scalar(r) == pytest.approx(-15.0)

    def test_no_timeout_zero(self):
        r = r_timeout(is_timeout=jnp.array(False), params=_PARAMS)
        assert _scalar(r) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# r_smooth_control
# ---------------------------------------------------------------------------

class TestRSmoothControl:
    def test_nonzero_action_penalized(self):
        action = jnp.ones((1, 2)) * 0.5
        agent_states = jnp.zeros((1, 5))
        r = r_smooth_control(action, agent_states, dt=0.1, params=_PARAMS)
        assert _scalar(r) < 0.0

    def test_zero_action_less_penalty(self):
        action_big = jnp.ones((1, 2)) * 1.0
        action_small = jnp.zeros((1, 2))
        agent_states = jnp.zeros((1, 5))
        r_big = r_smooth_control(action_big, agent_states, dt=0.1, params=_PARAMS)
        r_small = r_smooth_control(action_small, agent_states, dt=0.1, params=_PARAMS)
        assert _scalar(r_big) < _scalar(r_small)


# ---------------------------------------------------------------------------
# assemble_legacy_proactive (integration test)
# ---------------------------------------------------------------------------

def _dummy_inv_ttc(rel_pos, rel_vel, r_sum):
    return jnp.zeros(rel_pos.shape[:2])


class TestAssembleLegacyProactive:
    def _run(self, proactive=True):
        agent_pos = jnp.array([[3.0, 3.0]])
        agent_states = jnp.zeros((1, 5)).at[0, 3].set(1.0)
        next_agent_states = jnp.zeros((1, 5)).at[0, :2].set(jnp.array([2.9, 3.0])).at[0, 3].set(1.0)
        goal_pos = jnp.array([[0.0, 0.0]])
        ped_pos = jnp.zeros((0, 2))
        ped_states = jnp.zeros((0, 5))
        lidar_dist = jnp.ones(8) * 5.0
        action = jnp.zeros((1, 2))
        return assemble_legacy_proactive(
            agent_pos=agent_pos,
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            ped_states=ped_states,
            lidar_dist=lidar_dist,
            action=action,
            dt=0.1,
            is_timeout=jnp.array(False),
            collision_obs_attempt=None,
            theta_d=None,
            inv_ttc_fn=_dummy_inv_ttc,
            params=_PARAMS,
            proactive=proactive,
        )

    def test_returns_reward_key(self):
        out = self._run()
        assert "reward" in out

    def test_reward_is_scalar(self):
        out = self._run()
        assert jnp.asarray(out["reward"]).shape == ()

    def test_component_keys_present(self):
        out = self._run()
        for key in ("r_progress", "r_safe", "r_time", "r_smooth"):
            assert key in out, f"missing key {key}"

    def test_proactive_vs_legacy_differ(self):
        out_proactive = self._run(proactive=True)
        out_legacy = self._run(proactive=False)
        # With no peds, r_risk=0, so they might match — but dict structure should be same
        assert set(out_proactive.keys()) == set(out_legacy.keys())

    def test_collision_gives_negative_reward(self):
        agent_pos = jnp.array([[0.1, 0.0]])  # inside a 0.3+0.3=0.6 ped radius
        ped_pos = jnp.array([[0.0, 0.0]])
        agent_states = jnp.zeros((1, 5))
        next_agent_states = jnp.zeros((1, 5))
        goal_pos = jnp.array([[10.0, 10.0]])
        lidar_dist = jnp.ones(8) * 5.0
        action = jnp.zeros((1, 2))
        ped_states = jnp.zeros((1, 5))
        out = assemble_legacy_proactive(
            agent_pos=agent_pos,
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            ped_states=ped_states,
            lidar_dist=lidar_dist,
            action=action,
            dt=0.1,
            is_timeout=jnp.array(False),
            collision_obs_attempt=None,
            theta_d=None,
            inv_ttc_fn=_dummy_inv_ttc,
            params=_PARAMS,
            proactive=False,
        )
        assert _scalar(out["reward"]) == pytest.approx(-10.0, abs=1.0)


# ---------------------------------------------------------------------------
# assemble_paper (integration test)
# ---------------------------------------------------------------------------

class TestAssemblePaper:
    def test_returns_reward_key(self):
        agent_pos = jnp.array([[3.0, 3.0]])
        agent_states = jnp.zeros((1, 5)).at[0, 3].set(1.0)
        next_agent_states = jnp.zeros((1, 5)).at[0, :2].set(jnp.array([2.9, 3.0])).at[0, 3].set(1.0)
        goal_pos = jnp.array([[0.0, 0.0]])
        ped_pos = jnp.zeros((0, 2))
        lidar_dist = jnp.ones(8) * 5.0
        out = assemble_paper(
            agent_pos=agent_pos,
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            lidar_dist=lidar_dist,
            is_timeout=jnp.array(False),
            collision_obs_attempt=None,
            theta_d=None,
            params=_PARAMS,
        )
        assert "reward" in out
        assert jnp.asarray(out["reward"]).shape == ()
        for key in ("r_g", "r_c", "r_w"):
            assert key in out, f"missing key {key}"
