"""Unit tests for RobotDynamics — verifies shapes and invariants without the full env."""
import jax.numpy as jnp
import numpy as np
import pytest

from higcbf.env.robot_dynamics import RobotDynamics

# Minimal params needed by RobotDynamics
_BASE_PARAMS = {
    "v_min": 0.0,
    "v_max": 1.5,
    "w_min": -1.5,
    "w_max": 1.5,
    "delta_omega_max": 0.5,
    "delta_v_max": 0.5,
    "delta_w_step": 0.5,
    "delta_v_step": 0.5,
}


def _make_state(n: int = 3) -> jnp.ndarray:
    """Return a (n, 5) dummy agent state [x, y, θ, v, ω]."""
    return jnp.zeros((n, 5)).at[:, 3].set(1.0)  # v = 1


class TestDiscreteDetlaMode:
    def setup_method(self):
        params = {**_BASE_PARAMS, "action_mode": "discrete_delta"}
        self.dyn = RobotDynamics(params)

    def test_control_affine_shapes(self):
        state = _make_state(4)
        f, g = self.dyn.control_affine_dyn(state)
        assert f.shape == (4, 5), f"f shape {f.shape}"
        assert g.shape == (4, 5, 2), f"g shape {g.shape}"

    def test_action_lim_shape(self):
        lo, hi = self.dyn.action_lim()
        assert lo.shape == (2,)
        assert hi.shape == (2,)
        assert (hi >= lo).all()

    def test_state_lim_shape(self):
        lo, hi = self.dyn.state_lim()
        assert lo.shape == (5,)
        assert hi.shape == (5,)

    def test_apply_action_mode_identity(self):
        state = _make_state(2)
        action = jnp.ones((2, 2)) * 0.1
        out = self.dyn.apply_action_mode(state, action)
        assert jnp.allclose(out, action)

    def test_is_vw_command_mode_false(self):
        assert not self.dyn.is_vw_command_mode()

    def test_drift_at_zero_velocity(self):
        state = jnp.zeros((1, 5))  # v = 0, ω = 0
        f, _ = self.dyn.control_affine_dyn(state)
        assert jnp.allclose(f, jnp.zeros((1, 5))), "drift should be zero when v=ω=0"

    def test_f_xy_components(self):
        """ẋ = v cos θ, ẏ = v sin θ."""
        state = jnp.array([[0.0, 0.0, 0.0, 2.0, 0.0]])  # θ=0, v=2
        f, _ = self.dyn.control_affine_dyn(state)
        assert jnp.allclose(f[0, 0], 2.0, atol=1e-5), "ẋ should be v=2 when θ=0"
        assert jnp.allclose(f[0, 1], 0.0, atol=1e-5), "ẏ should be 0 when θ=0"


class TestContinuousVWMode:
    def setup_method(self):
        params = {**_BASE_PARAMS, "action_mode": "continuous_vw"}
        self.dyn = RobotDynamics(params)

    def test_is_vw_command_mode_true(self):
        assert self.dyn.is_vw_command_mode()

    def test_action_lim_bounds(self):
        lo, hi = self.dyn.action_lim()
        assert float(lo[0]) == -1.5  # w_min
        assert float(hi[1]) == 1.5  # v_max

    def test_apply_action_converts_to_increment(self):
        params = {**_BASE_PARAMS, "action_mode": "continuous_vw"}
        dyn = RobotDynamics(params)
        state = jnp.zeros((1, 5))  # v=0, ω=0
        action = jnp.array([[0.3, 0.5]])  # cmd ω=0.3, v=0.5
        inc = dyn.apply_action_mode(state, action)
        # delta should be clipped to a_w_max / a_v_max
        assert inc.shape == (1, 2)
        assert float(inc[0, 1]) == pytest.approx(0.5, abs=1e-5)


class TestDiscreteVWGridMode:
    def setup_method(self):
        params = {
            **_BASE_PARAMS,
            "action_mode": "discrete_vw_grid",
            "vw_grid_v": [0.5, 1.0],
            "vw_grid_w": [-0.5, 0.0, 0.5],
        }
        self.dyn = RobotDynamics(params)

    def test_is_vw_command_mode_true(self):
        assert self.dyn.is_vw_command_mode()

    def test_get_vw_grid_bins_shape(self):
        v_bins, w_bins = self.dyn.get_vw_grid_bins()
        assert v_bins.shape == (2,)
        assert w_bins.shape == (3,)

    def test_control_affine_shapes(self):
        state = _make_state(2)
        f, g = self.dyn.control_affine_dyn(state)
        assert f.shape == (2, 5)
        assert g.shape == (2, 5, 2)
