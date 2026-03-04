import numpy as np
import jax.numpy as jnp
import functools as ft
import jax
import jax.random as jr

from scipy.linalg import inv, solve_discrete_are
from typing import Callable, Tuple
from jax.lax import while_loop

from ..utils.typing import Array, Radius, BoolScalar, Pos, State, Action, PRNGKey
from ..utils.utils import merge01
from .obstacle import Obstacle, Rectangle, Cuboid, Sphere, Circle, MixedObstacle


def RK4_step(x_dot_fn: Callable, x: State, u: Action, dt: float) -> Array:
    k1 = x_dot_fn(x, u)
    k2 = x_dot_fn(x + 0.5 * dt * k1, u)
    k3 = x_dot_fn(x + 0.5 * dt * k2, u)
    k4 = x_dot_fn(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def get_lidar(start_point: Pos, obstacles: Obstacle, num_beams: int, sense_range: float, max_returns: int = 32):
    if isinstance(obstacles, (Rectangle, Circle, MixedObstacle)):
        thetas = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(num_beams, axis=0)
        ends = jnp.stack(
            [starts[..., 0] + jnp.cos(thetas) * sense_range, starts[..., 1] + jnp.sin(thetas) * sense_range],
            axis=-1)
    elif isinstance(obstacles, Cuboid) or isinstance(obstacles, Sphere):
        thetas = jnp.linspace(-np.pi / 2 + 2 * np.pi / num_beams, np.pi / 2 - 2 * np.pi / num_beams, num_beams // 2)
        phis = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(thetas.shape[0] * phis.shape[0] + 2, axis=0)

        def get_end_point(theta, phi):
            return jnp.array([
                start_point[0] + jnp.cos(theta) * jnp.cos(phi) * sense_range,
                start_point[1] + jnp.cos(theta) * jnp.sin(phi) * sense_range,
                start_point[2] + jnp.sin(theta) * sense_range
            ])

        def get_end_point_theta(theta):
            return jax.vmap(lambda phi: get_end_point(theta, phi))(phis)

        ends = merge01(jax.vmap(get_end_point_theta)(thetas))
        ends = jnp.concatenate([ends,
                                start_point[None, :] + jnp.array([[0., 0., sense_range]]),
                                start_point[None, :] + jnp.array([[0., 0., -sense_range]])], axis=0)
    else:
        raise NotImplementedError
    sensor_data = raytracing(starts, ends, obstacles, max_returns)

    return sensor_data


def compute_inv_ttc(rel_pos: Array, rel_vel: Array, r_sum: Array, eps: float = 1e-6) -> Array:
    """
    Compute inverse time-to-collision for disc agents using relative position/velocity.
    Returns 0 when agents are not closing or no real collision time exists.
    """
    a = jnp.sum(rel_vel ** 2, axis=-1)
    b = 2.0 * jnp.sum(rel_pos * rel_vel, axis=-1)
    c = jnp.sum(rel_pos ** 2, axis=-1) - r_sum ** 2
    disc = b ** 2 - 4.0 * a * c
    valid = (a > eps) & (disc >= 0.0) & (b < 0.0)
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t = (-b - sqrt_disc) / (2.0 * a + eps)
    inv_ttc = jnp.where(valid, 1.0 / (t + eps), 0.0)
    return inv_ttc


def inside_obstacles(points: Pos, obstacles: Obstacle, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def raytracing(starts: Pos, ends: Pos, obstacles: Obstacle, max_returns: int) -> Pos:
    # if the start point if inside the obstacle, return the start point
    is_in = inside_obstacles(starts, obstacles)

    def raytracing_single(start: Pos, end: Pos, obstacle: Obstacle):
        return obstacle.raytracing(start, end)

    def raytracing_any(start: Pos, end: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(raytracing_single, start, end))(obstacle).min()

    if obstacles.center.shape[0] == 0:
        alphas = jnp.ones(starts.shape[0]) * 1e6
    else:
        alphas = jax.vmap(ft.partial(raytracing_any, obstacle=obstacles))(starts, ends)
        alphas *= (1 - is_in)

    # assert max_returns <= alphas.shape[0]
    alphas_return = jnp.argsort(alphas)[:max_returns]

    hitting_points = starts + (ends - starts) * (alphas[..., None])

    return hitting_points[alphas_return]

# 这个函数用“随机采样 + 碰撞检查 + while_loop”生成 n 个合法的起点和目标，确保不在障碍里、彼此间隔足够、（可选）不超过最大行走距离。
def get_node_goal_rng(
        key: PRNGKey,
        side_length: float,
        dim: int,
        obstacles: Obstacle,
        n: int,
        min_dist: float,
        max_travel: float = None
) -> [Pos, Pos]:
    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    max_reset_iter = max_iter * n * 10
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))

    def get_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, this_key, _, all_nodes = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        return i_iter, this_key, jr.uniform(use_key, (dim,), minval=0, maxval=side_length), all_nodes

    def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, _, node, all_nodes = reset_input
        dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(node, obstacles, r=min_dist)
        valid = ~(collide | inside) | (i_iter >= max_iter)
        return ~valid

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, this_key, _, agent, all_goals = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        if max_travel is None:
            return i_iter, this_key, jr.uniform(use_key, (dim,), minval=0, maxval=side_length), agent, all_goals
        else:
            return i_iter, this_key, jr.uniform(
                use_key, (dim,), minval=-max_travel, maxval=max_travel) + agent, agent, all_goals

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, _, goal, agent, all_goals = reset_input
        dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(goal, obstacles, r=min_dist)
        outside = jnp.any(goal < 0) | jnp.any(goal > side_length)
        if max_travel is None:
            too_long = np.array(False, dtype=bool)
        else:
            too_long = jnp.linalg.norm(goal - agent) > max_travel
        valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array, Array]):
        # agent_id, key, states, goals, reset_iter
        agent_id, this_key, all_states, all_goals, reset_iter = reset_input
        reset_iter = reset_iter + 1
        agent_key, goal_key, this_key = jr.split(this_key, 3)
        agent_candidate = jr.uniform(agent_key, (dim,), minval=0, maxval=side_length)
        n_iter_agent, _, agent_candidate, _ = while_loop(
            cond_fun=non_valid_node, body_fun=get_node,
            init_val=(0, agent_key, agent_candidate, all_states)
        )
        all_states = all_states.at[agent_id].set(agent_candidate)

        if max_travel is None:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=side_length)
        else:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=max_travel) + agent_candidate

        n_iter_goal, _, goal_candidate, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals, reset_iter

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array, Array]):
        # agent_id, key, states, goals, reset_iter
        agent_id, this_key, all_states, all_goals, reset_iter = reset_input
        return jnp.logical_and(agent_id < n, reset_iter < max_reset_iter)

    _, _, states, goals, reset_iter = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals, jnp.array(0)))

    if not isinstance(reset_iter, jax.core.Tracer) and int(reset_iter) >= max_reset_iter:
        raise RuntimeError(
            f"get_node_goal_rng exceeded max_reset_iter={max_reset_iter} (n={n}, min_dist={min_dist}, "
            f"side_length={side_length}). Try reducing n, min_dist, obs density, or increasing area_size."
        )

    return states, goals
