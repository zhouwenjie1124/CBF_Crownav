import functools as ft
from typing import Tuple

import jax
import jax.numpy as jnp

from ..env.base import MultiAgentEnv
from ..env.ped_env_wrapper import PedEnvWrapper
from ..utils.graph import GraphsTuple
from ..utils.typing import Array, Done, Reward


def compute_gae_fn(
    values: Array,
    rewards: Reward,
    dones: Done,
    next_values: Array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Array, Array]:
    """Compute generalized advantage estimation for one trajectory."""
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = deltas

    def scan_fn(gae, inp):
        delta, done = inp
        gae_prev = delta + gamma * gae_lambda * (1 - done) * gae
        return gae_prev, gae_prev

    _, gaes_prev = jax.lax.scan(scan_fn, gaes[-1], (deltas[:-1], dones[:-1]), reverse=True)
    gaes = jnp.concatenate([gaes_prev, gaes[-1, None]], axis=0)
    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def compute_gae(
    values: Array,
    rewards: Reward,
    dones: Done,
    next_values: Array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Array, Array]:
    return jax.vmap(ft.partial(compute_gae_fn, gamma=gamma, gae_lambda=gae_lambda))(
        values,
        rewards,
        dones,
        next_values,
    )


def pwise_cbf_ped(
    graph: GraphsTuple,
    r_agent: float,
    r_robot: float,
    n_agent: int,
    n_obs: int,
    n_rays: int,
    k: int,
):
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=1, n_type=n_obs)

    agent_vel = a_states[:, 2:4]
    obs_vel = obs_states[:, 2:4]

    all_obs_pos = jnp.concatenate([a_states[:, :2], obs_states[:, :2]], axis=0)
    all_obs_vel = jnp.concatenate([agent_vel, obs_vel], axis=0)

    obs_r = jnp.concatenate([jnp.array([r_robot]), jnp.zeros(n_rays)], axis=0)
    all_r = jnp.concatenate([jnp.ones(n_agent) * r_agent, obs_r], axis=0)

    def single(state, agent_idx: int):
        pos = state[:2]
        vel = agent_vel[agent_idx]
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx] - (all_r[agent_idx] + all_r[k_idx]) ** 2

        k_xdiff = pos - all_obs_pos[k_idx]
        k_vdiff = vel - all_obs_vel[k_idx]
        k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
        k_h1 = k_h0_dot + 5.0 * k_dist_sq
        k_isobs = k_idx >= n_agent
        return k_h1, k_isobs

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(single, in_axes=(0, 0))
    ak_h0, ak_isobs = fn(a_states, agent_idx)
    return ak_h0, ak_isobs


def get_pwise_cbf_fn(env: MultiAgentEnv, k: int = 3):
    if isinstance(env, PedEnvWrapper):
        r_agent = env.params["ped_radius"]
        r_robot = env.params["car_radius"]
        n_agent = env.num_agents
        n_rays = env.params["n_rays"]
        n_obs = 1 + n_rays
        return ft.partial(
            pwise_cbf_ped,
            r_agent=r_agent,
            r_robot=r_robot,
            n_agent=n_agent,
            n_obs=n_obs,
            n_rays=n_rays,
            k=k,
        )

    raise NotImplementedError(f"No pairwise CBF function for env type: {type(env).__name__}")
