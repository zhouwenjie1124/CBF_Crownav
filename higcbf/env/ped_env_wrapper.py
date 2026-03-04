from typing import NamedTuple, Optional, Tuple

import jax.numpy as jnp

from .base import MultiAgentEnv
from ..utils.typing import Action, Array, State


class PedGraphEnvState(NamedTuple):
    ped_goal: Array  # (n_ped, 2)
    ped_lidar: Array  # (n_ped, n_rays, 2)
    ped_speed_pref: Array  # (n_ped,)


class PedEnvWrapper(MultiAgentEnv):
    """
    Minimal env wrapper to use DecShareCBF for pedestrians.
    This is NOT a standalone environment; only provides dynamics + u_ref.
    """

    def __init__(self, num_peds: int, params: dict, dt: float, n_rays: int):
        super().__init__(num_agents=num_peds, area_size=1.0, max_step=1, max_travel=None, dt=dt, params=params)
        self._n_rays = n_rays

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 4

    @property
    def edge_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    def reset(self, key: Array):
        raise NotImplementedError

    def step(self, graph, action, get_eval_info: bool = False):
        raise NotImplementedError

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        v_max = float(self.params.get("ped_v_max", self.params.get("ped_speed_pref", 1.0) * 1.5))
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -v_max, -v_max])
        upper_lim = jnp.array([jnp.inf, jnp.inf, v_max, v_max])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        acc_max = float(self.params.get("ped_acc_max", 3.0))
        lower_lim = jnp.array([-acc_max, -acc_max])
        upper_lim = jnp.array([acc_max, acc_max])
        return lower_lim, upper_lim

    def control_affine_dyn(self, state: State):
        assert state.ndim == 2
        f = jnp.concatenate([state[:, 2:4], jnp.zeros((state.shape[0], 2))], axis=1)
        g = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        return f, g

    def add_edge_feats(self, graph, state):
        raise NotImplementedError

    def get_graph(self, state):
        raise NotImplementedError

    def u_ref(self, graph):
        ped_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        ped_goals = graph.env_states.ped_goal
        ped_lidar = graph.env_states.ped_lidar
        ped_speed_pref = graph.env_states.ped_speed_pref

        pos_diff = ped_goals - ped_states[:, :2]
        dist = jnp.linalg.norm(pos_diff, axis=-1, keepdims=True)
        goal_dir = pos_diff / (dist + 1e-6)

        # Simple repulsion from ped_lidar hits to bias preferred direction.
        # ped_lidar: (n_ped, n_rays, 2)
        lidar_vec = ped_states[:, None, :2] - ped_lidar
        lidar_dist = jnp.linalg.norm(lidar_vec, axis=-1, keepdims=True)
        lidar_dir = lidar_vec / (lidar_dist + 1e-6)
        lidar_force = jnp.sum(lidar_dir / (lidar_dist + 1e-2), axis=1)

        v_dir = goal_dir + 0.2 * lidar_force
        v_dir = v_dir / (jnp.linalg.norm(v_dir, axis=-1, keepdims=True) + 1e-6)
        v_pref = ped_speed_pref[:, None] * v_dir
        k_v = float(self.params.get("ped_k_v", 1.0))
        v_pref = k_v * v_pref

        vel = ped_states[:, 2:4]
        k_a = float(self.params.get("ped_k_a", 1.5))
        acc = k_a * (v_pref - vel)
        acc = jnp.clip(acc, self.action_lim()[0], self.action_lim()[1])
        return acc

    def forward_graph(self, graph, action):
        raise NotImplementedError

    def safe_mask(self, graph):
        raise NotImplementedError

    def unsafe_mask(self, graph):
        raise NotImplementedError

    def collision_mask(self, graph):
        raise NotImplementedError

    def render_video(self, rollout, video_path, Ta_is_unsafe=None, viz_opts: dict = None, **kwargs):
        return None

    def finish_mask(self, graph):
        return jnp.zeros((self.num_agents,), dtype=bool)
