import functools as ft
import os
import pickle
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from flax.training.train_state import TrainState

from higcbf.algo.base import MultiAgentController
from higcbf.algo.module.policy import PPOPolicy
from higcbf.algo.module.value import ValueNet
from higcbf.algo.utils import compute_gae
from higcbf.env.base import MultiAgentEnv
from higcbf.trainer.data import Rollout
from higcbf.trainer.utils import compute_norm_and_clip
from higcbf.utils.graph import GraphsTuple
from higcbf.utils.typing import Action, Array, Params, PRNGKey
from higcbf.utils.utils import merge01, jax_vmap


class PPOBatch(NamedTuple):
    graph: GraphsTuple
    actions: Action
    log_pis: Array
    returns: Array
    advantages: Array
    old_values: Array


class PPO(MultiAgentController):

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            gnn_layers: int = 1,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_ratio: float = 0.2,
            ppo_epochs: int = 10,
            minibatch_size: int = 256,
            lr_actor: float = 3e-4,
            lr_value: float = 1e-3,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            vf_clip_range: Optional[float] = None,
            max_grad_norm: float = 0.5,
            seed: int = 0,
            concat_robot_state: bool = False,
            use_gru: bool = False,
            rnn_hidden_dim: int = 64,
            rnn_seq_len: int = 32,
            rnn_minibatch_chunks: int = 64,
            use_hetero_attn: bool = False,
            **kwargs
    ):
        super().__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        params = getattr(env, "params", {}) or {}
        self._action_mode = params.get("action_mode", None)
        self._discrete_joint = self._action_mode == "discrete_vw_grid"
        if self._action_mode is None:
            self._action_discrete = bool(params.get("action_discrete", False))
        else:
            self._action_discrete = self._action_mode in {"discrete_delta", "discrete_vw_grid"}
        self._discrete_action_values = None
        self._discrete_action_grid = None
        n_discrete_actions = None
        if self._discrete_joint:
            self._discrete_action_grid = self._build_vw_grid()
            n_discrete_actions = int(self._discrete_action_grid.shape[0])
        elif self._action_discrete:
            step_w = float(getattr(self._env, "params", {}).get("a_w_step", 0.0))
            step_v = float(getattr(self._env, "params", {}).get("a_v_step", 0.0))
            if step_w <= 0.0:
                step_w = float(getattr(self._env, "params", {}).get("delta_w_step", 0.1))
            if step_v <= 0.0:
                step_v = float(getattr(self._env, "params", {}).get("delta_v_step", 0.1))
            self._discrete_action_values = jnp.array(
                [[-step_w, 0.0, step_w], [-step_v, 0.0, step_v]], dtype=jnp.float32
            )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_clip_range = vf_clip_range if (vf_clip_range is not None and vf_clip_range > 0.0) else None
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.gnn_layers = gnn_layers
        self.concat_robot_state = concat_robot_state
        self.use_gru = use_gru
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_seq_len = rnn_seq_len
        self.rnn_minibatch_chunks = rnn_minibatch_chunks
        self.use_hetero_attn = use_hetero_attn

        u_lb, u_ub = env.action_lim()
        self._action_scale = (u_ub - u_lb) / 2.0
        self._action_bias = (u_ub + u_lb) / 2.0

        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        self.actor = PPOPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers,
            discrete_action=self._action_discrete,
            action_bins=3,
            discrete_joint=self._discrete_joint,
            n_discrete_actions=n_discrete_actions,
            concat_robot_state=concat_robot_state,
            use_gru=use_gru,
            rnn_hidden_dim=rnn_hidden_dim,
            use_hetero_attn=use_hetero_attn,
        )
        self.value = ValueNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            gnn_layers=gnn_layers,
            concat_robot_state=concat_robot_state,
            use_gru=use_gru,
            rnn_hidden_dim=rnn_hidden_dim,
            use_hetero_attn=use_hetero_attn,
        )

        key = jr.PRNGKey(seed)
        actor_key, value_key, key = jr.split(key, 3)
        if self.use_gru:
            init_done = jnp.array(False)
            init_actor_carry = self.actor.init_carry(n_agents=n_agents)
            init_value_carry = self.value.init_carry(n_agents=n_agents)
            actor_params = self.actor.dist_rnn.init(
                actor_key, nominal_graph, n_agents=n_agents, carry=init_actor_carry, done_prev=init_done
            )
            value_params = self.value.net_rnn.init(
                value_key, nominal_graph, n_agents, init_value_carry, init_done
            )
        else:
            actor_params = self.actor.dist.init(actor_key, nominal_graph, n_agents=n_agents)
            value_params = self.value.net.init(value_key, nominal_graph, n_agents)

        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.sample_action,
            params=actor_params,
            tx=optax.adam(learning_rate=lr_actor),
        )
        self.value_train_state = TrainState.create(
            apply_fn=self.value.get_value,
            params=value_params,
            tx=optax.adam(learning_rate=lr_value),
        )

        self.key = key
        self.rng = np.random.default_rng(seed=seed + 1)

    def _build_vw_grid(self) -> Array:
        env_params = getattr(self._env, "params", {}) or {}
        v_bins_raw = env_params.get("vw_grid_v", None)
        w_bins_raw = env_params.get("vw_grid_w", None)
        if v_bins_raw is None or w_bins_raw is None:
            raise ValueError("discrete_vw_grid requires env params 'vw_grid_v' and 'vw_grid_w'.")
        v_bins = np.asarray(v_bins_raw, dtype=np.float32).reshape(-1)
        w_bins = np.asarray(w_bins_raw, dtype=np.float32).reshape(-1)
        if v_bins.size == 0 or w_bins.size == 0:
            raise ValueError("vw grid bins must be non-empty.")
        if not np.all(np.isfinite(v_bins)) or not np.all(np.isfinite(w_bins)):
            raise ValueError("vw grid bins must be finite.")
        vv, ww = np.meshgrid(v_bins, w_bins, indexing="ij")
        # Public semantics are (v, w), while internal action order is [w, v].
        grid = np.stack([ww.reshape(-1), vv.reshape(-1)], axis=-1)
        return jnp.asarray(grid, dtype=jnp.float32)

    def _discretize_action(self, action: Action) -> Action:
        if not self._action_discrete or self._discrete_joint:
            return action
        step_w = float(getattr(self._env, "params", {}).get("a_w_step", 0.0))
        step_v = float(getattr(self._env, "params", {}).get("a_v_step", 0.0))
        if step_w <= 0.0:
            step_w = float(getattr(self._env, "params", {}).get("delta_w_step", 0.0))
        if step_v <= 0.0:
            step_v = float(getattr(self._env, "params", {}).get("delta_v_step", 0.0))
        if step_w <= 0.0 or step_v <= 0.0:
            return action
        w = action[:, 0]
        v = action[:, 1]
        w_disc = jnp.where(w > step_w / 2, step_w, jnp.where(w < -step_w / 2, -step_w, 0.0))
        v_disc = jnp.where(v > step_v / 2, step_v, jnp.where(v < -step_v / 2, -step_v, 0.0))
        return jnp.stack([w_disc, v_disc], axis=-1)

    def _index_to_action(self, idx: Array) -> Action:
        if self._discrete_joint:
            if self._discrete_action_grid is None:
                raise ValueError("Discrete action grid not initialized.")
            return self._discrete_action_grid[idx]
        if self._discrete_action_values is None:
            raise ValueError("Discrete action values not initialized.")
        w = self._discrete_action_values[0][idx[..., 0]]
        v = self._discrete_action_values[1][idx[..., 1]]
        return jnp.stack([w, v], axis=-1)

    def _action_to_index(self, action: Action) -> Array:
        if self._discrete_joint:
            if self._discrete_action_grid is None:
                raise ValueError("Discrete action grid not initialized.")
            action_flat = action.reshape((-1, action.shape[-1]))
            diff = action_flat[:, None, :] - self._discrete_action_grid[None, :, :]
            dist2 = jnp.sum(diff * diff, axis=-1)
            idx_flat = jnp.argmin(dist2, axis=-1)
            return idx_flat.reshape(action.shape[:-1])
        if self._discrete_action_values is None:
            raise ValueError("Discrete action values not initialized.")
        w_vals = self._discrete_action_values[0]
        v_vals = self._discrete_action_values[1]
        w_idx = jnp.argmin(jnp.abs(action[..., 0:1] - w_vals[None, :]), axis=-1)
        v_idx = jnp.argmin(jnp.abs(action[..., 1:2] - v_vals[None, :]), axis=-1)
        return jnp.stack([w_idx, v_idx], axis=-1)

    def _action_stats(self, actions: Action) -> dict:
        actions_np = np.asarray(actions)
        info = {
            "stats/action_w_mean": float(np.mean(actions_np[..., 0])),
            "stats/action_w_std": float(np.std(actions_np[..., 0])),
            "stats/action_v_mean": float(np.mean(actions_np[..., 1])),
            "stats/action_v_std": float(np.std(actions_np[..., 1])),
        }
        if not self._discrete_joint or self._discrete_action_grid is None:
            return info
        idx = np.asarray(self._action_to_index(jnp.asarray(actions_np))).reshape(-1).astype(np.int32)
        n_actions = int(self._discrete_action_grid.shape[0])
        counts = np.bincount(idx, minlength=n_actions).astype(np.float64)
        total = max(float(counts.sum()), 1.0)
        probs = counts / total
        info["stats/action_idx_mode_frac"] = float(np.max(probs))
        nz = probs > 0
        info["stats/action_idx_entropy"] = float(-(probs[nz] * np.log(probs[nz])).sum())
        topk = min(3, n_actions)
        top_idx = np.argsort(counts)[::-1][:topk]
        for i, idx_i in enumerate(top_idx, start=1):
            info[f"stats/action_idx_top{i}"] = float(idx_i)
            info[f"stats/action_idx_top{i}_frac"] = float(probs[idx_i])
        return info

    @property
    def config(self) -> dict:
        return {
            "action_mode": self._action_mode,
            "discrete_joint": self._discrete_joint,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_ratio": self.clip_ratio,
            "ppo_epochs": self.ppo_epochs,
            "minibatch_size": self.minibatch_size,
            "lr_actor": self.lr_actor,
            "lr_value": self.lr_value,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "vf_clip_range": self.vf_clip_range,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "gnn_layers": self.gnn_layers,
            "concat_robot_state": self.concat_robot_state,
            "use_gru": self.use_gru,
            "rnn_hidden_dim": self.rnn_hidden_dim,
            "rnn_seq_len": self.rnn_seq_len,
            "rnn_minibatch_chunks": self.rnn_minibatch_chunks,
            "n_discrete_actions": None if self._discrete_action_grid is None else int(self._discrete_action_grid.shape[0]),
            "use_hetero_attn": self.use_hetero_attn,
        }

    @property
    def actor_params(self) -> Params:
        return self.actor_train_state.params

    def _scale_action(self, action: Action) -> Action:
        return action * self._action_scale + self._action_bias

    def _unscale_action(self, action: Action) -> Action:
        scaled = (action - self._action_bias) / self._action_scale
        return jnp.clip(scaled, -0.999, 0.999)

    def init_actor_carry(self) -> Array:
        if not self.use_gru:
            return jnp.zeros((self.n_agents, 1), dtype=jnp.float32)
        return self.actor.init_carry(n_agents=self.n_agents)

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        if params is None:
            params = self.actor_params
        if self.use_gru:
            # Stateless fallback for callers that do not thread recurrent carry.
            carry = self.actor.init_carry(n_agents=self.n_agents)
            done_prev = jnp.array(False)
            raw_action, _ = self.actor.get_action_rnn(params, graph, carry, done_prev)
            if self._action_discrete:
                return self._index_to_action(raw_action)
            return self._discretize_action(self._scale_action(raw_action))
        raw_action = self.actor.get_action(params, graph)
        if self._action_discrete:
            return self._index_to_action(raw_action)
        action = self._scale_action(raw_action)
        return self._discretize_action(action)

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        if self.use_gru:
            # Stateless fallback for callers that do not thread recurrent carry.
            carry = self.actor.init_carry(n_agents=self.n_agents)
            done_prev = jnp.array(False)
            action, log_pi, _ = self.actor.sample_action_rnn(params, graph, key, carry, done_prev)
            log_pi = log_pi.sum(axis=-1)
            if self._action_discrete:
                return self._index_to_action(action), log_pi
            return self._scale_action(action), log_pi
        if self._action_discrete:
            raw_idx, log_pi = self.actor_train_state.apply_fn(params, graph, key)
            log_pi = log_pi.sum(axis=-1)
            return self._index_to_action(raw_idx), log_pi

        raw_action, log_pi = self.actor_train_state.apply_fn(params, graph, key)
        log_pi = log_pi.sum(axis=-1)
        return self._scale_action(raw_action), log_pi

    def act_rnn(
            self,
            graph: GraphsTuple,
            carry: Array,
            done_prev: Array,
            params: Optional[Params] = None,
    ) -> tuple[Action, Array]:
        if params is None:
            params = self.actor_params
        raw_action, new_carry = self.actor.get_action_rnn(params, graph, carry, done_prev)
        if self._action_discrete:
            return self._index_to_action(raw_action), new_carry
        action = self._discretize_action(self._scale_action(raw_action))
        return action, new_carry

    def step_rnn(
            self,
            graph: GraphsTuple,
            key: PRNGKey,
            carry: Array,
            done_prev: Array,
            params: Optional[Params] = None,
    ) -> tuple[Action, Array, Array]:
        if params is None:
            params = self.actor_params
        raw_action, log_pi, new_carry = self.actor.sample_action_rnn(params, graph, key, carry, done_prev)
        log_pi = log_pi.sum(axis=-1)
        if self._action_discrete:
            return self._index_to_action(raw_action), log_pi, new_carry
        action = self._scale_action(raw_action)
        return action, log_pi, new_carry

    def _chunk_rollout(self, rollout: Rollout) -> Rollout:
        n_env = rollout.rewards.shape[0]
        t_horizon = rollout.rewards.shape[1]
        seq_len = int(self.rnn_seq_len)
        if seq_len <= 0:
            raise ValueError(f"rnn_seq_len must be positive, got {seq_len}.")
        n_seq = t_horizon // seq_len
        if n_seq < 1:
            raise ValueError(
                f"rnn_seq_len={seq_len} is larger than rollout horizon={t_horizon}. "
                f"Please reduce --rnn-seq-len or increase --max-step."
            )
        t_use = n_seq * seq_len

        def chunk_leaf(x):
            if not hasattr(x, "shape"):
                return x
            if x.ndim < 2:
                return x
            if x.shape[0] != n_env or x.shape[1] != t_horizon:
                return x
            x = x[:, :t_use]
            x = x.reshape((n_env, n_seq, seq_len) + x.shape[2:])
            return merge01(x)

        return jtu.tree_map(chunk_leaf, rollout)

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_minibatch(
            self,
            actor_state: TrainState,
            value_state: TrainState,
            batch: PPOBatch,
            key: PRNGKey,
    ) -> tuple[TrainState, TrainState, dict]:
        def log_prob_single(actor_params: Params, graph: GraphsTuple, action: Action) -> Array:
            dist = self.actor.dist.apply(actor_params, graph, n_agents=self.n_agents)
            if self._action_discrete:
                action_idx = self._action_to_index(action)
                return dist.log_prob(action_idx).sum()
            raw_action = self._unscale_action(action)
            log_pi = dist.log_prob(raw_action)
            return log_pi.sum()

        def value_single(value_params: Params, graph: GraphsTuple) -> Array:
            return self.value.get_value(value_params, graph)

        def entropy_single(actor_params: Params, graph: GraphsTuple, key_: PRNGKey) -> Array:
            dist = self.actor.dist.apply(actor_params, graph, n_agents=self.n_agents)
            sample = dist.sample(seed=key_)
            log_pi = dist.log_prob(sample)
            return -log_pi.sum()

        def loss_fn(actor_params: Params, value_params: Params):
            log_pi = jax.vmap(ft.partial(log_prob_single, actor_params))(batch.graph, batch.actions)
            ratio = jnp.exp(log_pi - batch.log_pis)

            adv = batch.advantages
            unclipped = ratio * adv
            clipped = jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

            values = jax.vmap(ft.partial(value_single, value_params))(batch.graph)
            if self.vf_clip_range is not None:
                values_clipped = batch.old_values + jnp.clip(
                    values - batch.old_values,
                    -self.vf_clip_range,
                    self.vf_clip_range,
                )
                value_loss_unclipped = (batch.returns - values) ** 2
                value_loss_clipped = (batch.returns - values_clipped) ** 2
                value_loss = jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
            else:
                value_loss = jnp.mean((batch.returns - values) ** 2)

            entropy = jnp.array(0.0)
            if self.ent_coef > 0:
                ent_keys = jr.split(key, batch.actions.shape[0])
                entropy = jnp.mean(
                    jax.vmap(ft.partial(entropy_single, actor_params))(batch.graph, ent_keys)
                )

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            approx_kl = jnp.mean(batch.log_pis - log_pi)
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > self.clip_ratio).astype(jnp.float32))

            info = {
                "loss/policy": policy_loss,
                "loss/value": value_loss,
                "loss/entropy": entropy,
                "loss/total": loss,
                "stats/approx_kl": approx_kl,
                "stats/clipfrac": clipfrac,
            }
            return loss, info

        (loss, info), (grad_actor, grad_value) = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(actor_state.params, value_state.params)

        grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
        grad_value, grad_value_norm = compute_norm_and_clip(grad_value, self.max_grad_norm)

        actor_state = actor_state.apply_gradients(grads=grad_actor)
        value_state = value_state.apply_gradients(grads=grad_value)
        info = info | {"grad_norm/actor": grad_actor_norm, "grad_norm/value": grad_value_norm}
        return actor_state, value_state, info

    @ft.partial(jax.jit, static_argnums=(0,))
    def _update_recurrent(
            self,
            actor_state: TrainState,
            value_state: TrainState,
            rollout: Rollout,
            key: PRNGKey,
    ) -> tuple[TrainState, TrainState, dict]:
        done_prev = jnp.concatenate(
            [jnp.zeros_like(rollout.dones[:, :1], dtype=bool), rollout.dones[:, :-1]],
            axis=1,
        )

        def value_seq_pair(
                value_params: Params,
                graphs: GraphsTuple,
                next_graphs: GraphsTuple,
                done_prev_seq: Array,
                dones_seq: Array,
        ):
            def body(carry, inputs):
                graph_t, next_graph_t, done_prev_t, done_t = inputs
                value_t, carry_next = self.value.get_value_rnn(value_params, graph_t, carry, done_prev_t)
                # For bootstrap at t, next_state uses done_t as reset mask.
                next_value_t, _ = self.value.get_value_rnn(value_params, next_graph_t, carry_next, done_t)
                return carry_next, (value_t, next_value_t)

            init_carry = self.value.init_carry(n_agents=self.n_agents)
            (_, (vals, next_vals)) = jax.lax.scan(body, init_carry, (graphs, next_graphs, done_prev_seq, dones_seq))
            return vals, next_vals

        def eval_values(value_params: Params):
            values, next_values = jax.vmap(
                ft.partial(value_seq_pair, value_params),
                in_axes=(0, 0, 0, 0),
            )(rollout.graph, rollout.next_graph, done_prev, rollout.dones)
            return values, next_values

        # Keep targets fixed within this PPO epoch, matching feed-forward PPO behavior.
        values_old, next_values_old = eval_values(value_state.params)
        returns, advantages = compute_gae(
            values_old, rollout.rewards, rollout.dones, next_values_old, self.gamma, self.gae_lambda
        )

        def policy_eval_seq(actor_params: Params, graphs: GraphsTuple, actions: Action, done_prev_seq: Array, keys_seq: Array):
            def body(carry, inputs):
                graph_t, action_t, done_prev_t, key_t = inputs
                dist, carry_next = self.actor.dist_rnn.apply(
                    actor_params, graph_t, n_agents=self.n_agents, carry=carry, done_prev=done_prev_t
                )
                if self._action_discrete:
                    action_idx = self._action_to_index(action_t)
                    log_pi_t = dist.log_prob(action_idx).sum()
                else:
                    raw_action = self._unscale_action(action_t)
                    log_pi_t = dist.log_prob(raw_action).sum()
                sample = dist.sample(seed=key_t)
                entropy_t = -dist.log_prob(sample).sum()
                return carry_next, (log_pi_t, entropy_t)

            init_carry = self.actor.init_carry(n_agents=self.n_agents)
            _, (log_pis, entropies) = jax.lax.scan(body, init_carry, (graphs, actions, done_prev_seq, keys_seq))
            return log_pis, entropies

        ent_keys = jr.split(key, rollout.actions.shape[0] * rollout.actions.shape[1]).reshape(
            rollout.actions.shape[0], rollout.actions.shape[1], -1
        )

        def loss_fn(actor_params: Params, value_params: Params):
            values, _ = eval_values(value_params)
            log_pi_new, entropy = jax.vmap(
                ft.partial(policy_eval_seq, actor_params),
                in_axes=(0, 0, 0, 0),
            )(rollout.graph, rollout.actions, done_prev, ent_keys)

            ratio = jnp.exp(log_pi_new - rollout.log_pis)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            if self.vf_clip_range is not None:
                values_clipped = values_old + jnp.clip(
                    values - values_old,
                    -self.vf_clip_range,
                    self.vf_clip_range,
                )
                value_loss_unclipped = (returns - values) ** 2
                value_loss_clipped = (returns - values_clipped) ** 2
                value_loss = jnp.mean(jnp.maximum(value_loss_unclipped, value_loss_clipped))
            else:
                value_loss = jnp.mean((returns - values) ** 2)
            entropy_mean = jnp.mean(entropy)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

            approx_kl = jnp.mean(rollout.log_pis - log_pi_new)
            clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > self.clip_ratio).astype(jnp.float32))
            info = {
                "loss/policy": policy_loss,
                "loss/value": value_loss,
                "loss/entropy": entropy_mean,
                "loss/total": loss,
                "stats/approx_kl": approx_kl,
                "stats/clipfrac": clipfrac,
            }
            return loss, info

        (loss, info), (grad_actor, grad_value) = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(actor_state.params, value_state.params)

        grad_actor, grad_actor_norm = compute_norm_and_clip(grad_actor, self.max_grad_norm)
        grad_value, grad_value_norm = compute_norm_and_clip(grad_value, self.max_grad_norm)
        actor_state = actor_state.apply_gradients(grads=grad_actor)
        value_state = value_state.apply_gradients(grads=grad_value)
        info = info | {"grad_norm/actor": grad_actor_norm, "grad_norm/value": grad_value_norm}
        return actor_state, value_state, info

    def update(self, rollout: Rollout, step: int) -> dict:
        if self.use_gru:
            chunk_rollout = self._chunk_rollout(rollout)
            n_chunks = chunk_rollout.rewards.shape[0]
            mb_chunks = min(int(self.rnn_minibatch_chunks), int(n_chunks))
            if mb_chunks <= 0:
                raise ValueError(f"rnn_minibatch_chunks must be positive, got {self.rnn_minibatch_chunks}.")
            update_info = {}
            for _ in range(self.ppo_epochs):
                idx = self.rng.permutation(n_chunks)
                for start in range(0, n_chunks, mb_chunks):
                    mb_idx = idx[start: start + mb_chunks]
                    minibatch = jtu.tree_map(lambda x: x[mb_idx], chunk_rollout)
                    key, self.key = jr.split(self.key)
                    self.actor_train_state, self.value_train_state, update_info = self._update_recurrent(
                        self.actor_train_state, self.value_train_state, minibatch, key
                    )
            update_info = update_info | self._action_stats(rollout.actions)
            return update_info

        value_fn = jax_vmap(jax_vmap(ft.partial(self.value.get_value, self.value_train_state.params)))
        values = value_fn(rollout.graph)
        next_values = value_fn(rollout.next_graph)
        returns, advantages = compute_gae(
            values, rollout.rewards, rollout.dones, next_values, self.gamma, self.gae_lambda
        )

        batch = PPOBatch(rollout.graph, rollout.actions, rollout.log_pis, returns, advantages, values)
        batch = jtu.tree_map(merge01, batch)

        batch_size = batch.returns.shape[0]
        update_info = {}

        for _ in range(self.ppo_epochs):
            idx = self.rng.permutation(batch_size)
            for start in range(0, batch_size, self.minibatch_size):
                mb_idx = idx[start: start + self.minibatch_size]
                minibatch = jtu.tree_map(lambda x: x[mb_idx], batch)
                key, self.key = jr.split(self.key)
                self.actor_train_state, self.value_train_state, update_info = self._update_minibatch(
                    self.actor_train_state, self.value_train_state, minibatch, key
                )

        update_info = update_info | self._action_stats(rollout.actions)
        return update_info

    def save(self, save_dir: str, step: int):
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, "actor.pkl"), "wb"))
        pickle.dump(self.value_train_state.params, open(os.path.join(model_dir, "value.pkl"), "wb"))

    def load(self, load_dir: str, step: int):
        path = os.path.join(load_dir, str(step))
        self.actor_train_state = self.actor_train_state.replace(
            params=pickle.load(open(os.path.join(path, "actor.pkl"), "rb"))
        )
        self.value_train_state = self.value_train_state.replace(
            params=pickle.load(open(os.path.join(path, "value.pkl"), "rb"))
        )
