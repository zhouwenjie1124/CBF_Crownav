import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp

from typing import Type, Tuple
from abc import ABC, abstractproperty, abstractmethod

from .distribution import TanhTransformedDistribution, tfd
from ...utils.typing import Action, Array
from ...utils.graph import GraphsTuple
from ...nn.utils import default_nn_init, scaled_init
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...utils.typing import PRNGKey, Params


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tfd.Distribution:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass


class TanhNormal(PolicyDistribution):
    base_cls: Type[GNN]
    _nu: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5
    concat_robot_state: bool = False

    @property
    def std_dev_init_inv(self):
        # inverse of log(sum(exp())).
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)
        # x = x.nodes
        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(256, kernel_init=scaler_init, name="ScaleHid")(x)

        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min

        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)

    @property
    def nu(self):
        return self._nu


class CategoricalDiscrete(PolicyDistribution):
    base_cls: Type[GNN]
    _nu: int
    n_bins: int = 3
    concat_robot_state: bool = False

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)
        logits = nn.Dense(self._nu * self.n_bins, kernel_init=default_nn_init(), name="OutputDenseLogits")(x)
        logits = logits.reshape((logits.shape[0], self._nu, self.n_bins))
        dist = tfd.Categorical(logits=logits)
        return tfd.Independent(dist, reinterpreted_batch_ndims=1)

    @property
    def nu(self):
        return self._nu


class CategoricalJointDiscrete(PolicyDistribution):
    base_cls: Type[GNN]
    _nu: int
    concat_robot_state: bool = False

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)
        logits = nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDenseLogits")(x)
        return tfd.Categorical(logits=logits)

    @property
    def nu(self):
        return self._nu


class TanhNormalRNN(nn.Module):
    base_cls: Type[GNN]
    _nu: int
    hidden_dim: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5
    concat_robot_state: bool = False

    @property
    def std_dev_init_inv(self):
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(
            self,
            obs: GraphsTuple,
            n_agents: int,
            carry: Array,
            done_prev: Array,
            *args,
            **kwargs,
    ) -> Tuple[tfd.Distribution, Array]:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)

        done_prev = jnp.asarray(done_prev, dtype=bool)
        reset_mask = done_prev if done_prev.ndim > 0 else done_prev[None]
        carry = jnp.where(reset_mask[:, None], jnp.zeros_like(carry), carry)

        gru = nn.GRUCell(features=self.hidden_dim, name="GRU")
        carry, gru_out = gru(carry, x)

        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(256, kernel_init=scaler_init, name="ScaleHid")(gru_out)
        means = nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min

        distribution = tfd.Normal(loc=means, scale=stds)
        dist = tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)
        return dist, carry


class CategoricalDiscreteRNN(nn.Module):
    base_cls: Type[GNN]
    _nu: int
    hidden_dim: int
    n_bins: int = 3
    concat_robot_state: bool = False

    @nn.compact
    def __call__(
            self,
            obs: GraphsTuple,
            n_agents: int,
            carry: Array,
            done_prev: Array,
            *args,
            **kwargs,
    ) -> Tuple[tfd.Distribution, Array]:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)

        done_prev = jnp.asarray(done_prev, dtype=bool)
        reset_mask = done_prev if done_prev.ndim > 0 else done_prev[None]
        carry = jnp.where(reset_mask[:, None], jnp.zeros_like(carry), carry)

        gru = nn.GRUCell(features=self.hidden_dim, name="GRU")
        carry, gru_out = gru(carry, x)
        logits = nn.Dense(self._nu * self.n_bins, kernel_init=default_nn_init(), name="OutputDenseLogits")(gru_out)
        logits = logits.reshape((logits.shape[0], self._nu, self.n_bins))
        dist = tfd.Categorical(logits=logits)
        dist = tfd.Independent(dist, reinterpreted_batch_ndims=1)
        return dist, carry


class CategoricalJointDiscreteRNN(nn.Module):
    base_cls: Type[GNN]
    _nu: int
    hidden_dim: int
    concat_robot_state: bool = False

    @nn.compact
    def __call__(
            self,
            obs: GraphsTuple,
            n_agents: int,
            carry: Array,
            done_prev: Array,
            *args,
            **kwargs,
    ) -> Tuple[tfd.Distribution, Array]:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)

        done_prev = jnp.asarray(done_prev, dtype=bool)
        reset_mask = done_prev if done_prev.ndim > 0 else done_prev[None]
        carry = jnp.where(reset_mask[:, None], jnp.zeros_like(carry), carry)

        gru = nn.GRUCell(features=self.hidden_dim, name="GRU")
        carry, gru_out = gru(carry, x)
        logits = nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDenseLogits")(gru_out)
        dist = tfd.Categorical(logits=logits)
        return dist, carry


class Deterministic(nn.Module):
    base_cls: Type[GNN]
    head_cls: Type[nn.Module]
    _nu: int
    concat_robot_state: bool = False

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Action:
        x = self.base_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PrePolicyHeadLN")(x)
        x = self.head_cls()(x)
        x = nn.tanh(nn.Dense(self._nu, kernel_init=default_nn_init(), name="OutputDense")(x))
        return x


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        pass

    @abstractmethod
    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        pass


class DeterministicPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
            concat_robot_state: bool = False,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.policy_base = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.policy_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='PolicyHead'
        )
        self.net = Deterministic(
            base_cls=self.policy_base,
            head_cls=self.policy_head,
            _nu=action_dim,
            concat_robot_state=concat_robot_state,
        )
        self.std = 0.1

    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        return self.net.apply(params, obs, self.n_agents)

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        action = self.get_action(params, obs)
        log_pi = jnp.zeros_like(action)
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        raise NotImplementedError


class PPOPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            gnn_layers: int = 1,
            discrete_action: bool = False,
            action_bins: int = 3,
            discrete_joint: bool = False,
            n_discrete_actions: int | None = None,
            concat_robot_state: bool = False,
            use_gru: bool = False,
            rnn_hidden_dim: int = 64,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.use_gru = use_gru
        self.rnn_hidden_dim = rnn_hidden_dim
        self.dist_base = ft.partial(
            GNN,
            msg_dim=64,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=64,
            n_layers=gnn_layers
        )
        if discrete_action and use_gru:
            if discrete_joint:
                if n_discrete_actions is None or n_discrete_actions <= 0:
                    raise ValueError(f"n_discrete_actions must be positive, got {n_discrete_actions}.")
                self.dist_rnn = CategoricalJointDiscreteRNN(
                    base_cls=self.dist_base,
                    _nu=n_discrete_actions,
                    hidden_dim=rnn_hidden_dim,
                    concat_robot_state=concat_robot_state,
                )
            else:
                self.dist_rnn = CategoricalDiscreteRNN(
                    base_cls=self.dist_base,
                    _nu=action_dim,
                    hidden_dim=rnn_hidden_dim,
                    n_bins=action_bins,
                    concat_robot_state=concat_robot_state,
                )
            self.dist = None
        elif discrete_action:
            if discrete_joint:
                if n_discrete_actions is None or n_discrete_actions <= 0:
                    raise ValueError(f"n_discrete_actions must be positive, got {n_discrete_actions}.")
                self.dist = CategoricalJointDiscrete(
                    base_cls=self.dist_base,
                    _nu=n_discrete_actions,
                    concat_robot_state=concat_robot_state,
                )
            else:
                self.dist = CategoricalDiscrete(
                    base_cls=self.dist_base,
                    _nu=action_dim,
                    n_bins=action_bins,
                    concat_robot_state=concat_robot_state,
                )
            self.dist_rnn = None
        elif use_gru:
            self.dist_rnn = TanhNormalRNN(
                base_cls=self.dist_base,
                _nu=action_dim,
                hidden_dim=rnn_hidden_dim,
                concat_robot_state=concat_robot_state,
            )
            self.dist = None
        else:
            self.dist = TanhNormal(
                base_cls=self.dist_base,
                _nu=action_dim,
                concat_robot_state=concat_robot_state,
            )
            self.dist_rnn = None

    def init_carry(self, n_agents: int | None = None) -> Array:
        if n_agents is None:
            n_agents = self.n_agents
        return jnp.zeros((n_agents, self.rnn_hidden_dim), dtype=jnp.float32)

    def get_action(self, params: Params, obs: GraphsTuple) -> Action:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents)
        action = dist.mode()
        return action

    def sample_action(self, params: Params, obs: GraphsTuple, key: PRNGKey) -> Tuple[Action, Array]:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents)
        action = dist.sample(seed=key)
        log_pi = dist.log_prob(action)
        return action, log_pi

    def eval_action(self, params: Params, obs: GraphsTuple, action: Action, key: PRNGKey) -> Tuple[Array, Array]:
        dist = self.dist.apply(params, obs, n_agents=self.n_agents)
        log_pi = dist.log_prob(action)
        entropy = dist.entropy(seed=key)
        return log_pi, entropy

    def get_action_rnn(
            self, params: Params, obs: GraphsTuple, carry: Array, done_prev: Array
    ) -> Tuple[Action, Array]:
        dist, new_carry = self.dist_rnn.apply(params, obs, n_agents=self.n_agents, carry=carry, done_prev=done_prev)
        action = dist.mode()
        return action, new_carry

    def sample_action_rnn(
            self, params: Params, obs: GraphsTuple, key: PRNGKey, carry: Array, done_prev: Array
    ) -> Tuple[Action, Array, Array]:
        dist, new_carry = self.dist_rnn.apply(params, obs, n_agents=self.n_agents, carry=carry, done_prev=done_prev)
        action = dist.sample(seed=key)
        log_pi = dist.log_prob(action)
        return action, log_pi, new_carry

    def eval_action_rnn(
            self,
            params: Params,
            obs: GraphsTuple,
            action: Action,
            key: PRNGKey,
            carry: Array,
            done_prev: Array,
    ) -> Tuple[Array, Array, Array]:
        dist, new_carry = self.dist_rnn.apply(params, obs, n_agents=self.n_agents, carry=carry, done_prev=done_prev)
        log_pi = dist.log_prob(action)
        entropy = dist.entropy(seed=key)
        return log_pi, entropy, new_carry
