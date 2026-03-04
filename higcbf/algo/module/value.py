import functools as ft
import flax.linen as nn
import jax
import jax.numpy as jnp

from typing import Type

from ...nn.mlp import MLP
from ...nn.gnn import GNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params
from ...utils.graph import GraphsTuple


class StateFn(nn.Module):
    gnn_cls: Type[GNN]
    aggr_cls: Type[nn.Module]
    head_cls: Type[nn.Module]
    concat_robot_state: bool = False

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:
        # get node features
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PreValueHeadLN")(x)

        # aggregate information using attention
        gate_feats = self.aggr_cls()(x)
        gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
        attn = jax.nn.softmax(gate_feats, axis=-1)
        x = jnp.sum(attn[:, None] * x, axis=0)

        # pass through head class
        x = self.head_cls()(x)
        x = nn.Dense(1, kernel_init=default_nn_init())(x)

        return x


class StateFnRNN(nn.Module):
    gnn_cls: Type[GNN]
    aggr_cls: Type[nn.Module]
    head_cls: Type[nn.Module]
    hidden_dim: int
    concat_robot_state: bool = False

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, carry: Array, done_prev: Array, *args, **kwargs) -> tuple[Array, Array]:
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        if self.concat_robot_state:
            agent_states = obs.type_states(0, n_agents)
            x = jnp.concatenate([x, agent_states], axis=-1)
        x = nn.LayerNorm(name="PreValueHeadLN")(x)

        done_prev = jnp.asarray(done_prev, dtype=bool)
        reset_mask = done_prev if done_prev.ndim > 0 else done_prev[None]
        carry = jnp.where(reset_mask[:, None], jnp.zeros_like(carry), carry)

        gru = nn.GRUCell(features=self.hidden_dim, name="GRU")
        carry, gru_out = gru(carry, x)

        gate_feats = self.aggr_cls()(gru_out)
        gate_feats = nn.Dense(1, kernel_init=default_nn_init())(gate_feats).squeeze(-1)
        attn = jax.nn.softmax(gate_feats, axis=-1)
        x = jnp.sum(attn[:, None] * gru_out, axis=0)

        x = self.head_cls()(x)
        x = nn.Dense(1, kernel_init=default_nn_init())(x)
        return x, carry


class ValueNet:

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            gnn_layers: int = 1,
            concat_robot_state: bool = False,
            use_gru: bool = False,
            rnn_hidden_dim: int = 64,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.use_gru = use_gru
        self.rnn_hidden_dim = rnn_hidden_dim
        self.value_gnn = ft.partial(
            GNN,
            msg_dim=64,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=64,
            n_layers=gnn_layers
        )
        self.value_attn = ft.partial(
            MLP,
            hid_sizes=(128, 128),
            act=nn.relu,
            act_final=False,
            name='ValueAttn'
        )
        self.value_head = ft.partial(
            MLP,
            hid_sizes=(128, 128),
            act=nn.relu,
            act_final=False,
            name='ValueHead'
        )
        if use_gru:
            self.net_rnn = StateFnRNN(
                gnn_cls=self.value_gnn,
                aggr_cls=self.value_attn,
                head_cls=self.value_head,
                hidden_dim=rnn_hidden_dim,
                concat_robot_state=concat_robot_state,
            )
            self.net = None
        else:
            self.net = StateFn(
                gnn_cls=self.value_gnn,
                aggr_cls=self.value_attn,
                head_cls=self.value_head,
                concat_robot_state=concat_robot_state,
            )
            self.net_rnn = None

    def get_value(self, params: Params, obs: GraphsTuple) -> Array:
        values = self.net.apply(params, obs, self.n_agents)
        return values.squeeze()

    def init_carry(self, n_agents: int | None = None) -> Array:
        if n_agents is None:
            n_agents = self.n_agents
        return jnp.zeros((n_agents, self.rnn_hidden_dim), dtype=jnp.float32)

    def get_value_rnn(
            self,
            params: Params,
            obs: GraphsTuple,
            carry: Array,
            done_prev: Array,
    ) -> tuple[Array, Array]:
        values, new_carry = self.net_rnn.apply(params, obs, self.n_agents, carry, done_prev)
        return values.squeeze(), new_carry
