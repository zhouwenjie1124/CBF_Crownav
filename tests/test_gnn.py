"""Unit tests for the GNN module — shape checks and return_attn flag."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from higcbf.utils.graph import GraphsTuple
from higcbf.nn.gnn import GNN


def _make_dummy_graph(n_nodes: int = 6, node_dim: int = 8, edge_dim: int = 4):
    """Build a minimal fully-connected GraphsTuple for unit testing."""
    # Fully-connected edges (excluding self-loops)
    senders_list, receivers_list = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                senders_list.append(i)
                receivers_list.append(j)
    n_edges = len(senders_list)
    nodes = jnp.ones((n_nodes, node_dim))
    edges = jnp.ones((n_edges, edge_dim))
    senders = jnp.array(senders_list)
    receivers = jnp.array(receivers_list)
    # GraphsTuple(n_node, n_edge, nodes, edges, states, receivers, senders, node_type, env_states)
    graph = GraphsTuple(
        n_node=jnp.array(n_nodes),
        n_edge=jnp.array(n_edges),
        nodes=nodes,
        edges=edges,
        states=nodes,  # reuse node features as dummy states
        receivers=receivers,
        senders=senders,
        node_type=jnp.zeros(n_nodes, dtype=jnp.int32),
        env_states=None,
    )
    return graph


def _make_gnn(out_dim: int = 16, n_layers: int = 2):
    return GNN(
        msg_dim=32,
        hid_size_msg=(32,),
        hid_size_aggr=(32,),
        hid_size_update=(32,),
        out_dim=out_dim,
        n_layers=n_layers,
    )


class TestGNNShapes:
    def test_output_shape(self):
        graph = _make_dummy_graph(n_nodes=5, node_dim=8, edge_dim=4)
        gnn = _make_gnn(out_dim=16)
        key = jax.random.PRNGKey(0)
        params = gnn.init(key, graph)
        out = gnn.apply(params, graph)
        assert out.shape == (5, 16), f"unexpected shape {out.shape}"

    def test_single_layer(self):
        graph = _make_dummy_graph(n_nodes=3, node_dim=8, edge_dim=4)
        gnn = _make_gnn(out_dim=8, n_layers=1)
        key = jax.random.PRNGKey(1)
        params = gnn.init(key, graph)
        out = gnn.apply(params, graph)
        assert out.shape == (3, 8)

    def test_return_attn_tuple(self):
        graph = _make_dummy_graph(n_nodes=4, node_dim=8, edge_dim=4)
        gnn = _make_gnn(out_dim=16, n_layers=2)
        key = jax.random.PRNGKey(2)
        params = gnn.init(key, graph, return_attn=True)
        result = gnn.apply(params, graph, return_attn=True)
        assert isinstance(result, tuple), "return_attn=True should return a tuple"
        node_feats, attn_list = result
        assert node_feats.shape == (4, 16)
        assert isinstance(attn_list, list)

    def test_return_attn_false_is_array(self):
        graph = _make_dummy_graph(n_nodes=4, node_dim=8, edge_dim=4)
        gnn = _make_gnn(out_dim=16)
        key = jax.random.PRNGKey(3)
        params = gnn.init(key, graph)
        out = gnn.apply(params, graph, return_attn=False)
        assert isinstance(out, jnp.ndarray)

    def test_no_global_save_attn(self):
        """Make sure the old global save_attn variable was removed."""
        import higcbf.nn.gnn as gnn_module
        assert not hasattr(gnn_module, "save_attn"), "save_attn global should have been removed"
        assert not hasattr(gnn_module, "save_set_attn"), "save_set_attn should have been removed"
