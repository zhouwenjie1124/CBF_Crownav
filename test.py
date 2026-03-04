import argparse
import datetime
import functools as ft
import os
import pathlib
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml

try:
    import ipdb
except ImportError:  # pragma: no cover - optional debug dependency
    ipdb = None

from higcbf.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from higcbf.env import make_env
from higcbf.env.robot_ped_env import RobotPedEnv
from higcbf.env.base import RolloutResult
from higcbf.trainer.utils import get_bb_cbf
from higcbf.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from higcbf.utils.typing import Action, Array
from higcbf.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap, tree_concat_at_front


def _is_dubins_to_robotped_transfer(source_env_id: str, target_env_id: str) -> bool:
    return source_env_id == "DubinsCar" and target_env_id == "RobotPedEnv"


def _robotped_to_dubins_graph(
    robot_graph: GraphsTuple,
    n_rays: int,
    comm_radius: float,
) -> GraphsTuple:
    if n_rays <= 0:
        raise ValueError(f"n_rays must be positive for Dubins transfer, got {n_rays}.")

    # Read robot state (x, y, theta, v, w) and keep Dubins-compatible subset (x, y, theta, v).
    agent_full = robot_graph.type_states(type_idx=RobotPedEnv.AGENT, n_type=1)
    agent_state = jnp.concatenate([agent_full[:, :3], agent_full[:, 3:4]], axis=-1)
    goal_xy = robot_graph.env_states.goal[:, :2]
    goal_theta = jnp.arctan2(goal_xy[:, 1] - agent_state[:, 1], goal_xy[:, 0] - agent_state[:, 0])
    goal_state = jnp.concatenate([goal_xy, goal_theta[:, None], jnp.zeros((1, 1), dtype=agent_state.dtype)], axis=-1)

    # Use lidar nodes already built by RobotPedEnv as obstacle hits.
    lidar_full = robot_graph.type_states(type_idx=RobotPedEnv.OBS, n_type=n_rays)
    lidar_state = jnp.concatenate([lidar_full[:, :2], jnp.zeros((n_rays, 2), dtype=agent_state.dtype)], axis=-1)

    node_feats = jnp.zeros((2 + n_rays, 3), dtype=agent_state.dtype)
    node_feats = node_feats.at[0, 2].set(1.0)   # agent
    node_feats = node_feats.at[1, 1].set(1.0)   # goal
    node_feats = node_feats.at[2:, 0].set(1.0)  # obs

    node_type = jnp.zeros((2 + n_rays,), dtype=jnp.int32)
    node_type = node_type.at[1].set(1)  # goal
    node_type = node_type.at[2:].set(2)  # obs

    id_agent = jnp.array([0], dtype=jnp.int32)
    id_goal = jnp.array([1], dtype=jnp.int32)
    id_obs = jnp.arange(2, 2 + n_rays, dtype=jnp.int32)

    agent_pos = agent_state[:, :2]
    agent_v = jnp.concatenate(
        [
            (agent_state[:, 3] * jnp.cos(agent_state[:, 2]))[:, None],
            (agent_state[:, 3] * jnp.sin(agent_state[:, 2]))[:, None],
        ],
        axis=-1,
    )

    # agent-agent block (kept for shape/layout compatibility; masked out for single robot)
    pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
    dist = jnp.linalg.norm(pos_diff, axis=-1)
    dist = dist + jnp.eye(1) * (comm_radius + 1.0)
    v_diff = agent_v[:, None, :] - agent_v[None, :, :]
    agent_agent_feats = jnp.concatenate([pos_diff, v_diff], axis=-1)
    agent_agent_mask = dist < comm_radius
    agent_agent_edges = EdgeBlock(agent_agent_feats, agent_agent_mask, id_agent, id_agent)

    # agent-goal block
    goal_rel = agent_state[:, :2] - goal_state[:, :2]
    agent_goal_feats = jnp.concatenate([goal_rel, agent_v], axis=-1)
    goal_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
    safe_goal_norm = jnp.maximum(goal_norm, comm_radius)
    goal_coef = jnp.where(goal_norm > comm_radius, comm_radius / safe_goal_norm, 1.0)
    agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * goal_coef)
    agent_goal_edges = EdgeBlock(
        edge_feats=agent_goal_feats[None, :, :],
        edge_mask=jnp.ones((1, 1), dtype=bool),
        ids_recv=id_agent,
        ids_send=id_goal,
    )

    # agent-obstacle(lidar) block
    lidar_pos = agent_pos[0, :] - lidar_state[:, :2]
    lidar_feats = jnp.concatenate([agent_state[0, :2], agent_v[0]], axis=0)[None, :] - lidar_state
    lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
    active_lidar = lidar_dist < (comm_radius - 1e-1)
    agent_obs_edges = EdgeBlock(
        edge_feats=lidar_feats[None, :, :],
        edge_mask=active_lidar[None, :],
        ids_recv=id_agent,
        ids_send=id_obs,
    )

    return GetGraph(
        nodes=node_feats,
        node_type=node_type,
        edge_blocks=[agent_agent_edges, agent_goal_edges, agent_obs_edges],
        env_states=robot_graph.env_states,
        states=jnp.concatenate([agent_state, goal_state, lidar_state], axis=0),
    ).to_padded()


def _map_dubins_action_to_robotped(env, robot_graph: GraphsTuple, dubins_action: Array) -> Array:
    mode = env.params.get("action_mode", "discrete_delta")
    if mode in {"continuous_vw", "discrete_vw_grid"}:
        # Dubins output is [omega, a_v]; convert to RobotPed absolute [w_cmd, v_cmd].
        robot_state = robot_graph.type_states(type_idx=RobotPedEnv.AGENT, n_type=1)
        v_now = robot_state[:, 3]
        w_cmd = dubins_action[:, 0]
        v_cmd = v_now + dubins_action[:, 1] * env.dt
        action = jnp.stack([w_cmd, v_cmd], axis=-1)
    else:
        # RobotPed default expects [a_w, a_v]; keep the same order as Dubins [omega, a_v].
        action = dubins_action
    return env.clip_action(action)


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
    else:
        config = None

    # create environments
    if config is not None:
        num_agents = config.num_agents if args.num_agents is None else args.num_agents
        area_size = config.area_size if args.area_size is None else args.area_size
        num_obs = getattr(config, "obs", None) if args.obs is None else args.obs
        max_step = getattr(config, "max_step", None) if args.max_step is None else args.max_step
        max_travel = getattr(config, "max_travel", None) if args.max_travel is None else args.max_travel
        n_rays = getattr(config, "n_rays", None) if args.n_rays is None else args.n_rays
        num_ped = getattr(config, "n_ped", None) if getattr(args, "num_ped", None) is None else args.num_ped
        env_params = getattr(config, "env_params", None)
        if not isinstance(env_params, dict):
            env_params = None
    else:
        num_agents = args.num_agents
        area_size = args.area_size
        num_obs = args.obs
        max_step = args.max_step
        max_travel = args.max_travel
        n_rays = args.n_rays
        num_ped = getattr(args, "num_ped", None)
        env_params = None

    source_env_id = config.env if config is not None else args.env
    target_env_id = source_env_id if args.env is None else args.env
    use_dubins_robotped_transfer = _is_dubins_to_robotped_transfer(source_env_id, target_env_id)
    if use_dubins_robotped_transfer and args.num_agents is None:
        num_agents = 1
        print("Dubins->RobotPed transfer: auto-set num_agents=1 for RobotPedEnv.")

    if config is None and args.env is None:
        raise ValueError("--env is required when no --path config is provided.")
    if area_size is None:
        raise ValueError("--area-size is required when no --path config is provided.")
    if num_agents is None:
        raise ValueError("--num-agents is required when no --path config is provided.")

    env = make_env(
        env_id=target_env_id,
        num_agents=num_agents,
        num_obs=num_obs,
        area_size=area_size,
        max_step=max_step,
        max_travel=max_travel,
        n_rays=n_rays,
        num_ped=num_ped,
        env_params=env_params,
    )
    if args.continue_after_collision and hasattr(env, "params") and "terminate_on_collision" in env.params:
        env.params["terminate_on_collision"] = False
        print("Collision termination disabled for test rollout (continue_after_collision=True).")

    transfer_policy_cfg = None
    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)

            policy_env = env
            if use_dubins_robotped_transfer:
                source_env_params = getattr(config, "env_params", None)
                if not isinstance(source_env_params, dict):
                    source_env_params = None
                policy_env = make_env(
                    env_id=source_env_id,
                    num_agents=1,
                    num_obs=getattr(config, "obs", None),
                    area_size=area_size,
                    max_step=max_step,
                    max_travel=max_travel,
                    n_rays=getattr(config, "n_rays", None),
                    env_params=source_env_params,
                )
                transfer_policy_cfg = {
                    "n_rays": int(policy_env.params["n_rays"]),
                    "comm_radius": float(policy_env.params["comm_radius"]),
                }
                print(
                    "Transfer mode enabled: pretrained DubinsCar policy graph/action "
                    "adapted for RobotPedEnv evaluation."
                )

            algo_kwargs = {
                "algo": config.algo,
                "env": policy_env,
                "node_dim": policy_env.node_dim,
                "edge_dim": policy_env.edge_dim,
                "state_dim": policy_env.state_dim,
                "action_dim": policy_env.action_dim,
                "n_agents": policy_env.num_agents,
            }
            if config.algo == "ppo":
                algo_kwargs.update({
                    "gnn_layers": getattr(config, "gnn_layers", 1),
                    "gamma": getattr(config, "gamma", 0.99),
                    "gae_lambda": getattr(config, "gae_lambda", 0.95),
                    "clip_ratio": getattr(config, "clip_ratio", 0.2),
                    "ppo_epochs": getattr(config, "ppo_epochs", 10),
                    "minibatch_size": getattr(config, "minibatch_size", 256),
                    "lr_actor": getattr(config, "lr_actor", 3e-4),
                    "lr_value": getattr(config, "lr_value", 1e-3),
                    "ent_coef": getattr(config, "ent_coef", 0.0),
                    "vf_coef": getattr(config, "vf_coef", 0.5),
                    "vf_clip_range": getattr(config, "vf_clip_range", None),
                    "max_grad_norm": getattr(config, "max_grad_norm", 0.5),
                    "seed": getattr(config, "seed", 0),
                    "concat_robot_state": getattr(config, "concat_robot_state", False),
                    "use_gru": getattr(config, "use_gru", False),
                    "rnn_hidden_dim": getattr(config, "rnn_hidden_dim", 64),
                    "rnn_seq_len": getattr(config, "rnn_seq_len", 32),
                    "rnn_minibatch_chunks": getattr(config, "rnn_minibatch_chunks", 64),
                })
            elif config.algo in {"centralized_cbf", "dec_share_cbf"}:
                algo_kwargs.update({
                    "alpha": getattr(config, "alpha", 1.0),
                })
            else:
                algo_kwargs.update({
                    "gnn_layers": getattr(config, "gnn_layers", 1),
                    "batch_size": getattr(config, "batch_size", 256),
                    "buffer_size": getattr(config, "buffer_size", 512),
                    "horizon": getattr(config, "horizon", 32),
                    "lr_actor": getattr(config, "lr_actor", 3e-5),
                    "lr_cbf": getattr(config, "lr_cbf", 3e-5),
                    "alpha": getattr(config, "alpha", 1.0),
                    "eps": getattr(config, "eps", 0.02),
                    "inner_epoch": getattr(config, "inner_epoch", 8),
                    "loss_action_coef": getattr(config, "loss_action_coef", 0.0001),
                    "loss_unsafe_coef": getattr(config, "loss_unsafe_coef", 1.0),
                    "loss_safe_coef": getattr(config, "loss_safe_coef", 1.0),
                    "loss_h_dot_coef": getattr(config, "loss_h_dot_coef", 0.01),
                    "max_grad_norm": getattr(config, "max_grad_norm", 2.0),
                    "seed": getattr(config, "seed", 0),
                })
            algo = make_algo(**algo_kwargs)
            algo.load(model_path, step)
            if transfer_policy_cfg is None:
                act_fn = jax.jit(algo.act)
            else:
                transfer_n_rays = transfer_policy_cfg["n_rays"]
                transfer_comm_radius = transfer_policy_cfg["comm_radius"]

                def act_fn_transfer(robot_graph: GraphsTuple) -> Action:
                    dubins_graph = _robotped_to_dubins_graph(
                        robot_graph,
                        n_rays=transfer_n_rays,
                        comm_radius=transfer_comm_radius,
                    )
                    dubins_action = algo.act(dubins_graph)
                    return _map_dubins_action_to_robotped(env, robot_graph, dubins_action)

                act_fn = jax.jit(act_fn_transfer)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        act_fn = jax.jit(env.u_ref)
        step = 0

    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    if transfer_policy_cfg is not None and args.cbf is not None:
        print("Warning: --cbf is disabled in Dubins->RobotPed transfer mode.")
        args.cbf = None

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
        get_bb_cbf_fn = None
        cbf_fn = None

    use_recurrent_rollout = bool(algo is not None and getattr(algo, "use_gru", False) and hasattr(algo, "act_rnn"))
    use_nojit_rollout = bool(args.nojit_rollout and not use_recurrent_rollout)

    if use_recurrent_rollout:
        rollout_length = args.max_step if args.max_step is not None else env.max_episode_steps

        def rollout_fn_rnn(key: Array) -> RolloutResult:
            graph0 = env.reset(key)

            def body(carry, _):
                graph, done_prev, actor_carry = carry
                action, next_actor_carry = algo.act_rnn(graph, actor_carry, done_prev)
                graph_new, reward, cost, done, info = env.step(graph, action, get_eval_info=True)
                done_next = jnp.logical_or(done_prev, done)
                reward = jnp.where(done_prev, jnp.zeros_like(reward), reward)
                cost = jnp.where(done_prev, jnp.zeros_like(cost), cost)
                done = jnp.where(done_prev, jnp.ones_like(done, dtype=bool), done)
                graph_new = jtu.tree_map(lambda new, old: jnp.where(done_prev, old, new), graph_new, graph)
                next_actor_carry = jtu.tree_map(
                    lambda new, old: jnp.where(done_prev[..., None], old, new),
                    next_actor_carry,
                    actor_carry,
                )
                return (graph_new, done_next, next_actor_carry), (graph_new, action, reward, cost, done, info)

            init_carry = (graph0, jnp.array(False), algo.init_actor_carry())
            (_, _, _), (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(
                body, init_carry, None, length=rollout_length
            )
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)
            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        if args.nojit_rollout:
            print("GRU model detected; nojit rollout disabled to preserve recurrent state. Using jit rollout.")
        else:
            print("jit rollout (recurrent GRU)!")
        rollout_fn = jax_jit_np(rollout_fn_rnn)
        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    elif use_nojit_rollout:
        print("Only jit step, no jit rollout!")
        rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=args.no_video)

        is_unsafe_fn = None
        is_finish_fn = None
    else:
        print("jit rollout!")
        rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step))

        is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
        is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

    rewards = []
    costs = []
    path_lengths = []
    nav_times = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    cbfs = []

    def truncate_on_finish(
        rollout: RolloutResult,
        is_unsafe: Array,
        is_finish: Array,
    ) -> tuple[RolloutResult, Array, Array]:
        if is_finish is None:
            return rollout, is_unsafe, is_finish
        finish = np.array(is_finish)
        if finish.ndim == 1:
            finish = finish[:, None]
        has_finish = finish.any(axis=0)
        if not has_finish.any():
            return rollout, is_unsafe, is_finish
        first_finish = np.argmax(finish, axis=0)
        t_end = int(first_finish[has_finish].min())
        # t_end is an index into Tp1_graph; slice T_* to length t_end.
        if t_end >= len(rollout.T_reward):
            return rollout, is_unsafe, is_finish

        Tp1_graph = jtu.tree_map(lambda x: x[: t_end + 1], rollout.Tp1_graph)
        T_action = rollout.T_action[:t_end]
        T_reward = rollout.T_reward[:t_end]
        T_cost = rollout.T_cost[:t_end]
        T_done = rollout.T_done[:t_end]
        T_info = jtu.tree_map(lambda x: x[:t_end], rollout.T_info)
        rollout = RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        is_unsafe = None if is_unsafe is None else np.array(is_unsafe)[: t_end + 1]
        is_finish = np.array(is_finish)[: t_end + 1]
        return rollout, is_unsafe, is_finish
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)

        if use_nojit_rollout:
            rollout: RolloutResult
            rollout, is_unsafe, is_finish = rollout_fn(key_x0)
            # if not jnp.isnan(rollout.T_reward).any():
            rollout, is_unsafe, is_finish = truncate_on_finish(rollout, is_unsafe, is_finish)
            is_unsafes.append(is_unsafe)
            is_finishes.append(is_finish)
        else:
            rollout: RolloutResult = rollout_fn(key_x0)
            # if not jnp.isnan(rollout.T_reward).any():
            is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
            is_finish = is_finish_fn(rollout.Tp1_graph)
            rollout, is_unsafe, is_finish = truncate_on_finish(rollout, is_unsafe, is_finish)
            is_unsafes.append(is_unsafe)
            is_finishes.append(is_finish)

        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)

        epi_path_len = None
        epi_nav_time = None
        if rollout.Tp1_graph is not None:
            states = np.array(rollout.Tp1_graph.states)
            node_type = np.array(rollout.Tp1_graph.node_type)
            if node_type.ndim > 1:
                node_type = node_type[0]
            agent_idx = np.where(node_type == 0)[0]
            if agent_idx.size > 0:
                pos = states[:, agent_idx, :2]
                step_dist = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)  # (T, n_agent)
                if len(is_finishes) > 0:
                    finish = np.array(is_finishes[-1])  # (T+1, n_agent)
                    finish_idx = np.argmax(finish, axis=0)
                    has_finish = finish.any(axis=0)
                else:
                    finish_idx = np.zeros((agent_idx.size,), dtype=np.int32)
                    has_finish = np.zeros((agent_idx.size,), dtype=bool)
                T = step_dist.shape[0]
                finish_steps = np.where(has_finish, finish_idx, T)
                mask = (np.arange(T)[:, None] < finish_steps[None, :])
                epi_path_len = (step_dist * mask).sum(axis=0).mean()
                epi_nav_time = (finish_steps * env.dt).mean()
                path_lengths.append(epi_path_len)
                nav_times.append(epi_nav_time)

        if args.cbf is not None:
            cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
        else:
            cbfs.append(None)
        if len(is_unsafes) == 0:
            continue
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
        extra = ""
        if epi_path_len is not None and epi_nav_time is not None:
            extra = f", path_len: {epi_path_len:.3f}, nav_time: {epi_nav_time:.3f}s"
        print(
            f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
            f"safe rate: {safe_rate * 100:.3f}%,"
            f"finish rate: {finish_rate * 100:.3f}%, "
            f"success rate: {success_rate * 100:.3f}%{extra}"
        )

        rates.append(np.array([safe_rate, finish_rate, success_rate]))
    is_unsafe = np.stack([np.max(np.asarray(u), axis=0) for u in is_unsafes])
    is_finish = np.stack([np.max(np.asarray(f), axis=0) for f in is_finishes])

    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    success_mean, success_std = ((1 - is_unsafe) * is_finish).mean(), ((1 - is_unsafe) * is_finish).std()

    summary_extra = ""
    if len(path_lengths) > 0 and len(nav_times) > 0:
        summary_extra = (
            f", path_len: {np.mean(path_lengths):.3f} (min/max {np.min(path_lengths):.3f}/{np.max(path_lengths):.3f})"
            f", nav_time: {np.mean(nav_times):.3f}s (min/max {np.min(nav_times):.3f}/{np.max(nav_times):.3f})"
        )
    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%, "
        f"finish_rate: {finish_mean * 100:.3f}%, "
        f"success_rate: {success_mean * 100:.3f}%"
        f"{summary_extra}"
    )

    # save results
    if args.log:
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env.area_size},{env.params['n_obs']},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                    f"{finish_mean * 100:.3f},{finish_std * 100:.3f},"
                    f"{success_mean * 100:.3f},{success_std * 100:.3f}\n")

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe, cbf) in enumerate(zip(rollouts, is_unsafes, cbfs)):
        if algo_is_cbf:
            safe_rate, finish_rate, success_rate = rates[ii] * 100
            video_name = f"n{num_agents}_epi{ii:02}_sr{safe_rate:.0f}_fr{finish_rate:.0f}_sr{success_rate:.0f}"
        else:
            video_name = f"n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"

        viz_opts = {}
        if args.cbf is not None:
            video_name += f"_cbf{args.cbf}"
            viz_opts["cbf"] = [*cbf, args.cbf]
        if env.__class__.__name__ == "RobotPedEnv":
            viz_opts["heading"] = {
                "agent_ids": [0],
                "state_idx": 2,
                "length": float(env.params.get("car_radius", 0.2)) * 2.5,
                "color": "#111111",
            }

        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)


def main():
    # Running Example: python test.py --path "pretrained/DubinsCar/gcbf+" --epi 5 --area-size 4 -n 16 --obs 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--area-size", type=float, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5) #  number of episodes to test
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument(
        "--continue-after-collision",
        action="store_true",
        default=False,
        help="Test only: do not terminate episode on collision (training behavior unchanged).",
    )
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    if ipdb is not None:
        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
