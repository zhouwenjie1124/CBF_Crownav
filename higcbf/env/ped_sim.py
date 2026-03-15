import jax
import jax.numpy as jnp
import numpy as np

from ..utils.graph import GraphsTuple
from ..utils.typing import Array, State
from .ped_env_wrapper import PedGraphEnvState
from .utils import get_lidar
class PedCBFController:
    """Lazy DecShareCBF wrapper for pedestrians (no policy network)."""

    def __init__(self, *args, **kwargs):
        from ..algo.dec_share_cbf import DecShareCBF

        self._inner = DecShareCBF(*args, **kwargs)
        self._env = self._inner._env

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3):
        return self._inner.get_qp_action(graph, relax_penalty=relax_penalty)


def build_ped_graph(
    ped_states: Array,      # (n_ped, 4)          pedestrian states [x, y, vx, vy]
    robot_state: Array,     # (1, 4)              robot state in double-integrator coords [x, y, vx, vy]
    robot_lidar: Array,     # (n_rays, 2)         robot lidar hit positions [x, y]
    ped_goal: Array,        # (n_ped, 2)          goal position for each pedestrian [x, y]
    ped_lidar: Array,       # (n_ped, n_rays, 2)  lidar hit positions for each pedestrian [x, y]
    ped_speed_pref: Array,  # (n_ped,)            preferred speed scalar for each pedestrian
) -> GraphsTuple:
    """
    Pack raw pedestrian simulation states into a GraphsTuple for DecShareCBF.

    Node layout (n_nodes = n_ped + 1 + n_rays total nodes):
        [0 : n_ped]          — pedestrian agent nodes,  node_type=0
        [n_ped]              — robot node,              node_type=1 (obstacle)
        [n_ped+1 : n_nodes]  — robot lidar hit nodes,  node_type=1 (obstacle)

    states tensor (n_nodes, 4):
        pedestrian nodes : [x, y, vx, vy]  — copied directly from ped_states
        robot node       : [x, y, vx, vy]  — converted from robot_state
        lidar nodes      : [x, y,  0,  0]  — hit position, velocity padded with zeros

    Edges (n_nodes edges, star topology with node 0 as the shared sender):
        edge[i] = states[i] - states[0], i.e. relative state w.r.t. node 0.
        This is the minimal edge representation required by GraphsTuple;
        the CBF solver uses states directly, not edges.

    Returns:
        GraphsTuple whose env_states carries ped_goal / ped_lidar / ped_speed_pref,
        consumed by PedEnvWrapper.u_ref() to compute the reference control.
    """
    n_ped = ped_states.shape[0]
    n_rays = robot_lidar.shape[0]

    # Node order: pedestrian agents, robot, robot lidar hits
    n_nodes = n_ped + 1 + n_rays
    node_feats = jnp.zeros((n_nodes, 4), dtype=jnp.float32)

    # node_type=0: agent (pedestrian); node_type=1: obstacle (robot + lidar points)
    node_type = jnp.zeros((n_nodes,), dtype=jnp.int32)
    node_type = node_type.at[n_ped:].set(1)

    # Lidar hit points have no velocity; pad with zeros to get shape (n_rays, 4)
    robot_lidar_state = jnp.concatenate([robot_lidar, jnp.zeros((n_rays, 2))], axis=-1)
    # states: (n_nodes, 4), concatenated in node order
    states = jnp.concatenate([ped_states, robot_state, robot_lidar_state], axis=0)

    # Minimal star-topology edges: every node connects to node 0 as sender
    # Only needed to satisfy the GraphsTuple interface; CBF uses states directly
    receivers = jnp.arange(n_nodes, dtype=jnp.int32)
    senders = jnp.zeros((n_nodes,), dtype=jnp.int32)
    edges = states[receivers] - states[senders]  # Only needed to satisfy the GraphsTuple interface; CBF uses states directly

    n_node = jnp.array(n_nodes, dtype=jnp.int32)
    n_edge = jnp.array(n_nodes, dtype=jnp.int32)

    env_states = PedGraphEnvState(
        ped_goal=ped_goal, ped_lidar=ped_lidar, ped_speed_pref=ped_speed_pref
    )

    return GraphsTuple(
        n_node=n_node,
        n_edge=n_edge,
        nodes=node_feats,        # (n_nodes, 4)  zero-filled placeholder; actual states are in `states`
        edges=edges,             # (n_nodes, 4)  relative state of each node w.r.t. node 0
        states=states,           # (n_nodes, 4)  absolute state [x, y, vx, vy] for every node
        receivers=receivers,     # (n_nodes,)    destination node index for each edge
        senders=senders,         # (n_nodes,)    source node index for each edge (all 0)
        node_type=node_type,     # (n_nodes,)    0=agent (pedestrian), 1=obstacle (robot/lidar)
        env_states=env_states,   # PedGraphEnvState  carries goal, lidar, and preferred speed
        connectivity=None,
    )


def ped_sfm_step(
    ped_states: Array,
    ped_goals: Array,
    robot_state: Array,
    obstacles,
    params: dict,
    dt: float,
    ped_speed_pref: Array | None = None,
) -> Array:
    n_ped = ped_states.shape[0]
    n_rays = int(params["n_rays"])

    get_lidar_vmap = jax.vmap(
        lambda p: get_lidar(p, obstacles, num_beams=n_rays, sense_range=params["comm_radius"])
    )
    ped_states_4 = ped_states[:, :4]
    ped_lidar = get_lidar_vmap(ped_states_4[:, :2])

    pos = ped_states[:, :2]
    vel = ped_states[:, 2:4]

    # goal force
    goal_vec = ped_goals - pos
    goal_dir = goal_vec / (jnp.linalg.norm(goal_vec, axis=-1, keepdims=True) + 1e-6)
    if ped_speed_pref is None:
        v0 = float(params.get("ped_speed_pref", 1.0))
        v_des = v0 * goal_dir
    else:
        v_des = ped_speed_pref[:, None] * goal_dir
    tau = params.get("ped_tau", 0.5)
    f_goal = (v_des - vel) / tau

    # ped-ped repulsion
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
    dist = dist + jnp.eye(n_ped)[:, :, None] * 1e6
    n_ij = diff / dist
    A = params.get("sfm_A", 2.0)
    B = params.get("sfm_B", 0.3)
    r_sum = params.get("ped_radius", 0.04) * 2
    f_ped = jnp.sum(A * jnp.exp((r_sum - dist) / B) * n_ij, axis=1)

    # robot repulsion
    r_pos = robot_state[:1, :2]
    diff_r = pos - r_pos
    dist_r = jnp.linalg.norm(diff_r, axis=-1, keepdims=True) + 1e-6
    n_ir = diff_r / dist_r
    r_sum_r = params.get("ped_radius", 0.04) + params.get("car_radius", 0.05)
    f_robot = A * jnp.exp((r_sum_r - dist_r) / B) * n_ir

    # obstacle repulsion using ped lidar hits
    lidar_vec = pos[:, None, :] - ped_lidar
    lidar_dist = jnp.linalg.norm(lidar_vec, axis=-1, keepdims=True) + 1e-6
    lidar_dir = lidar_vec / lidar_dist
    f_obs = jnp.sum(A * jnp.exp((params.get("ped_radius", 0.04) - lidar_dist) / B) * lidar_dir, axis=1)

    acc = f_goal + f_ped + f_robot + f_obs
    vel_next = vel + dt * acc
    pos_next = pos + dt * vel_next

    ped_next = jnp.stack([pos_next[:, 0], pos_next[:, 1], vel_next[:, 0], vel_next[:, 1]], axis=-1)
    return jnp.concatenate([ped_next, jnp.zeros((n_ped, 1))], axis=-1)


def ped_cbf_step(
    controller: PedCBFController,
    ped_states: Array,
    ped_goals: Array,
    robot_state: Array,
    obstacles,
    params: dict,
    dt: float,
    ped_speed_pref: Array | None = None,
    ped_ignore_robot: Array | None = None,
) -> Array:
    n_rays = int(params["n_rays"])

    get_lidar_vmap = jax.vmap(
        lambda p: get_lidar(p, obstacles, num_beams=n_rays, sense_range=params["comm_radius"])
    )
    ped_states_4 = ped_states[:, :4]
    ped_lidar = get_lidar_vmap(ped_states_4[:, :2])

    robot_lidar = get_lidar(robot_state[0, :2], obstacles, num_beams=n_rays, sense_range=params["comm_radius"])
    # Convert robot state to [x, y, vx, vy] for double-integrator CBF
    robot_theta = robot_state[:, 2]
    robot_speed = robot_state[:, 3]
    robot_vel = jnp.stack(
        [jnp.cos(robot_theta) * robot_speed, jnp.sin(robot_theta) * robot_speed],
        axis=-1,
    )
    robot_state_di = jnp.concatenate([robot_state[:, :2], robot_vel], axis=-1)
    if ped_speed_pref is None:
        ped_speed_pref = jnp.full((ped_states_4.shape[0],), float(params.get("ped_speed_pref", 1.0)))
    ped_graph = build_ped_graph(
        ped_states_4, robot_state_di, robot_lidar, ped_goals, ped_lidar, ped_speed_pref
    )

    u_opt, _ = controller.get_qp_action(ped_graph)
    u_opt = controller._env.clip_action(u_opt)

    # Integrate Dubins-like dynamics for pedestrians
    f, g = controller._env.control_affine_dyn(ped_states_4)
    x_dot = f + jnp.einsum("nij,nj->ni", g, u_opt)
    ped_next_4_with_robot = ped_states_4 + x_dot * dt

    if ped_ignore_robot is None or ped_ignore_robot.size == 0:
        ped_next_4 = ped_next_4_with_robot
    else:
        ped_ignore_robot = jnp.asarray(ped_ignore_robot, dtype=bool)
        any_ignore = jnp.any(ped_ignore_robot)

        def _with_ignore(_):
            # Recompute with robot moved far away to ignore robot interaction.
            far = float(params.get("ped_ignore_robot_far", 1e6))
            robot_state_di_far = robot_state_di.at[:, :2].set(
                robot_state_di[:, :2] + jnp.array([far, far], dtype=robot_state_di.dtype)
            )
            ped_graph_no_robot = build_ped_graph(
                ped_states_4, robot_state_di_far, robot_lidar, ped_goals, ped_lidar, ped_speed_pref
            )
            u_opt_no_robot, _ = controller.get_qp_action(ped_graph_no_robot)
            u_opt_no_robot = controller._env.clip_action(u_opt_no_robot)
            x_dot_no_robot = f + jnp.einsum("nij,nj->ni", g, u_opt_no_robot)
            ped_next_4_no_robot = ped_states_4 + x_dot_no_robot * dt
            return jnp.where(ped_ignore_robot[:, None], ped_next_4_no_robot, ped_next_4_with_robot)

        ped_next_4 = jax.lax.cond(any_ignore, _with_ignore, lambda _: ped_next_4_with_robot, operand=None)

    return jnp.concatenate([ped_next_4, jnp.zeros((ped_next_4.shape[0], 1))], axis=-1)


def ped_orca_step(
    ped_states: Array,
    ped_goals: Array,
    robot_state: Array,
    obstacles,
    params: dict,
    dt: float,
    ped_speed_pref: Array | None = None,
) -> Array:
    try:
        import rvo2
    except Exception as exc:
        raise RuntimeError("ped_sim_mode='orca' requires the rvo2 package to be installed.") from exc

    n_ped = int(ped_states.shape[0])
    n_rays = int(params["n_rays"])
    ped_radius = float(params.get("ped_radius", 0.04))
    car_radius = float(params.get("car_radius", 0.05))

    time_step = float(params.get("orca_time_step", dt))
    neighbor_dist = float(params.get("orca_neighbor_dist", 5.0))
    max_neighbors = int(params.get("orca_max_neighbors", 10))
    time_horizon = float(params.get("orca_time_horizon", 5.0))
    time_horizon_obst = float(params.get("orca_time_horizon_obst", 5.0))
    max_speed = float(params.get("orca_max_speed", params.get("ped_speed_pref", 1.0) * 1.5))
    robot_max_speed = float(params.get("orca_robot_max_speed", max_speed))

    sim = rvo2.PyRVOSimulator(
        time_step, neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, ped_radius, max_speed
    )

    # Add static obstacles as ORCA polygons if available
    if hasattr(obstacles, "points"):
        for poly in obstacles.points:
            poly_np = np.asarray(poly)
            sim.addObstacle([(float(p[0]), float(p[1])) for p in poly_np])
        sim.processObstacles()

    ped_pos = np.asarray(ped_states[:, :2])
    ped_vel = np.asarray(ped_states[:, 2:4])

    robot_pos = np.asarray(robot_state[0, :2])
    robot_theta = float(robot_state[0, 2])
    robot_speed = float(robot_state[0, 3])
    robot_vel = np.array([np.cos(robot_theta) * robot_speed, np.sin(robot_theta) * robot_speed])

    # Add pedestrian agents
    for i in range(n_ped):
        sim.addAgent((float(ped_pos[i, 0]), float(ped_pos[i, 1])))

    # Add robot as an agent so pedestrians avoid it (robot itself is not controlled by ORCA here)
    robot_id = sim.addAgent(
        (float(robot_pos[0]), float(robot_pos[1])),
        neighbor_dist,
        max_neighbors,
        time_horizon,
        time_horizon_obst,
        car_radius,
        robot_max_speed,
    )

    # Set current velocities
    for i in range(n_ped):
        sim.setAgentVelocity(i, (float(ped_vel[i, 0]), float(ped_vel[i, 1])))
    sim.setAgentVelocity(robot_id, (float(robot_vel[0]), float(robot_vel[1])))

    # Set preferred velocities
    for i in range(n_ped):
        goal_vec = np.asarray(ped_goals[i]) - ped_pos[i]
        norm = np.linalg.norm(goal_vec) + 1e-6
        if ped_speed_pref is None:
            v_pref = float(params.get("ped_speed_pref", 1.0))
        else:
            v_pref = float(ped_speed_pref[i])
        pref = v_pref * (goal_vec / norm)
        sim.setAgentPrefVelocity(i, (float(pref[0]), float(pref[1])))
    sim.setAgentPrefVelocity(robot_id, (float(robot_vel[0]), float(robot_vel[1])))

    sim.doStep()

    # Read back pedestrian states
    next_states = np.zeros((n_ped, 4), dtype=np.float32)
    for i in range(n_ped):
        pos = sim.getAgentPosition(i)
        vel = sim.getAgentVelocity(i)
        next_states[i, 0] = pos[0]
        next_states[i, 1] = pos[1]
        next_states[i, 2] = vel[0]
        next_states[i, 3] = vel[1]

    return jnp.concatenate([jnp.array(next_states), jnp.zeros((n_ped, 1))], axis=-1)
