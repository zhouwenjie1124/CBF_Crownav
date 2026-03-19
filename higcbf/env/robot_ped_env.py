import pathlib
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxproxqp.jaxproxqp import JaxProxQP

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Reward, State
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle, Circle, MixedObstacle, CIRCLE, RECTANGLE
from .plot import render_video
from .utils import compute_inv_ttc, get_lidar, inside_obstacles, get_node_goal_rng
from .ped_env_wrapper import PedEnvWrapper
from .ped_sim import PedCBFController, ped_sfm_step, ped_cbf_step, ped_orca_step, build_ped_graph
from .robot_dynamics import RobotDynamics
from .path_planner import PathPlanner
from .rewards import assemble_legacy_proactive, assemble_paper, assemble_height


class RobotPedEnv(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    PED = 2
    OBS = 3

    class EnvState(NamedTuple):
        agent: AgentState  # (1, 5)
        goal: State  # (1, 5)
        ped: AgentState  # (n_ped, 5)
        ped_goal: State  # (n_ped, 2)
        ped_speed_pref: Array  # (n_ped,)
        ped_ignore_robot: Array  # (n_ped,)
        ped_active: Array  # (n_ped,)
        obstacle: Obstacle
        ped_key: Array  # (n_ped, 2)
        stuck_count: Array  # (1,)
        path_waypoints: Array  # (path_max_waypoints, 2)
        path_valid: Array  # (path_max_waypoints,)
        path_len: Array  # (1,)
        subgoal_idx: Array  # (1,)

        @property
        def n_ped(self) -> int:
            return self.ped.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        # Environment parameters
        "car_radius": 0.3,
        "comm_radius": 6.0,
        "avoid_mode": "cbf",  # robot avoida mode : "none", "sfm", or "cbf" (for robot u_ref)
        "n_rays": 16,
        "obs_len_range": [0.5, 1.0],
        "n_obs": 6,
        "obs_type": "mixed",  # "rectangle", "circle", or "mixed"
        "obs_circle_frac": 0.5,
        "hard_obstacle": True, # if True: robot can not enter obstacles; if False: robot can enter obstacles but receives cost
        # Pedestrian parameters
        "n_ped": 8,
        "ped_count_min": 5,
        "ped_count_max": 9,
        "ped_radius": 0.3,
        "ped_goal_replan_dist": 0.2,
        "ped_sim_mode": "cbf",  # "sfm", "cbf", or "orca(unfinished)"
        "ped_speed_pref": 1.0,
        "ped_speed_pref_min": 0.5,
        "ped_speed_pref_max": 1.0,
        "ped_ignore_robot_frac": 0.0,
        # robot start-goal distance constraint (meters)
        "robot_goal_dist_min": 6.0,
        "robot_goal_dist_max": 8.0,
        "ped_speed_init_min": 0.5,
        "ped_speed_init_max": 1.0,
        # ORCA
        "orca_neighbor_dist": 5.0,
        "orca_max_neighbors": 10,
        "orca_time_horizon": 5.0,
        "orca_time_horizon_obst": 5.0,
        "orca_max_speed": 1.5,
        # Action is acceleration (a_w, a_v), applied each step: v_next = v + a_v * dt
        "v_min": 0,
        "v_max": 1.5,
        "delta_v_max": 1,
        "delta_omega_max": 1,
        "delta_v_step": 0.5,
        "delta_w_step": 1,
        # action constraint 
        "a_v_max": 1,
        "a_w_max": 1,
        "w_min": -1.5,
        "w_max": 1.5,
        "a_v_step": 1,
        "a_w_step": 1,
        "action_mode": "discrete_delta",  # "discrete_delta", "continuous_vw", or "discrete_vw_grid"
        "action_discrete": True,
        # Absolute command grid in public (v, w) semantics for discrete_vw_grid.
        "vw_grid_v": [-1.5, -0.75, 0.0, 0.75, 1.5],
        "vw_grid_w": [-1.5, -0.75, 0.0, 0.75, 1.5],
        # Graph construction
        "normalize_nodes": True,
        "normalize_edges": True,
        # Termination conditions
        "goal_tolerance": 0.1,
        "stop_tolerance": 0.025,
        "terminate_on_collision": True,  # keep training behavior; can be disabled for test-time rollout
        # Reward Function 1 
        "safety_margin": 0.2,
        "kappa_succ": 20.0,
        "kappa_col": -20.0,
        "kappa_safe": 1.5,
        "kappa_prog": 4.0,
        "ttc_threshold": 1.0,
        "kappa_ttc": -0.5,
        "kappa_inv": -0.2,
        "ttc_eps": 1e-5,
        "kappa_delta": 0.5,
        "lambda_omega": 1.0,
        "kappa_omega": 0.000125,
        "kappa_time": 0.025,
        "kappa_timeout": -15.0,
        # reward mode: "legacy" (original), "proactive" (stuck-avoid), "paper" (reward2)
        "reward_mode": "proactive",
        # Reward Function 2 
        "r_goal": 20.0,
        "r_path": 3.2,
        "g_m": None,  # goal tolerance (synced to car_radius in __init__)
        "r_collision": -20.0,
        "r_obstacle": -0.2,
        "d_r": None,  # collision radius (synced to car_radius in __init__)
        "d_m": 1.2,
        
        "r_rotation": -0.1,
        "omega_m": 1.0,
        # safety penalty mode: "sum" (all close agents/obs) or "min" (only nearest)
        "safe_penalty_mode": "min",
        # desired heading reward (VO-based)
        "heading_reward": True,
        "heading_n_samples": 33,
        "heading_theta_m": np.pi / 6,
        "heading_r_angle": 0.6,
        # path-based heading shaping
        "path_reward_enable": True,
        "astar_grid_size": 64,
        "astar_max_expand": 4096,
        "astar_allow_diag": True,
        "path_max_waypoints": 128,
        "astar_fallback_to_goal": True,
        # Pure Pursuit lookahead target for heading shaping.
        "lookahead_enable": True,
        "lookahead_l0": 0.8,
        "lookahead_kv": 1.2,
        "lookahead_min": 0.8,
        "lookahead_max": 3.0,
        "lookahead_eps": 1e-6,
        # training-time CBF filter on velocity commands [w_cmd, v_cmd]
        "train_cbf_filter": True,
        "train_cbf_alpha": 1.0,
        "train_cbf_sigma": 0.3,
        "train_cbf_eps": 1e-6,
        "train_cbf_weight": 0.1,
        "train_cbf_use_ped": True,
        "train_cbf_use_lidar": True,
        "train_cbf_use_walls": True,
        # anti-stuck shaping
        "anti_stuck_enable": False,
        "anti_stuck_k": 0.02,
        "anti_stuck_progress_eps": 0.01,
        "anti_stuck_speed_eps": 0.05,
        "anti_stuck_free_steps": 10,
        "anti_stuck_use_obs_blocked": True,
        "anti_stuck_obs_clear_thresh": None,
    }

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        max_step: int = 256,
        max_travel: float = None,
        dt: float = 0.1,
        params: dict = None,
    ):
        assert num_agents == 1, "RobotPedEnv only supports a single robot."
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)
        # RobotDynamics handles param syncing and validation (mutates self._params in-place)
        self._dynamics = RobotDynamics(self._params)
        # Keep reward thresholds consistent with robot size
        self._params["g_m"] = self._params["car_radius"]
        self._params["d_r"] = self._params["car_radius"]
        obs_type = self._params.get("obs_type", "rectangle")
        if obs_type == "circle":
            self.create_obstacles = jax.vmap(Circle.create)
        elif obs_type == "mixed":
            self.create_obstacles = None  # handled per-episode in reset
        else:
            self.create_obstacles = jax.vmap(Rectangle.create)
        self.enable_stop = True

        self._ped_env = PedEnvWrapper(
            num_peds=self._params["n_ped"], params=self._params, dt=self.dt, n_rays=self._params["n_rays"]
        )
        self._ped_controller = PedCBFController(
            env=self._ped_env,
            node_dim=self._ped_env.node_dim,
            edge_dim=self._ped_env.edge_dim,
            state_dim=self._ped_env.state_dim,
            action_dim=self._ped_env.action_dim,
            n_agents=self._params["n_ped"],
            alpha=1.0,
        )

        # Robot CBF controller (single-agent) for avoid_mode="cbf"
        self._robot_cbf_env = PedEnvWrapper(
            num_peds=1, params=self._params, dt=self.dt, n_rays=self._params["n_rays"]
        )
        self._robot_cbf_controller = PedCBFController(
            env=self._robot_cbf_env,
            node_dim=self._robot_cbf_env.node_dim,
            edge_dim=self._robot_cbf_env.edge_dim,
            state_dim=self._robot_cbf_env.state_dim,
            action_dim=self._robot_cbf_env.action_dim,
            n_agents=1,
            alpha=1.0,
        )
        # Path planner (A* + pure-pursuit lookahead)
        self._path_planner = PathPlanner(self._params, self.area_size)

    @property
    def state_dim(self) -> int:
        return 5  # robot: x, y, theta, v, w | ped: x, y, vx, vy, 0

    @property
    def node_dim(self) -> int:
        # one-hot(4) + x, y, vx, vy, cos(theta), sin(theta), v, w
        # + goal_dx, goal_dy, cos(theta_err), sin(theta_err)
        return 16

    @property
    def edge_dim(self) -> int:
        return 5  # x_rel, y_rel, v_rel_x, v_rel_y, inv_ttc

    @property
    def action_dim(self) -> int:
        return 2  # a_w, a_v (angular/linear acceleration)

    # ------------------------------------------------------------------
    # Dynamics delegates (thin wrappers – logic lives in RobotDynamics)
    # ------------------------------------------------------------------

    def _is_vw_command_mode(self) -> bool:
        return self._dynamics.is_vw_command_mode()

    def _get_vw_grid_bins(self) -> tuple[Array, Array]:
        return self._dynamics.get_vw_grid_bins()

    # ------------------------------------------------------------------
    # Path planning delegates (thin wrappers – logic lives in PathPlanner)
    # ------------------------------------------------------------------

    def _empty_path_plan(self, goal_xy: Array) -> tuple[Array, Array, Array]:
        return self._path_planner.empty_plan(goal_xy)

    def _plan_path_astar(self, obstacles, start_xy, goal_xy):
        return self._path_planner.plan(obstacles, start_xy, goal_xy)

    def _compute_lookahead_distance(self, v_abs):
        return self._path_planner.lookahead_dist(v_abs)

    def _compute_path_projection_s(self, path_waypoints, path_len, agent_xy):
        return self._path_planner.project(path_waypoints, path_len, agent_xy)

    def _interp_path_at_s(self, path_waypoints, path_len, s_query):
        return self._path_planner.interpolate(path_waypoints, path_len, s_query)

    def _lookahead_target_from_state(self, env_state, agent_states):
        return self._path_planner.get_target(
            env_state.path_waypoints,
            env_state.path_len,
            env_state.goal[0, :2],
            agent_states,
        )

    def _heading_target_from_state(self, env_state, agent_states, advance=True):
        del advance
        target, _ = self._lookahead_target_from_state(env_state, agent_states)
        return target, jnp.array(0, dtype=jnp.int32)

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # obstacles
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (self._params["n_obs"], 2), minval=0, maxval=self.area_size)
        obs_type = self._params.get("obs_type", "rectangle")
        if obs_type == "circle":
            radius_key, key = jr.split(key, 2)
            obs_radius = jr.uniform(
                radius_key,
                (self._params["n_obs"],),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            obstacles = self.create_obstacles(obs_pos, obs_radius)
        elif obs_type == "mixed":
            frac = float(self._params.get("obs_circle_frac", 0.5))
            frac = float(np.clip(frac, 0.0, 1.0))
            n_obs = int(self._params["n_obs"])
            type_key, key = jr.split(key, 2)
            is_circle = jr.uniform(type_key, (n_obs,)) < frac

            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (n_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_obs,), minval=0, maxval=2 * np.pi)
            rects = jax.vmap(Rectangle.create)(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

            radius_key, key = jr.split(key, 2)
            obs_radius = jr.uniform(
                radius_key,
                (n_obs,),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )

            types = jnp.where(is_circle, CIRCLE, RECTANGLE)
            width = jnp.where(is_circle, 0.0, rects.width)
            height = jnp.where(is_circle, 0.0, rects.height)
            theta = jnp.where(is_circle, 0.0, rects.theta)
            radius = jnp.where(is_circle, obs_radius, 0.0)
            points = jnp.where(is_circle[:, None, None], jnp.zeros_like(rects.points), rects.points)

            obstacles = MixedObstacle(
                type=types,
                center=obs_pos,
                width=width,
                height=height,
                theta=theta,
                radius=radius,
                points=points,
            )
        else:
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (self._params["n_obs"], 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (self._params["n_obs"],), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # robot + ped initial states and goals
        n_ped = self._params["n_ped"]
        n_total = 1 + n_ped
        goal_key, key = jr.split(key, 2)
        states, goals = get_node_goal_rng(
            key,
            self.area_size,
            2,
            obstacles,
            n_total,
            2 * self.params["car_radius"],
            self.max_travel,
        )

        # Enforce robot start-goal distance in [robot_goal_dist_min, robot_goal_dist_max]
        goal_dist_min = float(self._params.get("robot_goal_dist_min", 5.0))
        goal_dist_max = float(self._params.get("robot_goal_dist_max", 6.0))
        agent_pos = states[0]
        ped_goals = goals[1:]
        min_dist = 2 * self.params["car_radius"]
        max_iter = 1024

        def goal_invalid(goal):
            dist = jnp.linalg.norm(goal - agent_pos)
            dist_ok = jnp.logical_and(dist >= goal_dist_min, dist <= goal_dist_max)
            outside = jnp.any(goal < 0.0) | jnp.any(goal > self.area_size)
            inside = inside_obstacles(goal[None, :], obstacles, r=self._params["car_radius"])[0]
            too_close = jnp.array(False)
            if ped_goals.shape[0] > 0:
                too_close = jnp.any(jnp.linalg.norm(ped_goals - goal, axis=-1) <= min_dist)
            return (~dist_ok) | outside | inside | too_close

        def goal_body(carry):
            i_iter, this_key, _ = carry
            this_key, a_key, d_key = jr.split(this_key, 3)
            angle = jr.uniform(a_key, (), minval=0.0, maxval=2 * np.pi)
            dist = jr.uniform(d_key, (), minval=goal_dist_min, maxval=goal_dist_max)
            goal = agent_pos + dist * jnp.array([jnp.cos(angle), jnp.sin(angle)])
            return i_iter + 1, this_key, goal

        def goal_cond(carry):
            i_iter, _, goal = carry
            return jnp.logical_and(i_iter < max_iter, goal_invalid(goal))

        _, _, robot_goal = jax.lax.while_loop(
            goal_cond, goal_body, (jnp.array(0), goal_key, goals[0])
        )
        goals = goals.at[0].set(robot_goal)

        # add robot heading; initialize pedestrian velocities
        theta_key, key = jr.split(key, 2)
        states = jnp.concatenate([states, jnp.zeros((n_total, 3))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((n_total, 3))], axis=1)
        rand_theta = jr.uniform(theta_key, (1,), minval=-np.pi, maxval=np.pi)
        states = states.at[0, 2].set(rand_theta[0])
        goal_theta = jnp.arctan2(goals[0, 1] - states[0, 1], goals[0, 0] - states[0, 0])
        goals = goals.at[0, 2].set(goal_theta)

        # randomize pedestrian initial velocities
        ped_key, key = jr.split(key, 2)
        ped_speeds = jr.uniform(
            ped_key,
            (n_ped,),
            minval=float(self._params.get("ped_speed_init_min", 0.5)),
            maxval=float(self._params.get("ped_speed_init_max", 1.0)),
        )
        ped_key, key = jr.split(key, 2)
        ped_angles = jr.uniform(ped_key, (n_ped,), minval=-np.pi, maxval=np.pi)
        ped_vx = ped_speeds * jnp.cos(ped_angles)
        ped_vy = ped_speeds * jnp.sin(ped_angles)
        ped_key, key = jr.split(key, 2)
        ped_speed_pref = jr.uniform(
            ped_key,
            (n_ped,),
            minval=float(self._params.get("ped_speed_pref_min", 0.5)),
            maxval=float(self._params.get("ped_speed_pref_max", 1.0)),
        )
        ped_key, key = jr.split(key, 2)
        ped_ignore_robot = jr.uniform(ped_key, (n_ped,)) < float(self._params.get("ped_ignore_robot_frac", 0.0))
        if n_ped > 0:
            ped_min = self._params.get("ped_count_min", n_ped)
            ped_max = self._params.get("ped_count_max", n_ped)
            if ped_min is None:
                ped_min = n_ped
            if ped_max is None:
                ped_max = n_ped
            ped_min = jnp.asarray(ped_min, dtype=jnp.int32)
            ped_max = jnp.asarray(ped_max, dtype=jnp.int32)
            ped_min = jnp.clip(ped_min, 0, n_ped)
            ped_max = jnp.clip(ped_max, ped_min, n_ped)
            ped_key, key = jr.split(key, 2)
            n_active = jr.randint(ped_key, (), minval=ped_min, maxval=ped_max + 1)
            ped_key, key = jr.split(key, 2)
            perm = jr.permutation(ped_key, n_ped)
            rank = jnp.zeros_like(perm).at[perm].set(jnp.arange(n_ped))
            ped_active = rank < n_active
        else:
            ped_active = jnp.zeros((0,), dtype=bool)
        if n_ped > 0:
            states = states.at[1:, 2].set(ped_vx)
            states = states.at[1:, 3].set(ped_vy)

        agent = states[:1]
        goal = goals[:1]
        path_waypoints, path_valid, path_len = self._plan_path_astar(
            obstacles, agent[0, :2], goal[0, :2]
        )
        ped = states[1:]
        ped_goal = goals[1:, :2]
        if n_ped > 0:
            far = self.area_size * 10.0
            ped_active_f = ped_active[:, None]
            ped_pos = states[1:, :2]
            ped_pos = jnp.where(ped_active_f, ped_pos, jnp.full_like(ped_pos, far))
            states = states.at[1:, :2].set(ped_pos)
            states = states.at[1:, 2].set(jnp.where(ped_active, states[1:, 2], 0.0))
            states = states.at[1:, 3].set(jnp.where(ped_active, states[1:, 3], 0.0))
            ped_goal = jnp.where(ped_active_f, ped_goal, ped_pos)
            ped_speed_pref = jnp.where(ped_active, ped_speed_pref, 0.0)
            ped = states[1:]
        ped_key = jr.split(key, n_ped)

        env_states = self.EnvState(
            agent=agent,
            goal=goal,
            ped=ped,
            ped_goal=ped_goal,
            ped_speed_pref=ped_speed_pref,
            ped_ignore_robot=ped_ignore_robot,
            ped_active=ped_active,
            obstacle=obstacles,
            ped_key=ped_key,
            stuck_count=jnp.zeros((1,), dtype=jnp.float32),
            path_waypoints=path_waypoints,
            path_valid=path_valid,
            path_len=path_len,
            subgoal_idx=jnp.zeros((1,), dtype=jnp.int32),
        )
        return self.get_graph(env_states)

    def agent_step_euler(self, agent_states: AgentState, action: Action, stop_mask: Array) -> AgentState:
        assert action.shape == (1, self.action_dim)
        x_dot = self.agent_xdot(agent_states, action) * (1 - stop_mask)[:, None]
        n_state_agent_new = agent_states + x_dot * self.dt
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        x_dot = jnp.concatenate(
            [
                (jnp.cos(agent_states[:, 2]) * agent_states[:, 3])[:, None],
                (jnp.sin(agent_states[:, 2]) * agent_states[:, 3])[:, None],
                (agent_states[:, 4])[:, None],
                (action[:, 1])[:, None],
                (action[:, 0])[:, None],
            ],
            axis=1,
        )
        return x_dot

    def step(self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False):
        self._t += 1

        agent_states = graph.type_states(type_idx=0, n_type=1)
        goal_states = graph.env_states.goal
        ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
        obstacles = graph.env_states.obstacle

        action_policy = self.clip_action(action)
        action_cmd, cbf_info = self._filter_train_cbf_action(graph, action_policy)
        action = self._apply_action_mode(agent_states, action_cmd)
        stop_mask = self.stop_mask(graph)
        if not self.enable_stop:
            stop_mask = 0 * stop_mask
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_agent_pos = next_agent_states[:, :2]
        collision_obs_attempt = inside_obstacles(next_agent_pos, obstacles, r=self._params["car_radius"])
        if self._params.get("hard_obstacle", False):
            # hard constraint: do not enter obstacles, stop robot if it tries to
            next_agent_states = jnp.where(collision_obs_attempt[:, None], agent_states, next_agent_states)
            next_agent_states = next_agent_states.at[:, 3].set(
                jnp.where(collision_obs_attempt, 0.0, next_agent_states[:, 3])
            )
            next_agent_states = next_agent_states.at[:, 4].set(
                jnp.where(collision_obs_attempt, 0.0, next_agent_states[:, 4])
            )
            next_agent_pos = next_agent_states[:, :2]

        # re-sample pedestrian goals if reached
        ped_goal, ped_key = self._refresh_ped_goals(
            graph.env_states.ped,
            graph.env_states.ped_goal,
            graph.env_states.ped_key,
            obstacles,
            graph.env_states.ped_active,
        )

        if self._params["ped_sim_mode"] == "sfm":
            next_ped_states = ped_sfm_step(
                ped_states,
                ped_goal,
                next_agent_states,
                obstacles,
                self._params,
                self.dt,
                graph.env_states.ped_speed_pref,
            )
        elif self._params["ped_sim_mode"] == "cbf":
            next_ped_states = ped_cbf_step(
                self._ped_controller,
                ped_states,
                ped_goal,
                next_agent_states,
                obstacles,
                self._params,
                self.dt,
                graph.env_states.ped_speed_pref,
                graph.env_states.ped_ignore_robot,
            )
        elif self._params["ped_sim_mode"] == "orca":
            next_ped_states = ped_orca_step(
                ped_states,
                ped_goal,
                next_agent_states,
                obstacles,
                self._params,
                self.dt,
                graph.env_states.ped_speed_pref,
            )
        else:
            raise ValueError(f"Unknown ped_sim_mode: {self._params['ped_sim_mode']}")

        if self._params["n_ped"] > 0:
            active = graph.env_states.ped_active
            next_ped_states = jnp.where(active[:, None], next_ped_states, ped_states)

        info = {}
        cbf_weight = float(self._params.get("train_cbf_weight", 0.0))
        if get_eval_info:
            reward_terms = self._compute_reward_terms_selected(
                graph, next_agent_states, action, collision_obs_attempt
            )
            reward = reward_terms["reward"] + cbf_weight * cbf_info["r_cbf"]
            reward_terms["r_cbf"] = cbf_info["r_cbf"]
            reward_terms["cbf_h"] = cbf_info["cbf_h"]
            reward_terms["cbf_a"] = cbf_info["cbf_a"]
            reward_terms["cbf_b"] = cbf_info["cbf_b"]
            reward_terms["cbf_c"] = cbf_info["cbf_c"]
            reward_terms["cbf_active"] = cbf_info["cbf_active"]
            reward_terms["v_policy"] = cbf_info["v_policy"]
            reward_terms["v_safe"] = cbf_info["v_safe"]
            info.update(reward_terms)
        else:
            reward = self._compute_reward(graph, next_agent_states, action, collision_obs_attempt)
            reward = reward + cbf_weight * cbf_info["r_cbf"]
        r_stuck, next_stuck_count, stuck_info = self._compute_anti_stuck(graph, next_agent_states)
        reward = reward + r_stuck
        next_subgoal_idx = jnp.zeros((1,), dtype=jnp.int32)
        if get_eval_info:
            info.update(stuck_info)
            subgoal_xy, lookahead_dbg = self._lookahead_target_from_state(
                graph.env_states, next_agent_states
            )
            subgoal_dist = jnp.linalg.norm(next_agent_pos[0] - subgoal_xy)
            info.update(
                {
                    "subgoal_idx": jnp.array(0.0, dtype=jnp.float32),
                    "subgoal_dist": subgoal_dist,
                    "path_len": graph.env_states.path_len.squeeze().astype(jnp.float32),
                    "path_valid_ratio": graph.env_states.path_valid.astype(jnp.float32).mean(),
                }
            )
            info.update(lookahead_dbg)
        next_ped_pos = next_ped_states[:, :2]
        dist = jnp.linalg.norm(next_agent_pos[:, None, :] - next_ped_pos[None, :, :], axis=-1)
        collision_ped = (self._params["car_radius"] + self._params["ped_radius"] > dist).any(axis=1)
        collision_obs = collision_obs_attempt
        collision = jnp.logical_or(collision_ped, collision_obs)
        goal_pos = goal_states[:, :2]
        reach = jnp.linalg.norm(next_agent_pos - goal_pos, axis=-1) <= self._params["goal_tolerance"]
        timeout = jnp.array(self._t >= self.max_episode_steps)
        done = jnp.logical_or(reach, timeout)
        if self._params.get("terminate_on_collision", True):
            done = jnp.logical_or(collision, done)
        done = done.squeeze()
        cost = self._compute_cost_from_states(next_agent_pos, next_ped_pos, obstacles)
        if self._params.get("hard_obstacle", False):
            cost = cost + collision_obs_attempt.mean()

        next_state = self.EnvState(
            agent=next_agent_states,
            goal=goal_states,
            ped=next_ped_states,
            ped_goal=ped_goal,
            ped_speed_pref=graph.env_states.ped_speed_pref,
            ped_ignore_robot=graph.env_states.ped_ignore_robot,
            ped_active=graph.env_states.ped_active,
            obstacle=obstacles,
            ped_key=ped_key,
            stuck_count=next_stuck_count,
            path_waypoints=graph.env_states.path_waypoints,
            path_valid=graph.env_states.path_valid,
            path_len=graph.env_states.path_len,
            subgoal_idx=next_subgoal_idx,
        )
        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        ped_pos = graph.type_states(type_idx=2, n_type=self._params["n_ped"])[:, :2]
        obstacles = graph.env_states.obstacle
        return self._compute_cost_from_states(agent_pos, ped_pos, obstacles)

    def _compute_cost_from_states(
        self, agent_pos: Array, ped_pos: Array, obstacles: Obstacle
    ) -> Cost:
        # robot-ped collision
        dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        collision_ped = (self._params["car_radius"] + self._params["ped_radius"] > dist).any(axis=1)
        cost = collision_ped.mean()

        # robot-obstacle collision
        collision_obs = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision_obs.mean()
        return cost

    def _compute_anti_stuck(
        self,
        graph: EnvGraphsTuple,
        next_agent_states: AgentState,
    ) -> tuple[Array, Array, dict]:
        prev_count = graph.env_states.stuck_count
        zero = jnp.array(0.0, dtype=jnp.float32)
        info = {
            "r_stuck": zero,
            "stuck_flag": zero,
            "stuck_count": prev_count.squeeze(),
            "stuck_progress": zero,
            "stuck_obs_blocked": zero,
        }
        if not bool(self._params.get("anti_stuck_enable", False)):
            return zero, prev_count, info

        progress_eps = float(self._params.get("anti_stuck_progress_eps", 0.01))
        speed_eps = float(self._params.get("anti_stuck_speed_eps", 0.05))
        free_steps = float(self._params.get("anti_stuck_free_steps", 10))
        k_stuck = float(self._params.get("anti_stuck_k", 0.02))
        use_obs_blocked = bool(self._params.get("anti_stuck_use_obs_blocked", True))

        agent_states = graph.type_states(type_idx=0, n_type=1)
        agent_pos = agent_states[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        d_prev = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        d_next = jnp.linalg.norm(next_agent_states[:, :2] - goal_pos, axis=-1)
        progress = d_prev - d_next
        v_abs = jnp.abs(next_agent_states[:, 3])
        far_from_goal = d_next > float(self._params["goal_tolerance"])

        stuck_flag = (progress < progress_eps) & (v_abs < speed_eps) & far_from_goal

        lidar = get_lidar(
            agent_pos[0, :],
            graph.env_states.obstacle,
            self._params["n_rays"],
            self._params["comm_radius"],
        )
        lidar_dist = jnp.linalg.norm(lidar - agent_pos[0, :], axis=-1)
        obs_clear = lidar_dist - self._params["car_radius"]
        min_obs_clear = jnp.min(obs_clear)
        obs_thresh = self._params.get("anti_stuck_obs_clear_thresh", None)
        if obs_thresh is None:
            obs_thresh = float(self._params.get("safety_margin", 0.2))
        obs_blocked = min_obs_clear < float(obs_thresh)
        if use_obs_blocked:
            stuck_flag = jnp.logical_and(stuck_flag, obs_blocked)

        next_count = jnp.where(stuck_flag, prev_count + 1.0, jnp.zeros_like(prev_count))
        excess = jnp.maximum(next_count - free_steps, 0.0)
        r_stuck = -(k_stuck * excess).squeeze()

        info = {
            "r_stuck": r_stuck,
            "stuck_flag": stuck_flag.astype(jnp.float32).squeeze(),
            "stuck_count": next_count.squeeze(),
            "stuck_progress": progress.squeeze(),
            "stuck_obs_blocked": obs_blocked.astype(jnp.float32),
        }
        return r_stuck, next_count, info

    # ------------------------------------------------------------------
    # Reward computation – delegates to higcbf/env/rewards.py
    # ------------------------------------------------------------------

    def _get_lidar_dist(self, graph: EnvGraphsTuple) -> Array:
        """Helper: get per-ray distance from robot to lidar hit."""
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        lidar = get_lidar(agent_pos[0], graph.env_states.obstacle,
                          self._params["n_rays"], self._params["comm_radius"])
        return jnp.linalg.norm(lidar - agent_pos[0], axis=-1)

    def _get_theta_d(self, graph: EnvGraphsTuple, next_agent_states: AgentState) -> Array | None:
        """Helper: compute desired heading angle (None if heading reward disabled)."""
        if not bool(self._params.get("heading_reward", False)):
            return None
        agent_states = graph.type_states(type_idx=0, n_type=1)
        ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
        target_xy, _ = self._heading_target_from_state(graph.env_states, next_agent_states)
        return self._desired_heading_angle(
            agent_states, graph.env_states.goal, ped_states,
            graph.env_states.obstacle, target_pos=target_xy[None, :]
        )

    def _compute_reward_terms_selected(
        self,
        graph: EnvGraphsTuple,
        next_agent_states: AgentState,
        action: Action,
        collision_obs_attempt: Array | None = None,
    ) -> dict:
        reward_mode = str(self._params.get("reward_mode", "proactive")).lower()
        agent_states = graph.type_states(type_idx=0, n_type=1)
        agent_pos = agent_states[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
        ped_pos = ped_states[:, :2]
        is_timeout = jnp.array(self._t >= self.max_episode_steps)
        lidar_dist = self._get_lidar_dist(graph)
        theta_d = self._get_theta_d(graph, next_agent_states)

        if reward_mode == "height":
            return assemble_height(
                agent_pos=agent_pos,
                next_agent_states=next_agent_states,
                goal_pos=goal_pos,
                ped_pos=ped_pos,
                lidar_dist=lidar_dist,
                is_timeout=is_timeout,
                collision_obs_attempt=collision_obs_attempt,
                params=self._params,
            )
        if reward_mode == "paper":
            return assemble_paper(
                agent_pos=agent_pos,
                agent_states=agent_states,
                next_agent_states=next_agent_states,
                goal_pos=goal_pos,
                ped_pos=ped_pos,
                lidar_dist=lidar_dist,
                is_timeout=is_timeout,
                collision_obs_attempt=collision_obs_attempt,
                theta_d=theta_d,
                params=self._params,
            )
        proactive = (reward_mode == "proactive")
        return assemble_legacy_proactive(
            agent_pos=agent_pos,
            agent_states=agent_states,
            next_agent_states=next_agent_states,
            goal_pos=goal_pos,
            ped_pos=ped_pos,
            ped_states=ped_states,
            lidar_dist=lidar_dist,
            action=action,
            dt=self.dt,
            is_timeout=is_timeout,
            collision_obs_attempt=collision_obs_attempt,
            theta_d=theta_d,
            inv_ttc_fn=self._compute_inv_ttc,
            params=self._params,
            proactive=proactive,
        )

    def _compute_reward(
        self,
        graph: EnvGraphsTuple,
        next_agent_states: AgentState,
        action: Action,
        collision_obs_attempt: Array | None = None,
    ) -> Reward:
        terms = self._compute_reward_terms_selected(graph, next_agent_states, action, collision_obs_attempt)
        reward = terms["reward"]
        assert reward.shape == tuple()
        return reward

    def _desired_heading_angle(
        self,
        agent_states: AgentState,
        goal_states: State,
        ped_states: AgentState,
        obstacles: Obstacle | None = None,
        target_pos: Array | None = None,
    ) -> Array:
        """Compute desired heading angle in robot local frame using VO-style sampling."""
        r_pos = agent_states[:, :2]
        r_theta = agent_states[:, 2]
        r_v = agent_states[:, 3]
        if target_pos is None:
            g_pos = goal_states[:, :2]
        else:
            g_pos = target_pos[:, :2]

        c = jnp.cos(r_theta)
        s = jnp.sin(r_theta)

        g_diff = g_pos - r_pos
        g_x = c * g_diff[:, 0] + s * g_diff[:, 1]
        g_y = -s * g_diff[:, 0] + c * g_diff[:, 1]
        theta_g = jnp.arctan2(g_y, g_x)

        obj_pos_list = []
        obj_vel_list = []
        r_sum_list = []
        if ped_states.shape[0] > 0:
            ped_pos = ped_states[:, :2]
            ped_vel = ped_states[:, 2:4]
            obj_pos_list.append(ped_pos)
            obj_vel_list.append(ped_vel)
            r_sum_list.append(
                jnp.full((ped_pos.shape[0],), float(self._params["car_radius"] + self._params["ped_radius"]))
            )
        if obstacles is not None:
            lidar = get_lidar(
                r_pos[0, :],
                obstacles,
                self._params["n_rays"],
                self._params["comm_radius"],
            )
            if lidar.shape[0] > 0:
                obj_pos_list.append(lidar)
                obj_vel_list.append(jnp.zeros((lidar.shape[0], 2)))
                r_sum_list.append(jnp.full((lidar.shape[0],), float(self._params["car_radius"])))

        if len(obj_pos_list) == 0:
            return theta_g

        obj_pos = jnp.concatenate(obj_pos_list, axis=0)
        obj_vel = jnp.concatenate(obj_vel_list, axis=0)
        r_sum = jnp.concatenate(r_sum_list, axis=0)

        diff = obj_pos[None, :, :] - r_pos[:, None, :]
        pBx = c[:, None] * diff[:, :, 0] + s[:, None] * diff[:, :, 1]
        pBy = -s[:, None] * diff[:, :, 0] + c[:, None] * diff[:, :, 1]
        vBx = c[:, None] * obj_vel[None, :, 0] + s[:, None] * obj_vel[None, :, 1]
        vBy = -s[:, None] * obj_vel[None, :, 0] + c[:, None] * obj_vel[None, :, 1]

        dist = jnp.sqrt(pBx ** 2 + pBy ** 2)
        mask = dist <= float(self._params.get("comm_radius", 5.0))

        sin_beta = jnp.clip(r_sum[None, :] / (dist + 1e-6), 0.0, 1.0)
        beta = jnp.arcsin(sin_beta)
        theta = jnp.arctan2(pBy, pBx)

        n_samples = int(self._params.get("heading_n_samples", 33))
        theta_u = jnp.linspace(-jnp.pi, jnp.pi, n_samples)
        sin_u = jnp.sin(theta_u)
        cos_u = jnp.cos(theta_u)

        num = r_v[:, None, None] * sin_u[None, None, :] - vBy[:, :, None]
        den = r_v[:, None, None] * cos_u[None, None, :] - vBx[:, :, None]
        theta_v = jnp.arctan2(num, den)

        ang_diff = jnp.arctan2(jnp.sin(theta_v - theta[:, :, None]), jnp.cos(theta_v - theta[:, :, None]))
        in_cone = (jnp.abs(ang_diff) <= beta[:, :, None]) & mask[:, :, None]
        free = ~jnp.any(in_cone, axis=1)

        dist_u = jnp.abs(
            jnp.arctan2(jnp.sin(theta_u[None, :] - theta_g[:, None]), jnp.cos(theta_u[None, :] - theta_g[:, None]))
        )
        dist_u = jnp.where(free, dist_u, jnp.inf)
        has_free = jnp.any(free, axis=1)
        idx = jnp.argmin(dist_u, axis=1)
        theta_d = jnp.where(has_free, theta_u[idx], theta_g)
        return theta_d

    def render_video(self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 80, **kwargs):
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=1,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            n_ped=self.params["n_ped"],
            r_ped=self.params["ped_radius"],
            dt=self.dt,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs,
        )

    def _compute_inv_ttc(self, rel_pos: Array, rel_vel: Array, r_sum: Array) -> Array:
        return compute_inv_ttc(rel_pos, rel_vel, r_sum, eps=self._params.get("ttc_eps", 1e-6))

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        normalize_edges = bool(self._params.get("normalize_edges", False))
        if normalize_edges:
            comm_scale = jnp.maximum(jnp.asarray(self._params["comm_radius"]), 1e-6)
            v_max = jnp.maximum(jnp.asarray(self._params["v_max"]), 1e-6)
            ped_v_max = self._params.get(
                "ped_v_max",
                self._params.get("ped_speed_pref_max", self._params.get("ped_speed_pref", self._params["v_max"])),
            )
            v_scale = jnp.maximum(jnp.asarray(ped_v_max), v_max)
            ttc_scale = jnp.maximum(jnp.asarray(self._params["ttc_threshold"]), 1e-6)

        agent_pos = state.agent[:, :2]

        # robot-ped
        ped_pos = state.ped[:, :2]
        pos_diff = agent_pos[:, None, :] - ped_pos[None, :, :]
        ped_v = state.ped[:, 2:4]
        agent_v = jnp.stack(
            [jnp.cos(state.agent[:, 2]) * state.agent[:, 3], jnp.sin(state.agent[:, 2]) * state.agent[:, 3]],
            axis=-1,
        )
        rel_vel = agent_v[:, None, :] - ped_v[None, :, :]
        r_sum = self._params["car_radius"] + self._params["ped_radius"]
        inv_ttc = self._compute_inv_ttc(pos_diff, rel_vel, r_sum)
        if normalize_edges:
            pos_feat = pos_diff / comm_scale
            vel_feat = rel_vel / v_scale
            inv_ttc_feat = jnp.clip(inv_ttc * ttc_scale, 0.0, 1.0)
            state_diff = jnp.concatenate([pos_feat, vel_feat, inv_ttc_feat[:, :, None]], axis=-1)
        else:
            state_diff = jnp.concatenate([pos_diff, rel_vel, inv_ttc[:, :, None]], axis=-1)
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        agent_ped_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.array([0])
        id_ped = jnp.arange(1, 1 + self._params["n_ped"])
        agent_ped_edges = EdgeBlock(state_diff, agent_ped_mask, id_agent, id_ped)

        # robot-obs (lidar)
        n_hits = self._params["n_rays"]
        id_obs = jnp.arange(1 + self._params["n_ped"], 1 + self._params["n_ped"] + n_hits)
        lidar_pos = agent_pos[0, :] - lidar_data[:, :2]
        agent_v0 = agent_v[0]
        rel_vel_obs = agent_v0[None, :] - jnp.zeros_like(lidar_pos)
        inv_ttc_obs = self._compute_inv_ttc(lidar_pos, rel_vel_obs, self._params["car_radius"])
        if normalize_edges:
            lidar_pos_feat = lidar_pos / comm_scale
            lidar_vel_feat = rel_vel_obs / v_scale
            inv_ttc_obs_feat = jnp.clip(inv_ttc_obs * ttc_scale, 0.0, 1.0)
            lidar_feats = jnp.concatenate([lidar_pos_feat, lidar_vel_feat, inv_ttc_obs_feat[:, None]], axis=-1)
        else:
            lidar_feats = jnp.concatenate([lidar_pos, rel_vel_obs, inv_ttc_obs[:, None]], axis=-1)
        lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
        active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
        agent_obs_mask = jnp.logical_and(jnp.ones((1, n_hits)), active_lidar[None, :])
        agent_obs_edges = EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent, id_obs)

        return [agent_ped_edges, agent_obs_edges]

    def control_affine_dyn(self, state: State):
        return self._dynamics.control_affine_dyn(state)

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        # GraphsTuple is padded with one dummy node; forward_graph state is unpadded.
        # Keep state/node_type lengths aligned for masking and padded-edge indexing.
        n_graph_nodes = graph.node_type.shape[0]
        n_state_nodes = state.shape[0]
        if n_state_nodes < n_graph_nodes:
            state = jnp.concatenate([state, graph.states[n_state_nodes:n_graph_nodes]], axis=0)
        elif n_state_nodes > n_graph_nodes:
            state = state[:n_graph_nodes]

        node_type = graph.node_type
        agent_mask = (node_type == RobotPedEnv.AGENT)
        ped_mask = (node_type == RobotPedEnv.PED)

        pos = state[:, :2]
        vel = jnp.zeros_like(pos)
        theta = state[:, 2]
        v = state[:, 3]
        agent_v = jnp.stack([jnp.cos(theta) * v, jnp.sin(theta) * v], axis=-1)
        vel = vel + agent_mask[:, None] * agent_v
        vel = vel + ped_mask[:, None] * state[:, 2:4]

        rel_pos = pos[graph.receivers] - pos[graph.senders]
        rel_vel = vel[graph.receivers] - vel[graph.senders]
        recv_type = node_type[graph.receivers]
        r_sum = jnp.where(
            recv_type == RobotPedEnv.PED,
            self._params["car_radius"] + self._params["ped_radius"],
            self._params["car_radius"],
        )
        inv_ttc = self._compute_inv_ttc(rel_pos, rel_vel, r_sum)
        if self._params.get("normalize_edges", False):
            comm_scale = jnp.maximum(jnp.asarray(self._params["comm_radius"]), 1e-6)
            v_max = jnp.maximum(jnp.asarray(self._params["v_max"]), 1e-6)
            ped_v_max = self._params.get(
                "ped_v_max",
                self._params.get("ped_speed_pref_max", self._params.get("ped_speed_pref", self._params["v_max"])),
            )
            v_scale = jnp.maximum(jnp.asarray(ped_v_max), v_max)
            ttc_scale = jnp.maximum(jnp.asarray(self._params["ttc_threshold"]), 1e-6)
            rel_pos = rel_pos / comm_scale
            rel_vel = rel_vel / v_scale
            inv_ttc = jnp.clip(inv_ttc * ttc_scale, 0.0, 1.0)
        edge_feats = jnp.concatenate([rel_pos, rel_vel, inv_ttc[:, None]], axis=-1)
        return graph._replace(edges=edge_feats, states=state)

        ## Graph construction : define node features, edge features, and edge connectivity based on current state
    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        n_hits = self._params["n_rays"]
        n_nodes = 1 + self._params["n_ped"] + n_hits
        
        # node features: one-hot + kinematics (goal is encoded in robot node features)
        node_feats = jnp.zeros((n_nodes, self.node_dim))
        node_feats = node_feats.at[0, 0].set(1)  # robot
        node_feats = node_feats.at[1 : 1 + self._params["n_ped"], 2].set(1)  # ped
        node_feats = node_feats.at[-n_hits:, 3].set(1)  # obs
        
        # node types: one of {AGENT=0, GOAL=1, PED=2, OBS=3} for indexing type-specific states/edges
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[1 : 1 + self._params["n_ped"]].set(RobotPedEnv.PED)
        node_type = node_type.at[-n_hits:].set(RobotPedEnv.OBS)

        # get_lidar 得到每条激光射线的命中点坐标 (x,y) 拼 3 个 0 变成 5 维（和 state 维度对齐）
        lidar_data = get_lidar(state.agent[0, :2], state.obstacle, self._params["n_rays"], self._params["comm_radius"])
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros((lidar_data.shape[0], 3))], axis=-1)

        # feature layout: [onehot4, x, y, vx, vy, cosθ, sinθ, v, w, goal_dx, goal_dy, cos(err), sin(err)]
        # optional normalization for node features only (states/edges remain in meters)
        normalize_nodes = bool(self._params.get("normalize_nodes", False))
        if normalize_nodes:
            area_scale = jnp.maximum(jnp.asarray(self.area_size), 1e-6)
            comm_scale = jnp.maximum(jnp.asarray(self._params["comm_radius"]), 1e-6)
            v_max = jnp.maximum(jnp.asarray(self._params["v_max"]), 1e-6)
            w_max = jnp.maximum(jnp.asarray(self._params["w_max"]), 1e-6)
            ped_v_max = self._params.get(
                "ped_v_max",
                self._params.get("ped_speed_pref_max", self._params.get("ped_speed_pref", self._params["v_max"])),
            )
            ped_v_max = jnp.maximum(jnp.asarray(ped_v_max), 1e-6)
        # robot
        r_pos = state.agent[0, :2]
        r_theta = state.agent[0, 2]
        r_v = state.agent[0, 3]
        r_w = state.agent[0, 4]
        r_vx = r_v * jnp.cos(r_theta)
        r_vy = r_v * jnp.sin(r_theta)
        g_pos = state.goal[0, :2]
        g_dxdy = g_pos - r_pos
        g_theta = jnp.arctan2(g_dxdy[1], g_dxdy[0])
        theta_err = jnp.arctan2(jnp.sin(g_theta - r_theta), jnp.cos(g_theta - r_theta))
        r_pos_feat = r_pos / area_scale if normalize_nodes else r_pos
        r_vx_feat = r_vx / v_max if normalize_nodes else r_vx
        r_vy_feat = r_vy / v_max if normalize_nodes else r_vy
        r_v_feat = r_v / v_max if normalize_nodes else r_v
        r_w_feat = r_w / w_max if normalize_nodes else r_w
        g_dxdy_feat = g_dxdy / area_scale if normalize_nodes else g_dxdy
        node_feats = node_feats.at[0, 4:16].set(
            jnp.array([
                r_pos_feat[0], r_pos_feat[1],
                r_vx_feat, r_vy_feat,
                jnp.cos(r_theta), jnp.sin(r_theta),
                r_v_feat, r_w_feat,
                g_dxdy_feat[0], g_dxdy_feat[1],
                jnp.cos(theta_err), jnp.sin(theta_err),
            ])
        )

        # pedestrians (position relative to robot + velocity)
        if self._params["n_ped"] > 0:
            ped_pos = state.ped[:, :2]
            ped_vel = state.ped[:, 2:4]
            ped_pos_rel = ped_pos - r_pos[None, :]
            ped_pos_feat = ped_pos_rel / comm_scale if normalize_nodes else ped_pos_rel
            ped_vel_feat = ped_vel / ped_v_max if normalize_nodes else ped_vel
            ped_feat = jnp.concatenate([ped_pos_feat, ped_vel_feat], axis=-1)
            node_feats = node_feats.at[1 : 1 + self._params["n_ped"], 4:8].set(ped_feat)

        # obstacles (lidar hits: position relative to robot)
        lidar_pos_rel = lidar_data[:, :2] - r_pos[None, :]
        lidar_pos_feat = lidar_pos_rel / comm_scale if normalize_nodes else lidar_pos_rel
        node_feats = node_feats.at[-n_hits:, 4:6].set(lidar_pos_feat)

        edge_blocks = self.edge_blocks(state, lidar_data)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.ped, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        return self._dynamics.state_lim()

    def action_lim(self) -> Tuple[Action, Action]:
        return self._dynamics.action_lim()

    def _discretize_action(self, action: Action) -> Action:
        return self._dynamics.discretize_action(action)

    def _apply_action_mode(self, agent_states: AgentState, action: Action) -> Action:
        return self._dynamics.apply_action_mode(agent_states, action)

    def _empty_train_cbf_info(self) -> dict:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return {
            "cbf_h": zero,
            "cbf_a": zero,
            "cbf_b": zero,
            "cbf_c": zero,
            "cbf_active": zero,
            "v_policy": zero,
            "v_safe": zero,
            "r_cbf": zero,
        }

    def _compute_train_cbf_barrier(self, graph: EnvGraphsTuple) -> tuple[Array, Array]:
        eps = float(self._params.get("train_cbf_eps", 1e-6))
        agent_states = graph.type_states(type_idx=0, n_type=1)
        agent_pos = agent_states[0, :2]

        h_terms = []
        grad_terms = []

        if bool(self._params.get("train_cbf_use_ped", True)) and self._params["n_ped"] > 0:
            ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
            ped_pos = ped_states[:, :2]
            rel = agent_pos[None, :] - ped_pos
            dist = jnp.linalg.norm(rel, axis=-1)
            h_ped = dist - (self._params["car_radius"] + self._params["ped_radius"])
            grad_ped = rel / (dist[:, None] + eps)
            h_terms.append(h_ped)
            grad_terms.append(grad_ped)

        if bool(self._params.get("train_cbf_use_lidar", True)):
            lidar = get_lidar(
                agent_pos,
                graph.env_states.obstacle,
                self._params["n_rays"],
                self._params["comm_radius"],
            )
            if lidar.shape[0] > 0:
                rel_obs = agent_pos[None, :] - lidar
                dist_obs = jnp.linalg.norm(rel_obs, axis=-1)
                h_obs = dist_obs - self._params["car_radius"]
                grad_obs = rel_obs / (dist_obs[:, None] + eps)
                h_terms.append(h_obs)
                grad_terms.append(grad_obs)

        if bool(self._params.get("train_cbf_use_walls", True)):
            x = agent_pos[0]
            y = agent_pos[1]
            car_r = float(self._params["car_radius"])
            h_wall = jnp.array(
                [
                    x - car_r,
                    self.area_size - x - car_r,
                    y - car_r,
                    self.area_size - y - car_r,
                ],
                dtype=jnp.float32,
            )
            grad_wall = jnp.array(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=jnp.float32
            )
            h_terms.append(h_wall)
            grad_terms.append(grad_wall)

        if len(h_terms) == 0:
            return jnp.array(1.0, dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32)

        h_all = jnp.concatenate(h_terms, axis=0)
        grad_all = jnp.concatenate(grad_terms, axis=0)
        min_idx = jnp.argmin(h_all)
        return h_all[min_idx], grad_all[min_idx]

    def _filter_train_cbf_action(self, graph: EnvGraphsTuple, action_cmd: Action) -> tuple[Action, dict]:
        info = self._empty_train_cbf_info()
        action_cmd = self.clip_action(action_cmd)
        if not self._is_vw_command_mode():
            return action_cmd, info
        if not bool(self._params.get("train_cbf_filter", False)):
            return action_cmd, info

        eps = float(self._params.get("train_cbf_eps", 1e-6))
        alpha = float(self._params.get("train_cbf_alpha", 1.0))
        sigma = float(self._params.get("train_cbf_sigma", 0.3))
        sigma2 = max(sigma * sigma, eps)

        agent_states = graph.type_states(type_idx=0, n_type=1)
        h, grad_h = self._compute_train_cbf_barrier(graph)

        w_policy = action_cmd[0, 0]
        v_policy = action_cmd[0, 1]

        # Include w_cmd via half-step heading in one-step translational direction.
        theta_eff = agent_states[0, 2] + 0.5 * self.dt * w_policy
        heading = jnp.array([jnp.cos(theta_eff), jnp.sin(theta_eff)], dtype=jnp.float32)

        a = jnp.dot(grad_h, heading)
        b = -alpha * h
        c = a * v_policy - b

        v_safe = jnp.where(
            c >= 0.0,
            v_policy,
            v_policy + (b - a * v_policy) * a / (a * a + eps),
        )
        v_safe = jnp.clip(v_safe, self._params["v_min"], self._params["v_max"])

        action_safe = action_cmd.at[0, 1].set(v_safe)
        action_safe = self.clip_action(action_safe)

        dv_sq = (v_policy - v_safe) ** 2
        cbf_penalty = jnp.maximum(-c, 0.0) + (1.0 - jnp.exp(-dv_sq / sigma2))
        r_cbf = -cbf_penalty

        info = {
            "cbf_h": h,
            "cbf_a": a,
            "cbf_b": b,
            "cbf_c": c,
            "cbf_active": (c < 0.0).astype(jnp.float32),
            "v_policy": v_policy,
            "v_safe": v_safe,
            "r_cbf": r_cbf,
        }
        return action_safe, info

    def _pack_action(
        self, w_des: Array, v_des: Array, delta_w: Array, delta_v: Array
    ) -> Action:
        if self._is_vw_command_mode():
            return self.clip_action(jnp.stack([w_des, v_des], axis=-1))
        return self.clip_action(jnp.stack([delta_w, delta_v], axis=-1))

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent_states = graph.type_states(type_idx=0, n_type=1)
        goal_states = graph.env_states.goal
        pos_diff = agent_states[:, :2] - goal_states[:, :2]

        k_omega = 1.0
        k_omega_d = 0.3
        k_v = 2.3
        k_a = 2.5

        avoid_mode = self._params.get("avoid_mode", "none")

        dist_to_goal = jnp.linalg.norm(pos_diff, axis=-1)
        slow_radius = float(self._params.get("slow_radius", 1.0))
        stop_radius = float(self._params.get("stop_tolerance", 0.025))

        def apply_goal_slowdown(v_des: Array, w_des: Array) -> tuple[Array, Array]:
            # Scale speed near the goal and stop inside stop_radius.
            scale = jnp.clip(dist_to_goal / (slow_radius + 1e-6), a_min=0.0, a_max=1.0)
            v_des = v_des * scale
            v_des = jnp.where(dist_to_goal <= stop_radius, 0.0, v_des)
            w_des = jnp.where(dist_to_goal <= stop_radius, 0.0, w_des)
            return v_des, w_des

        if avoid_mode == "cbf":
            # PID-like desired v,w from goal tracking
            theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
            theta = agent_states[:, 2] % (2 * jnp.pi)
            theta_diff = jnp.arctan2(jnp.sin(theta_t - theta), jnp.cos(theta_t - theta))
            vx = agent_states[:, 3] * jnp.cos(theta)
            vy = agent_states[:, 3] * jnp.sin(theta)
            dx = pos_diff[:, 0]
            dy = pos_diff[:, 1]
            dtheta_t = (dx * vy - dy * vx) / (dx * dx + dy * dy + 1e-6)
            w_des = (k_omega * theta_diff + k_omega_d * dtheta_t) / (1.0 + k_omega_d)
            v_des = k_v * jnp.linalg.norm(pos_diff, axis=-1)
            v_des = jnp.clip(v_des, a_min=self._params["v_min"], a_max=self._params["v_max"])
            v_des, w_des = apply_goal_slowdown(v_des, w_des)

            v = agent_states[:, 3]
            w = agent_states[:, 4]
            delta_w_ref = k_omega * (w_des - w)
            delta_v_ref = k_a * (v_des - v)

            # Build obstacle set: pedestrians + lidar hits
            agent_pos = agent_states[:, :2]
            ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
            ped_pos = ped_states[:, :2]
            ped_vel = ped_states[:, 2:4]
            lidar = get_lidar(
                agent_pos[0, :], graph.env_states.obstacle, self._params["n_rays"], self._params["comm_radius"]
            )
            obs_pos = jnp.concatenate([ped_pos, lidar], axis=0)
            obs_vel = jnp.concatenate([ped_vel, jnp.zeros((lidar.shape[0], 2))], axis=0)
            ped_r = self._params["ped_radius"]
            car_r = self._params["car_radius"]
            obs_r = jnp.concatenate(
                [
                    jnp.full((self._params["n_ped"],), car_r + ped_r),
                    jnp.full((self._params["n_rays"],), car_r),
                ],
                axis=0,
            )

            cbf_k = int(self._params.get("cbf_k", 3))
            n_obs_total = self._params["n_ped"] + self._params["n_rays"]
            cbf_k = min(cbf_k, n_obs_total)
            h1_gain = float(self._params.get("cbf_h1_gain", 5.0))

            def h1(state: Array) -> Array:
                pos = state[:2]
                theta_s = state[2]
                v_s = state[3]
                agent_vel = jnp.array([jnp.cos(theta_s) * v_s, jnp.sin(theta_s) * v_s])
                dist_sq = ((pos - obs_pos) ** 2).sum(axis=-1)
                k_idx = jnp.argsort(dist_sq)[:cbf_k]
                k_pos = obs_pos[k_idx]
                k_vel = obs_vel[k_idx]
                k_r = obs_r[k_idx]
                k_xdiff = pos - k_pos
                k_vdiff = agent_vel - k_vel
                k_h0 = (k_xdiff ** 2).sum(axis=-1) - (k_r ** 2)
                k_h0_dot = 2.0 * (k_xdiff * k_vdiff).sum(axis=-1)
                return k_h0_dot + h1_gain * k_h0

            agent_state = agent_states[0]
            h = h1(agent_state)  # (k,)
            h_x = jax.jacfwd(h1)(agent_state)  # (k, 5)
            dyn_f, dyn_g = self.control_affine_dyn(agent_states)
            dyn_f = dyn_f[0]
            dyn_g = dyn_g[0]
            Lf_h = jnp.einsum("k d, d -> k", h_x, dyn_f)
            Lg_h = jnp.einsum("k d, d u -> k u", h_x, dyn_g)

            alpha = float(self._params.get("cbf_alpha", 1.0))
            relax_penalty = float(self._params.get("cbf_relax_penalty", 1e3))

            u_ref = jnp.concatenate([delta_w_ref, delta_v_ref], axis=-1)
            a_w_max = float(self._params.get("a_w_max", self._params["delta_omega_max"]))
            a_v_max = float(self._params.get("a_v_max", self._params["delta_v_max"]))
            u_lb = jnp.array([-a_w_max, -a_v_max], dtype=jnp.float32)
            u_ub = jnp.array([a_w_max, a_v_max], dtype=jnp.float32)

            H = jnp.eye(2 + cbf_k, dtype=jnp.float32)
            H = H.at[-cbf_k:, -cbf_k:].set(10.0)
            g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(cbf_k, dtype=jnp.float32)])
            C = -jnp.concatenate([Lg_h, jnp.eye(cbf_k, dtype=jnp.float32)], axis=1)
            b = Lf_h + alpha * h

            r_lb = jnp.zeros((cbf_k,), dtype=jnp.float32)
            r_ub = jnp.full((cbf_k,), jnp.inf, dtype=jnp.float32)
            l_box = jnp.concatenate([u_lb, r_lb], axis=0)
            u_box = jnp.concatenate([u_ub, r_ub], axis=0)

            qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
            settings = JaxProxQP.Settings.default()
            settings.max_iter = 100
            settings.dua_gap_thresh_abs = None
            solver = JaxProxQP(qp, settings)
            sol = solver.solve()
            u_opt = sol.x[:2]

            delta_w = u_opt[0:1]
            delta_v = u_opt[1:2]
            w_cmd = jnp.clip(w + delta_w, a_min=self._params["w_min"], a_max=self._params["w_max"])
            v_cmd = jnp.clip(v + delta_v, a_min=self._params["v_min"], a_max=self._params["v_max"])
            return self._pack_action(w_cmd, v_cmd, delta_w, delta_v)

        if avoid_mode == "sfm":
            agent_pos = agent_states[:, :2]
            ped_pos = graph.type_states(type_idx=2, n_type=self._params["n_ped"])[:, :2]

            theta = agent_states[:, 2]
            v = agent_states[:, 3]
            w = agent_states[:, 4]
            vel = jnp.stack([jnp.cos(theta) * v, jnp.sin(theta) * v], axis=-1)

            goal_vec = goal_states[:, :2] - agent_pos
            goal_dir = goal_vec / (jnp.linalg.norm(goal_vec, axis=-1, keepdims=True) + 1e-6)
            v0 = self._params.get("ped_speed_pref", 1.0)
            v_des = v0 * goal_dir
            tau = self._params.get("ped_tau", 0.5)
            f_goal = (v_des - vel) / tau

            diff = agent_pos[:, None, :] - ped_pos[None, :, :]
            dist = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
            n_ij = diff / dist
            A = self._params.get("sfm_A", 2.0)
            B = self._params.get("sfm_B", 0.3)
            r_sum = self._params.get("ped_radius", 0.4) + self._params.get("car_radius", 0.5)
            f_ped = jnp.sum(A * jnp.exp((r_sum - dist) / B) * n_ij, axis=1)

            lidar = get_lidar(
                agent_pos[0, :], graph.env_states.obstacle, self._params["n_rays"], self._params["comm_radius"]
            )
            lidar_vec = agent_pos[:, None, :] - lidar[None, :, :]
            lidar_dist = jnp.linalg.norm(lidar_vec, axis=-1, keepdims=True) + 1e-6
            lidar_dir = lidar_vec / lidar_dist
            f_obs = jnp.sum(
                A * jnp.exp((self._params.get("car_radius", 0.5) - lidar_dist) / B) * lidar_dir, axis=1
            )

            acc = f_goal + f_ped + f_obs
            vel_next = vel + self.dt * acc
            speed_des = jnp.linalg.norm(vel_next, axis=-1)
            theta_t = jnp.arctan2(vel_next[:, 1], vel_next[:, 0])
            theta_diff = jnp.arctan2(jnp.sin(theta_t - theta), jnp.cos(theta_t - theta))
            w_des = k_omega * theta_diff
            speed_des, w_des = apply_goal_slowdown(speed_des, w_des)
            delta_w = k_omega * (w_des - w)
            delta_v = k_a * (speed_des - v)
            return self._pack_action(w_des, speed_des, delta_w, delta_v)

        theta_t = jnp.arctan2(-pos_diff[:, 1], -pos_diff[:, 0]) % (2 * jnp.pi)
        theta = agent_states[:, 2] % (2 * jnp.pi)
        # Continuous wrapped angle error to avoid sign flips near +/- pi
        theta_diff = jnp.arctan2(jnp.sin(theta_t - theta), jnp.cos(theta_t - theta))
        # Line-of-sight angle rate (goal is static), used as derivative term
        vx = agent_states[:, 3] * jnp.cos(theta)
        vy = agent_states[:, 3] * jnp.sin(theta)
        dx = pos_diff[:, 0]
        dy = pos_diff[:, 1]
        dtheta_t = (dx * vy - dy * vx) / (dx * dx + dy * dy + 1e-6)
        # PD on heading error with implicit omega in error derivative
        w_des = (k_omega * theta_diff + k_omega_d * dtheta_t) / (1.0 + k_omega_d)
        v_des = k_v * jnp.linalg.norm(pos_diff, axis=-1)
        v_des = jnp.clip(v_des, a_min=self._params["v_min"], a_max=self._params["v_max"])
        v_des, w_des = apply_goal_slowdown(v_des, w_des)

        v = agent_states[:, 3]
        w = agent_states[:, 4]
        delta_w = k_omega * (w_des - w)
        delta_v = k_a * (v_des - v)
        return self._pack_action(w_des, v_des, delta_w, delta_v)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        agent_states = graph.type_states(type_idx=0, n_type=1)
        ped_states = graph.type_states(type_idx=2, n_type=self._params["n_ped"])
        obs_states = graph.type_states(type_idx=3, n_type=self._params["n_rays"])
        action = self.clip_action(action)
        action, _ = self._filter_train_cbf_action(graph, action)
        action = self._apply_action_mode(agent_states, action)
        stop_mask = self.stop_mask(graph)
        next_agent_states = self.agent_step_euler(agent_states, action, stop_mask)
        next_states = jnp.concatenate([next_agent_states, ped_states, obs_states], axis=0)
        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        ped_pos = graph.type_states(type_idx=2, n_type=self._params["n_ped"])[:, :2]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        safe_ped = jnp.greater(dist, self._params["car_radius"] * 3)
        safe_ped = jnp.min(safe_ped, axis=1)
        safe_obs = jnp.logical_not(inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2))
        return jnp.logical_and(safe_ped, safe_obs)

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        ped_pos = graph.type_states(type_idx=2, n_type=self._params["n_ped"])[:, :2]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        unsafe_ped = jnp.less(dist, self._params["car_radius"] * 2 + self._params["ped_radius"])
        unsafe_ped = jnp.max(unsafe_ped, axis=1)
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)
        return jnp.logical_or(unsafe_ped, unsafe_obs)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        ped_pos = graph.type_states(type_idx=2, n_type=self._params["n_ped"])[:, :2]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - ped_pos[None, :, :], axis=-1)
        collision_ped = jnp.less(dist, self._params["car_radius"] + self._params["ped_radius"])
        collision_ped = jnp.max(collision_ped, axis=1)
        collision_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])
        return jnp.logical_or(collision_ped, collision_obs)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) <= self._params["goal_tolerance"]
        return reach

    def stop_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=1)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        stop = jnp.linalg.norm(agent_pos - goal_pos, axis=1) <= self._params["stop_tolerance"]
        return stop

    def _refresh_ped_goals(
        self,
        ped_states: State,
        ped_goal: State,
        ped_key: Array,
        obstacles: Obstacle,
        ped_active: Array | None = None,
    ) -> Tuple[State, Array]:
        pos = ped_states[:, :2]
        dist = jnp.linalg.norm(pos - ped_goal, axis=-1)
        reached = dist < float(self._params["ped_goal_replan_dist"])
        if ped_active is not None:
            reached = jnp.logical_and(reached, ped_active)

        def sample_goal(key):
            def cond_fn(carry):
                _, goal = carry
                return inside_obstacles(goal, obstacles, r=self._params["ped_radius"])

            def body_fn(carry):
                key_inner, _ = carry
                key_inner, sub = jr.split(key_inner)
                goal = jr.uniform(sub, (2,), minval=0, maxval=self.area_size)
                return key_inner, goal

            key, sub = jr.split(key)
            goal0 = jr.uniform(sub, (2,), minval=0, maxval=self.area_size)
            key, goal = jax.lax.while_loop(cond_fn, body_fn, (key, goal0))
            return key, goal

        def update_one(p_key, g_old, r_flag):
            key_new, g_new = jax.lax.cond(
                r_flag,
                lambda k: sample_goal(k),
                lambda k: (k, g_old),
                p_key,
            )
            return key_new, g_new

        key_new, goal_new = jax.vmap(update_one)(ped_key, ped_goal, reached)
        return goal_new, key_new
