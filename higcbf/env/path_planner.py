"""
Path planning and pure-pursuit lookahead for the single robot.

Extracted from robot_ped_env.py to allow independent testing.

Classes
-------
PathPlanner
    Holds params and area_size; provides plan(), project(), interpolate(),
    lookahead_dist() and get_target().
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..utils.typing import Array
from .obstacle import Obstacle
from .utils import inside_obstacles


class PathPlanner:
    """
    A* path planner + pure-pursuit lookahead on a grid map.

    Parameters
    ----------
    params   : env params dict (reads astar_*, lookahead_*, path_* keys)
    area_size: float – side length of the square environment [m]
    """

    def __init__(self, params: dict, area_size: float) -> None:
        self._params = params
        self._area_size = area_size

    # ------------------------------------------------------------------
    # Fallback / empty path
    # ------------------------------------------------------------------

    def empty_plan(self, goal_xy: Array) -> Tuple[Array, Array, Array]:
        """
        Return an empty path (or a single-waypoint fallback to the goal).

        Returns
        -------
        waypoints : (max_wp, 2)
        valid     : (max_wp,) bool
        path_len  : (1,) int32
        """
        max_wp = int(self._params.get("path_max_waypoints", 128))
        waypoints = jnp.zeros((max_wp, 2), dtype=jnp.float32)
        valid = jnp.zeros((max_wp,), dtype=bool)
        if bool(self._params.get("astar_fallback_to_goal", True)):
            waypoints = waypoints.at[0].set(goal_xy)
            valid = valid.at[0].set(True)
            path_len = jnp.array([1], dtype=jnp.int32)
        else:
            path_len = jnp.array([0], dtype=jnp.int32)
        return waypoints, valid, path_len

    # ------------------------------------------------------------------
    # A* planning
    # ------------------------------------------------------------------

    def plan(
        self,
        obstacles: Obstacle,
        start_xy: Array,
        goal_xy: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Plan a collision-free path from start_xy to goal_xy using A*.

        Returns
        -------
        waypoints : (max_wp, 2)
        valid     : (max_wp,) bool
        path_len  : (1,) int32
        """
        grid_size = int(self._params.get("astar_grid_size", 64))
        max_expand = int(self._params.get("astar_max_expand", 4096))
        max_wp = int(self._params.get("path_max_waypoints", 128))
        allow_diag = bool(self._params.get("astar_allow_diag", True))
        n_nodes = grid_size * grid_size
        cell_size = float(self._area_size) / float(grid_size)

        grid = jnp.arange(grid_size, dtype=jnp.float32)
        centers_1d = (grid + 0.5) * cell_size
        xs, ys = jnp.meshgrid(centers_1d, centers_1d, indexing="xy")
        centers = jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)

        start_ix = jnp.clip(jnp.floor(start_xy[0] / cell_size).astype(jnp.int32), 0, grid_size - 1)
        start_iy = jnp.clip(jnp.floor(start_xy[1] / cell_size).astype(jnp.int32), 0, grid_size - 1)
        goal_ix = jnp.clip(jnp.floor(goal_xy[0] / cell_size).astype(jnp.int32), 0, grid_size - 1)
        goal_iy = jnp.clip(jnp.floor(goal_xy[1] / cell_size).astype(jnp.int32), 0, grid_size - 1)
        start_idx = start_iy * grid_size + start_ix
        goal_idx = goal_iy * grid_size + goal_ix

        occ = inside_obstacles(centers, obstacles, r=float(self._params["car_radius"]))
        occ = occ.at[start_idx].set(False)
        occ = occ.at[goal_idx].set(False)

        dx = jnp.array([1, -1, 0, 0, 1, 1, -1, -1], dtype=jnp.int32)
        dy = jnp.array([0, 0, 1, -1, 1, -1, 1, -1], dtype=jnp.int32)
        dcost = jnp.array(
            [1.0, 1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)],
            dtype=jnp.float32,
        )
        diag = jnp.array([False, False, False, False, True, True, True, True])
        move_mask = jnp.where(diag, jnp.array(allow_diag), jnp.array(True))

        g_goal = centers[goal_idx]
        h = jnp.linalg.norm(centers - g_goal[None, :], axis=-1)
        inf = jnp.array(1e9, dtype=jnp.float32)
        g = jnp.full((n_nodes,), inf, dtype=jnp.float32)
        f = jnp.full((n_nodes,), inf, dtype=jnp.float32)
        parent = jnp.full((n_nodes,), -1, dtype=jnp.int32)
        open_mask = jnp.zeros((n_nodes,), dtype=bool)
        closed_mask = jnp.zeros((n_nodes,), dtype=bool)

        g = g.at[start_idx].set(0.0)
        f = f.at[start_idx].set(h[start_idx])
        parent = parent.at[start_idx].set(start_idx)
        open_mask = open_mask.at[start_idx].set(True)

        def expand_neighbor(ii, carry):
            g_i, f_i, p_i, o_i, current, cx, cy, g_cur, closed_cur = carry
            nx = cx + dx[ii]
            ny = cy + dy[ii]
            in_bounds = (nx >= 0) & (nx < grid_size) & (ny >= 0) & (ny < grid_size)
            nb = ny * grid_size + nx
            nb = jnp.clip(nb, 0, n_nodes - 1)
            active = in_bounds & move_mask[ii] & (~occ[nb]) & (~closed_cur[nb])
            cand_g = g_cur + dcost[ii]
            better = cand_g < g_i[nb]
            do_update = active & better
            g_i = g_i.at[nb].set(jnp.where(do_update, cand_g, g_i[nb]))
            f_i = f_i.at[nb].set(jnp.where(do_update, cand_g + h[nb], f_i[nb]))
            p_i = p_i.at[nb].set(jnp.where(do_update, current, p_i[nb]))
            o_i = o_i.at[nb].set(jnp.where(do_update, True, o_i[nb]))
            return g_i, f_i, p_i, o_i, current, cx, cy, g_cur, closed_cur

        def search_iter(_, carry):
            g_i, f_i, p_i, o_i, c_i, found_i, done_i = carry

            def do_search(carry_inner):
                g_s, f_s, p_s, o_s, c_s, found_s, _ = carry_inner
                masked_f = jnp.where(o_s, f_s, inf)
                current = jnp.argmin(masked_f)
                has_open = jnp.logical_and(o_s[current], masked_f[current] < inf)

                def no_open(_):
                    return g_s, f_s, p_s, o_s, c_s, found_s, jnp.array(True)

                def expand_open(_):
                    o_n = o_s.at[current].set(False)
                    c_n = c_s.at[current].set(True)
                    reached = current == goal_idx

                    def done_goal(_):
                        return g_s, f_s, p_s, o_n, c_n, jnp.array(True), jnp.array(True)

                    def do_expand(_):
                        cx = current % grid_size
                        cy = current // grid_size
                        g_cur = g_s[current]
                        g_n, f_n, p_n, o_n2, _, _, _, _, _ = jax.lax.fori_loop(
                            0, 8, expand_neighbor,
                            (g_s, f_s, p_s, o_n, current, cx, cy, g_cur, c_n)
                        )
                        return g_n, f_n, p_n, o_n2, c_n, jnp.array(False), jnp.array(False)

                    return jax.lax.cond(reached, done_goal, do_expand, operand=None)

                return jax.lax.cond(has_open, expand_open, no_open, operand=None)

            return jax.lax.cond(done_i, lambda x: x, do_search, carry)

        g, f, parent, open_mask, closed_mask, found, _ = jax.lax.fori_loop(
            0,
            max_expand,
            search_iter,
            (g, f, parent, open_mask, closed_mask, jnp.array(False), jnp.array(False)),
        )

        def build_path(_):
            rev_nodes = jnp.full((max_wp,), start_idx, dtype=jnp.int32)

            def backtrack(i, carry):
                nodes_i, cur_i, length_i, done_i = carry
                nodes_i = nodes_i.at[i].set(jnp.where(done_i, nodes_i[i], cur_i))
                length_i = jnp.where(done_i, length_i, length_i + 1)
                reached = cur_i == start_idx
                parent_i = parent[cur_i]
                cur_next = jnp.where(jnp.logical_or(done_i, reached), cur_i, parent_i)
                done_next = jnp.logical_or(done_i, reached)
                return nodes_i, cur_next, length_i, done_next

            rev_nodes, _, valid_len, _ = jax.lax.fori_loop(
                0, max_wp, backtrack,
                (rev_nodes, goal_idx, jnp.array(0, dtype=jnp.int32), jnp.array(False))
            )
            idx = jnp.arange(max_wp, dtype=jnp.int32)
            read_idx = jnp.clip(valid_len - 1 - idx, 0, max_wp - 1)
            path_nodes = rev_nodes[read_idx]
            valid = idx < valid_len
            waypoints = centers[path_nodes]
            waypoints = jnp.where(valid[:, None], waypoints, jnp.zeros_like(waypoints))
            last = jnp.maximum(valid_len - 1, 0)
            waypoints = waypoints.at[last].set(goal_xy)
            path_len = jnp.array([valid_len], dtype=jnp.int32)
            return waypoints, valid, path_len

        return jax.lax.cond(
            found, build_path, lambda _: self.empty_plan(goal_xy), operand=None
        )

    # ------------------------------------------------------------------
    # Pure-pursuit lookahead helpers
    # ------------------------------------------------------------------

    def lookahead_dist(self, v_abs: Array) -> Array:
        """Compute speed-adaptive lookahead distance [m]."""
        l0 = float(self._params.get("lookahead_l0", 0.8))
        kv = float(self._params.get("lookahead_kv", 1.2))
        lmin = float(self._params.get("lookahead_min", 0.8))
        lmax = float(self._params.get("lookahead_max", 3.0))
        return jnp.clip(kv * v_abs + l0, a_min=lmin, a_max=lmax)

    def project(
        self,
        path_waypoints: Array,
        path_len: Array,
        agent_xy: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Project agent position onto the path and return arc-length parameter s*.

        Returns
        -------
        s_star   : scalar float – arc length of closest projection
        seg_idx  : scalar int   – index of closest segment
        proj_xy  : (2,)         – closest point on path
        total_len: scalar float – total path length [m]
        """
        eps = float(self._params.get("lookahead_eps", 1e-6))
        pos = agent_xy[0] if agent_xy.ndim == 2 else agent_xy
        n_pts = path_len.squeeze().astype(jnp.int32)
        n_seg = jnp.maximum(n_pts - 1, 0)
        max_seg = path_waypoints.shape[0] - 1
        seg_idx = jnp.arange(max_seg, dtype=jnp.int32)
        active = seg_idx < n_seg

        p0 = path_waypoints[:max_seg]
        p1 = path_waypoints[1:max_seg + 1]
        d = p1 - p0
        len2 = jnp.sum(d * d, axis=-1)
        seg_len = jnp.sqrt(jnp.maximum(len2, 0.0))

        rel = pos[None, :] - p0
        denom = jnp.where(len2 > eps, len2, 1.0)
        t = jnp.clip(jnp.sum(rel * d, axis=-1) / denom, 0.0, 1.0)
        t = jnp.where(active, t, 0.0)
        proj = p0 + t[:, None] * d

        dist2 = jnp.sum((proj - pos[None, :]) ** 2, axis=-1)
        dist2 = jnp.where(active, dist2, jnp.inf)
        best_idx = jnp.argmin(dist2)
        has_seg = n_seg > 0

        seg_len_masked = jnp.where(active, seg_len, 0.0)
        prefix_end = jnp.cumsum(seg_len_masked)
        prefix_start = prefix_end - seg_len_masked
        total_len = prefix_end[-1] if max_seg > 0 else jnp.array(0.0, dtype=jnp.float32)

        s_star = jnp.where(
            has_seg,
            prefix_start[best_idx] + t[best_idx] * seg_len[best_idx],
            jnp.array(0.0, dtype=jnp.float32),
        )
        proj_xy = jnp.where(
            has_seg,
            proj[best_idx],
            jnp.where(n_pts > 0, path_waypoints[0], pos),
        )
        seg_out = jnp.where(has_seg, best_idx, jnp.array(0, dtype=jnp.int32))
        return s_star, seg_out, proj_xy, total_len

    def interpolate(self, path_waypoints: Array, path_len: Array, s_query: Array) -> Array:
        """
        Interpolate path position at arc-length s_query.

        Returns
        -------
        target_xy : (2,)
        """
        eps = float(self._params.get("lookahead_eps", 1e-6))
        n_pts = path_len.squeeze().astype(jnp.int32)
        max_seg = path_waypoints.shape[0] - 1
        n_seg = jnp.maximum(n_pts - 1, 0)
        seg_idx = jnp.arange(max_seg, dtype=jnp.int32)
        active = seg_idx < n_seg

        p0 = path_waypoints[:max_seg]
        p1 = path_waypoints[1:max_seg + 1]
        d = p1 - p0
        seg_len = jnp.linalg.norm(d, axis=-1)
        seg_len_masked = jnp.where(active, seg_len, 0.0)
        prefix_end = jnp.cumsum(seg_len_masked)
        prefix_start = prefix_end - seg_len_masked
        total_len = prefix_end[-1] if max_seg > 0 else jnp.array(0.0, dtype=jnp.float32)
        s = jnp.clip(s_query, a_min=0.0, a_max=total_len)

        resid = s - prefix_start
        clamped = jnp.clip(resid, a_min=0.0, a_max=seg_len)
        err = jnp.abs(resid - clamped)
        err = jnp.where(active, err, jnp.inf)
        best_idx = jnp.argmin(err)
        has_seg = n_seg > 0
        ratio = jnp.where(seg_len[best_idx] > eps, clamped[best_idx] / seg_len[best_idx], 0.0)
        seg_target = p0[best_idx] + ratio * d[best_idx]
        return jnp.where(
            has_seg,
            seg_target,
            jnp.where(n_pts > 0, path_waypoints[0], jnp.zeros((2,), dtype=jnp.float32)),
        )

    def get_target(
        self,
        path_waypoints: Array,
        path_len: Array,
        goal_xy: Array,
        agent_states: Array,   # (1, 5) or (5,)
    ) -> Tuple[Array, dict]:
        """
        Compute the lookahead target on the planned path.

        Falls back to goal_xy when path is unavailable or lookahead is disabled.

        Returns
        -------
        target_xy : (2,)
        dbg       : dict of debug scalars
        """
        use_path = bool(self._params.get("path_reward_enable", True))
        use_lookahead = bool(self._params.get("lookahead_enable", True))
        path_len_scalar = path_len.squeeze().astype(jnp.int32)
        path_available = jnp.logical_and(jnp.array(use_path), path_len_scalar > 0)
        do_lookahead = jnp.logical_and(path_available, jnp.array(use_lookahead))

        pos = agent_states[0, :2] if agent_states.ndim == 2 else agent_states[:2]
        v_abs = jnp.abs(agent_states[0, 3] if agent_states.ndim == 2 else agent_states[3])
        ld = self.lookahead_dist(v_abs)

        s_star, seg_idx, proj_xy, total_len = self.project(path_waypoints, path_len, pos)
        s_goal = jnp.clip(s_star + ld, a_min=0.0, a_max=total_len)
        lookahead_xy = self.interpolate(path_waypoints, path_len, s_goal)
        target_xy = jnp.where(do_lookahead, lookahead_xy, goal_xy)

        dbg = {
            "lookahead_ld": ld.astype(jnp.float32),
            "lookahead_s_star": s_star.astype(jnp.float32),
            "lookahead_s_goal": s_goal.astype(jnp.float32),
            "lookahead_path_len_m": total_len.astype(jnp.float32),
            "lookahead_proj_dist": jnp.linalg.norm(pos - proj_xy).astype(jnp.float32),
            "lookahead_target_dist": jnp.linalg.norm(pos - target_xy).astype(jnp.float32),
            "lookahead_seg_idx": seg_idx.astype(jnp.float32),
            "lookahead_active": do_lookahead.astype(jnp.float32),
        }
        return target_xy, dbg
