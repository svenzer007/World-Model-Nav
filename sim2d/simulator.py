"""Episode-level simulator orchestration for V4 structured dynamic-obstacle data."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .astar import astar_search, path_to_world
from .config import DatasetConfig, action_space_to_list
from .dynamics import (
    disk_collides_with_dynamic,
    disk_collides_with_occupancy,
    minimum_dynamic_clearance,
    minimum_static_clearance,
)
from .expert_policy import DWAExpertPolicy
from .lidar import cast_lidar_scan
from .maps import generate_map
from .obstacles import sample_dynamic_obstacles
from .utils import (
    euclidean,
    build_static_obstacle_index,
    distance_point_to_polyline,
    inflate_occupancy_grid,
    occupied_boundary_cell_centers,
    path_length,
    sample_heading_toward,
    world_to_grid,
)
from .waypoint import compute_local_subgoal


class EpisodeSimulator:
    """Sample environments and roll out expert-controlled navigation episodes."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.expert = DWAExpertPolicy(config.expert_config, config.robot_config)
        self._fixed_map_context = self._build_fixed_map_context() if self.config.map_config.map_yaml else None

    def _build_fixed_map_context(self):
        """Precompute fixed-map planning data shared by every episode attempt."""
        map_bundle = generate_map(self.config.map_config, seed=0, map_index=0)
        inflation = self.config.robot_config.radius + self.config.map_config.inflation_margin
        planning_occupancy = inflate_occupancy_grid(
            map_bundle.occupancy,
            inflation_radius_m=inflation,
            resolution=map_bundle.resolution,
        )
        occupied_points = build_static_obstacle_index(
            occupied_boundary_cell_centers(map_bundle.occupancy, map_bundle.resolution, map_bundle.origin)
        )
        return map_bundle, planning_occupancy, occupied_points

    def generate_episode(self, episode_id: int, seed: int) -> dict[str, object]:
        """Generate one accepted V4 episode, resampling dynamically boring rollouts."""
        rng = np.random.default_rng(seed)
        best_episode: dict[str, object] | None = None
        best_score: tuple[int, int, float, float] = (-1, -1, -1.0, -1.0)
        attempts = max(1, int(self.config.dynamic_interaction_resample_limit))

        for dynamic_attempt in range(attempts):
            candidate_seed = int(rng.integers(0, 2**31 - 1))
            candidate = self._generate_episode_candidate(episode_id=episode_id, seed=candidate_seed)
            candidate["seed"] = int(seed)
            candidate["candidate_seed"] = int(candidate_seed)
            candidate["dynamic_interaction_attempt"] = int(dynamic_attempt)
            stats = candidate["episode_stats"]
            trajectory_quality_ok = self._episode_has_trajectory_quality(candidate)
            score = (
                int(trajectory_quality_ok),
                int(stats["dynamic_event_steps"]),
                float(stats["dynamic_risk_fraction"]),
                float(stats["dynamic_visible_fraction"]),
            )
            if score > best_score:
                best_episode = candidate
                best_score = score
            if trajectory_quality_ok and self._episode_has_dynamic_interaction(candidate):
                candidate["episode_stats"]["dynamic_interaction_accepted"] = True
                candidate["episode_stats"]["dynamic_interaction_fallback"] = False
                return candidate

        if best_episode is None:
            raise RuntimeError("Failed to generate any V4 episode candidate")
        best_episode["episode_stats"]["dynamic_interaction_accepted"] = False
        best_episode["episode_stats"]["dynamic_interaction_fallback"] = True
        return best_episode

    def _generate_episode_candidate(self, episode_id: int, seed: int) -> dict[str, object]:
        """Generate one complete episode dictionary."""
        episode_rng = np.random.default_rng(seed)
        map_bundle, planning_occupancy, occupied_points, start_pose, goal_point, path_rc, path_world, dyn_obstacles = (
            self._sample_episode_context(episode_id, episode_rng)
        )
        path_distance = path_length(path_world)
        direct_distance = euclidean(start_pose[:2], goal_point)
        path_directness_ratio = path_distance / max(direct_distance, 1e-6)
        path_turn_angle_sum = self._path_turn_angle_sum(path_world)

        occupancy = map_bundle.occupancy
        robot_pose = start_pose.copy()
        steps: list[dict[str, object]] = []
        robot_trajectory = [robot_pose.copy()]
        dynamic_trajectories: dict[str, list[np.ndarray]] = {
            obstacle.obstacle_id: [obstacle.position.copy()] for obstacle in dyn_obstacles
        }

        done_reason = "timeout"
        episode_collision_any = False

        for step_index in range(self.config.max_steps):
            waypoint_now = compute_local_subgoal(robot_pose, path_world, self.config.lookahead_distance)
            geometry_now = self._compute_geometry_labels(
                robot_pose=robot_pose,
                waypoint=waypoint_now,
                path_world=path_world,
                occupied_points=occupied_points,
                dynamic_obstacles=dyn_obstacles,
                map_resolution=map_bundle.resolution,
                suffix="t",
            )
            lidar_fields_t = self._compute_lidar_fields(
                robot_pose=robot_pose,
                occupancy=occupancy,
                map_bundle=map_bundle,
                dynamic_obstacles=dyn_obstacles,
                suffix="t",
            )

            decision = self.expert.select_action(
                robot_pose=robot_pose,
                current_subgoal_world=waypoint_now["selected_subgoal_world"],
                path_world=path_world,
                planning_occupancy=planning_occupancy,
                true_occupancy=occupancy,
                resolution=map_bundle.resolution,
                origin=map_bundle.origin,
                occupied_points=occupied_points,
                dynamic_obstacles=dyn_obstacles,
            )

            dynamic_t = [obstacle.snapshot() for obstacle in dyn_obstacles]
            robot_pose_next = self._apply_action(robot_pose, decision.v, decision.omega)
            for obstacle in dyn_obstacles:
                obstacle.step(self.config.robot_config.dt)
            dynamic_t1 = [obstacle.snapshot() for obstacle in dyn_obstacles]
            structured_dynamic = self._compute_structured_dynamic_fields(
                robot_pose_t=robot_pose,
                robot_pose_t1=robot_pose_next,
                dynamic_t=dynamic_t,
                dynamic_t1=dynamic_t1,
            )

            static_collision = disk_collides_with_occupancy(
                robot_pose_next[:2],
                self.config.robot_config.radius,
                occupancy,
                map_bundle.resolution,
                map_bundle.origin,
            )
            dynamic_collision = disk_collides_with_dynamic(
                robot_pose_next[:2],
                self.config.robot_config.radius,
                dyn_obstacles,
            )
            goal_reached = euclidean(robot_pose_next[:2], goal_point) <= self.config.robot_config.goal_tolerance

            collision_t1 = static_collision or dynamic_collision
            episode_collision_any = episode_collision_any or collision_t1
            done = collision_t1 or goal_reached or (step_index == self.config.max_steps - 1)
            if goal_reached:
                done_reason = "goal_reached"
            elif static_collision:
                done_reason = "static_collision"
            elif dynamic_collision:
                done_reason = "dynamic_collision"
            elif done:
                done_reason = "timeout"

            waypoint_next = compute_local_subgoal(robot_pose_next, path_world, self.config.lookahead_distance)
            geometry_next = self._compute_geometry_labels(
                robot_pose=robot_pose_next,
                waypoint=waypoint_next,
                path_world=path_world,
                occupied_points=occupied_points,
                dynamic_obstacles=dyn_obstacles,
                map_resolution=map_bundle.resolution,
                suffix="t1",
            )
            lidar_fields_t1 = self._compute_lidar_fields(
                robot_pose=robot_pose_next,
                occupancy=occupancy,
                map_bundle=map_bundle,
                dynamic_obstacles=dyn_obstacles,
                suffix="t1",
            )

            steps.append(
                {
                    "t": int(step_index),
                    "dt": float(self.config.robot_config.dt),
                    "robot_pose": robot_pose.copy(),
                    "robot_pose_t": robot_pose.copy(),
                    "robot_pose_t1": robot_pose_next.copy(),
                    **lidar_fields_t,
                    "goal_t": waypoint_now["goal_features"],
                    "action_index": int(decision.action_index),
                    "action_name": decision.action_name,
                    "v": float(decision.v),
                    "omega": float(decision.omega),
                    "v_t": float(decision.v),
                    "omega_t": float(decision.omega),
                    **lidar_fields_t1,
                    "goal_t1": waypoint_next["goal_features"],
                    "collision_t1": bool(collision_t1),
                    "dynamic_collision_t1": bool(dynamic_collision),
                    "done_t1": bool(done),
                    "done_reason_t1": done_reason,
                    "dynamic_obstacles_t": dynamic_t,
                    "dynamic_obstacles_t1": dynamic_t1,
                    **structured_dynamic,
                    "nearest_path_index": int(waypoint_now["nearest_path_index"]),
                    "selected_subgoal_world": waypoint_now["selected_subgoal_world"].copy(),
                    **geometry_now,
                    **geometry_next,
                    "dynamic_visible_t": self._dynamic_visible_from_clearance(
                        float(geometry_now["min_dynamic_clearance_t"])
                    ),
                    "progress_to_subgoal": float(
                        geometry_now["distance_to_subgoal_t"] - geometry_next["distance_to_subgoal_t1"]
                    ),
                    "risk_t1": self._risk_from_clearance(float(geometry_next["min_total_clearance_t1"])),
                    "dynamic_risk_t1": self._dynamic_risk_from_clearance(
                        float(geometry_next["min_dynamic_clearance_t1"])
                    ),
                    "expert_debug": decision.expert_debug,
                    "chosen_action_min_clearance": float(decision.chosen_action_min_clearance),
                    "emergency_override_used": bool(decision.emergency_override_used),
                }
            )

            robot_pose = robot_pose_next
            robot_trajectory.append(robot_pose.copy())
            for obstacle in dyn_obstacles:
                dynamic_trajectories[obstacle.obstacle_id].append(obstacle.position.copy())
            if done:
                break

        self._add_causal_world_velocity_fields(steps)
        self._add_dynamic_velocity_targets(steps)
        self._add_collision_horizon_labels(steps, horizons=(3, 5))
        self._add_dynamic_horizon_and_event_labels(steps, horizons=(3, 5))
        dynamic_stats = self._compute_dynamic_episode_stats(steps)
        robot_trajectory_array = np.array(robot_trajectory, dtype=float)
        trajectory_stats = self._compute_trajectory_quality_stats(robot_trajectory_array, path_world, path_distance)

        return {
            "episode_id": int(episode_id),
            "seed": int(seed),
            "schema_version": "v4_dynamic_state_data_3",
            "map_id": map_bundle.map_id,
            "map_type": map_bundle.map_type,
            "occupancy_grid": occupancy.copy(),
            "planning_occupancy": planning_occupancy.copy(),
            "inflation_radius_m": float(self.config.robot_config.radius + self.config.map_config.inflation_margin),
            "inflation_radius_cells": int(
                np.ceil(
                    (self.config.robot_config.radius + self.config.map_config.inflation_margin)
                    / self.config.map_config.resolution
                )
            ),
            "resolution": float(map_bundle.resolution),
            "origin": np.array(map_bundle.origin, dtype=float),
            "world_size": np.array(
                [
                    self.config.map_config.width * self.config.map_config.resolution,
                    self.config.map_config.height * self.config.map_config.resolution,
                ],
                dtype=float,
            ),
            "lidar_config": asdict(self.config.lidar_config),
            "action_space": action_space_to_list(),
            "robot_config": asdict(self.config.robot_config),
            "expert_config": asdict(self.config.expert_config),
            "dynamic_label_config": {
                "dynamic_residual_epsilon": float(self.config.dynamic_residual_epsilon),
                "dynamic_visible_min_beams": int(self.config.dynamic_visible_min_beams),
                "dynamic_risk_tau": float(self.config.dynamic_risk_tau),
                "dynamic_risk_positive_threshold": float(self.config.dynamic_risk_positive_threshold),
                "dynamic_visibility_radius": float(self.config.dynamic_visibility_radius),
                "min_dynamic_visible_fraction": float(self.config.min_dynamic_visible_fraction),
                "min_dynamic_risk_fraction": float(self.config.min_dynamic_risk_fraction),
                "min_dynamic_event_steps": int(self.config.min_dynamic_event_steps),
                "dynamic_interaction_resample_limit": int(self.config.dynamic_interaction_resample_limit),
            },
            "collision_geometry": {
                "static_clearance": "nearest occupied cell center distance minus robot radius and half-cell pad",
                "dynamic_clearance": "nearest dynamic obstacle center distance minus robot and obstacle radii",
                "total_clearance": "minimum of static and dynamic clearance",
                "risk_tau": float(self.config.risk_tau),
                "dynamic_risk_tau": float(self.config.dynamic_risk_tau),
                "dynamic_risk": "1.0 if dynamic clearance <= 0 else exp(-dynamic_clearance / dynamic_risk_tau)",
                "dynamic_residual": "relu(lidar_static_t - lidar_t)",
                "dynamic_visible_t": "min_dynamic_clearance_t < dynamic_visibility_radius",
                "dynamic_event_step_t": (
                    "dynamic_visible_t OR dynamic_risk_t1 >= dynamic_risk_positive_threshold OR "
                    "dynamic_collision_within_3 OR dynamic_collision_within_5"
                ),
            },
            "start_pose": start_pose.copy(),
            "final_goal": goal_point.copy(),
            "astar_path_world": path_world.copy(),
            "astar_path_grid": np.array(path_rc, dtype=np.int32),
            "num_dynamic_obstacles": int(self.config.dynamic_obstacle_config.num_dynamic_obstacles),
            "dynamic_obstacle_ids": [obstacle.obstacle_id for obstacle in dyn_obstacles],
            "dynamic_obstacle_metadata": [obstacle.route_metadata() for obstacle in dyn_obstacles],
            "dynamic_obstacle_routes": [obstacle.route_metadata() for obstacle in dyn_obstacles],
            "steps": steps,
            "episode_stats": {
                "num_steps": int(len(steps)),
                "done_reason": done_reason,
                "success": bool(done_reason == "goal_reached"),
                "collision": bool(episode_collision_any),
                "astar_path_length": float(path_distance),
                "start_goal_distance": float(direct_distance),
                "path_directness_ratio": float(path_directness_ratio),
                "path_turn_angle_sum": float(path_turn_angle_sum),
                "robot_trajectory": robot_trajectory_array,
                "dynamic_trajectories": {
                    obstacle_id: np.array(points, dtype=float) for obstacle_id, points in dynamic_trajectories.items()
                },
                **dynamic_stats,
                **trajectory_stats,
            },
        }

    def _compute_trajectory_quality_stats(
        self,
        robot_trajectory: np.ndarray,
        path_world: np.ndarray,
        astar_path_length: float,
    ) -> dict[str, float]:
        robot_path_length = path_length(robot_trajectory[:, :2]) if len(robot_trajectory) else 0.0
        if len(robot_trajectory) and len(path_world):
            max_deviation = max(
                float(distance_point_to_polyline(point, path_world)) for point in robot_trajectory[:, :2]
            )
        else:
            max_deviation = float("inf")
        return {
            "robot_path_length": float(robot_path_length),
            "robot_path_to_astar_ratio": float(robot_path_length / max(float(astar_path_length), 1e-6)),
            "max_robot_path_deviation": float(max_deviation),
        }

    def _episode_has_trajectory_quality(self, episode: dict[str, object]) -> bool:
        stats = episode["episode_stats"]
        max_deviation = self.config.max_robot_path_deviation
        if max_deviation is not None and float(stats["max_robot_path_deviation"]) > float(max_deviation):
            return False
        max_ratio = self.config.max_robot_trajectory_path_ratio
        if max_ratio is not None and float(stats["robot_path_to_astar_ratio"]) > float(max_ratio):
            return False
        return True

    def _sample_episode_context(self, episode_id: int, rng: np.random.Generator):
        generator_types = tuple(self.config.map_config.generator_types)
        preferred_map_type = generator_types[episode_id % len(generator_types)] if generator_types else None
        preferred_attempts = max(1, int(round(self.config.episode_resample_limit * 0.75)))
        for attempt_index in range(self.config.episode_resample_limit):
            map_seed = int(rng.integers(0, 2**31 - 1))
            map_type = preferred_map_type if attempt_index < preferred_attempts else None
            if self._fixed_map_context is not None:
                map_bundle, planning_occupancy, occupied_points = self._fixed_map_context
            else:
                map_bundle = generate_map(self.config.map_config, map_seed, episode_id, map_type=map_type)
                inflation = self.config.robot_config.radius + self.config.map_config.inflation_margin
                planning_occupancy = inflate_occupancy_grid(
                    map_bundle.occupancy,
                    inflation_radius_m=inflation,
                    resolution=map_bundle.resolution,
                )
                occupied_points = build_static_obstacle_index(
                    occupied_boundary_cell_centers(
                        map_bundle.occupancy,
                        map_bundle.resolution,
                        map_bundle.origin,
                    )
                )
            start_pose, goal_point = self._sample_start_and_goal(planning_occupancy, map_bundle.origin, rng)
            if start_pose is None or goal_point is None:
                continue
            start_rc = world_to_grid(start_pose[:2], map_bundle.resolution, map_bundle.origin)
            goal_rc = world_to_grid(goal_point, map_bundle.resolution, map_bundle.origin)
            path_rc = astar_search(planning_occupancy, start_rc, goal_rc)
            if not path_rc:
                continue
            path_world = path_to_world(path_rc, map_bundle.resolution, map_bundle.origin)
            path_distance = path_length(path_world)
            if path_distance < self.config.map_config.min_path_length:
                continue
            direct_distance = euclidean(start_pose[:2], goal_point)
            directness_ratio = path_distance / max(direct_distance, 1e-6)
            if directness_ratio < self.config.map_config.min_path_directness_ratio:
                continue
            turn_angle_sum = self._path_turn_angle_sum(path_world)
            if turn_angle_sum < self.config.map_config.min_path_turn_angle_sum:
                continue
            dyn_obstacles = sample_dynamic_obstacles(
                occupancy=planning_occupancy,
                resolution=map_bundle.resolution,
                origin=map_bundle.origin,
                map_config=self.config.map_config,
                robot_config=self.config.robot_config,
                dynamic_config=self.config.dynamic_obstacle_config,
                path_world=path_world,
                rng_seed=int(rng.integers(0, 2**31 - 1)),
                avoid_points=[start_pose[:2], goal_point],
            )
            if len(dyn_obstacles) != int(self.config.dynamic_obstacle_config.num_dynamic_obstacles):
                continue
            return map_bundle, planning_occupancy, occupied_points, start_pose, goal_point, path_rc, path_world, dyn_obstacles
        raise RuntimeError("Failed to sample a valid episode after repeated attempts")

    @staticmethod
    def _path_turn_angle_sum(path_world: np.ndarray) -> float:
        """Return cumulative absolute heading change along a path polyline."""
        if len(path_world) < 3:
            return 0.0
        deltas = np.diff(path_world, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-6
        if int(np.sum(valid)) < 2:
            return 0.0
        headings = np.arctan2(deltas[valid, 1], deltas[valid, 0])
        diffs = np.diff(headings)
        wrapped = (diffs + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.sum(np.abs(wrapped)))

    def _sample_start_and_goal(self, occupancy: np.ndarray, origin: Sequence[float], rng: np.random.Generator):
        free_indices = np.argwhere(occupancy == 0)
        if len(free_indices) == 0:
            return None, None

        for _ in range(500):
            start_idx = int(rng.integers(0, len(free_indices)))
            goal_idx = int(rng.integers(0, len(free_indices)))
            start_row, start_col = free_indices[start_idx]
            goal_row, goal_col = free_indices[goal_idx]
            start_xy = np.array(
                [
                    origin[0] + (start_col + 0.5) * self.config.map_config.resolution,
                    origin[1] + (start_row + 0.5) * self.config.map_config.resolution,
                ],
                dtype=float,
            )
            goal_xy = np.array(
                [
                    origin[0] + (goal_col + 0.5) * self.config.map_config.resolution,
                    origin[1] + (goal_row + 0.5) * self.config.map_config.resolution,
                ],
                dtype=float,
            )
            if euclidean(start_xy, goal_xy) < self.config.map_config.min_start_goal_distance:
                continue
            start_heading = sample_heading_toward(start_xy, goal_xy, noise_std=0.35, rng=rng)
            start_pose = np.array([start_xy[0], start_xy[1], start_heading], dtype=float)
            if disk_collides_with_occupancy(
                start_xy,
                self.config.robot_config.radius,
                occupancy,
                self.config.map_config.resolution,
                origin,
            ):
                continue
            if disk_collides_with_occupancy(
                goal_xy,
                self.config.robot_config.radius,
                occupancy,
                self.config.map_config.resolution,
                origin,
            ):
                continue
            return start_pose, goal_xy
        return None, None

    def _apply_action(self, robot_pose: np.ndarray, v: float, omega: float) -> np.ndarray:
        from .dynamics import unicycle_step

        return unicycle_step(robot_pose, v, omega, self.config.robot_config.dt)

    def _compute_structured_dynamic_fields(
        self,
        robot_pose_t: np.ndarray,
        robot_pose_t1: np.ndarray,
        dynamic_t: list[dict[str, object]],
        dynamic_t1: list[dict[str, object]],
    ) -> dict[str, object]:
        """Build fixed-order V4 obstacle arrays from true obstacle snapshots."""
        ids_t = [str(obstacle["obstacle_id"]) for obstacle in dynamic_t]
        ids_t1 = [str(obstacle["obstacle_id"]) for obstacle in dynamic_t1]
        if ids_t != ids_t1:
            raise ValueError(f"Dynamic obstacle identity changed within a step: {ids_t} != {ids_t1}")

        world_t = np.asarray([np.asarray(obstacle["position"], dtype=float) for obstacle in dynamic_t], dtype=float)
        world_t1 = np.asarray([np.asarray(obstacle["position"], dtype=float) for obstacle in dynamic_t1], dtype=float)
        radii = np.asarray([float(obstacle["radius"]) for obstacle in dynamic_t], dtype=float)
        delta_world = world_t1 - world_t
        dt = max(1e-9, float(self.config.robot_config.dt))
        vel_world = delta_world / dt
        vel_robot_axes = self._world_vectors_to_robot_frame(vel_world, float(robot_pose_t[2]))
        robot_vel_world = (np.asarray(robot_pose_t1[:2], dtype=float) - np.asarray(robot_pose_t[:2], dtype=float)) / dt
        rel_vel_robot = self._world_vectors_to_robot_frame(vel_world - robot_vel_world[None, :], float(robot_pose_t[2]))
        robot_t = self._world_points_to_robot_frame(world_t, robot_pose_t).astype(np.float32)
        robot_t1 = self._world_points_to_robot_frame(world_t1, robot_pose_t1).astype(np.float32)
        delta_robot = robot_t1 - robot_t
        rel_vel_robot_fd = delta_robot / dt

        return {
            "dynamic_obstacle_ids": ids_t,
            "dynamic_obstacles_world_t": world_t.astype(np.float32),
            "dynamic_obstacles_world_t1": world_t1.astype(np.float32),
            "dynamic_obstacle_delta_world_t": delta_world.astype(np.float32),
            "dynamic_obstacles_robot_t": robot_t,
            "dynamic_obstacles_robot_t1": robot_t1,
            "dynamic_obstacle_delta_robot_t": delta_robot.astype(np.float32),
            "dynamic_obstacle_radii": radii.astype(np.float32),
            "dynamic_obstacle_vel_world_t": vel_world.astype(np.float32),
            "dynamic_obstacle_vel_robot_t": vel_robot_axes.astype(np.float32),
            "dynamic_obstacle_rel_vel_robot_t": rel_vel_robot.astype(np.float32),
            "dynamic_obstacle_rel_vel_robot_fd_t": rel_vel_robot_fd.astype(np.float32),
        }

    @staticmethod
    def _add_dynamic_velocity_targets(steps: list[dict[str, object]]) -> None:
        """Fill t+1 finite-difference relative-velocity targets after the episode is built."""
        for index, step in enumerate(steps):
            if index + 1 < len(steps):
                next_velocity = np.asarray(steps[index + 1]["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=np.float32)
            else:
                next_velocity = np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=np.float32)
            step["dynamic_obstacle_rel_vel_robot_fd_t1"] = next_velocity.copy()

    @staticmethod
    def _add_causal_world_velocity_fields(steps: list[dict[str, object]]) -> None:
        """Fill causal world velocity used by the world-frame practical branch."""
        for index, step in enumerate(steps):
            dt = max(1e-9, float(step.get("dt", 0.1)))
            if index == 0:
                current_snapshots = list(step.get("dynamic_obstacles_t", []))
                if not current_snapshots:
                    raise ValueError("Cannot bootstrap causal world velocity without dynamic_obstacles_t snapshots")
                causal_velocity = np.asarray(
                    [np.asarray(snapshot["velocity"], dtype=np.float32) for snapshot in current_snapshots],
                    dtype=np.float32,
                )
            else:
                current_world = np.asarray(step["dynamic_obstacles_world_t"], dtype=np.float32)
                previous_world = np.asarray(steps[index - 1]["dynamic_obstacles_world_t"], dtype=np.float32)
                causal_velocity = (current_world - previous_world) / dt
            step["dynamic_obstacle_vel_world_causal_t"] = causal_velocity.astype(np.float32)

    @staticmethod
    def _world_points_to_robot_frame(points_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """Express world-frame points as robot-relative positions."""
        points = np.asarray(points_world, dtype=float)
        translated = points - np.asarray(robot_pose[:2], dtype=float)[None, :]
        return EpisodeSimulator._world_vectors_to_robot_frame(translated, float(robot_pose[2]))

    @staticmethod
    def _world_vectors_to_robot_frame(vectors_world: np.ndarray, robot_theta: float) -> np.ndarray:
        """Rotate world-frame vectors into robot axes using R(-theta)."""
        vectors = np.asarray(vectors_world, dtype=float)
        cos_theta = float(np.cos(robot_theta))
        sin_theta = float(np.sin(robot_theta))
        rotation = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=float)
        return vectors @ rotation.T

    def _compute_geometry_labels(
        self,
        robot_pose: np.ndarray,
        waypoint: dict[str, object],
        path_world: np.ndarray,
        occupied_points: np.ndarray,
        dynamic_obstacles,
        map_resolution: float,
        suffix: str,
    ) -> dict[str, object]:
        """Compute V2 path, subgoal, heading, and clearance labels for one pose."""
        goal_features = np.asarray(waypoint["goal_features"], dtype=float)
        heading_error = float(np.arctan2(goal_features[1], goal_features[0]))
        static_clearance = minimum_static_clearance(
            robot_pose[:2],
            self.config.robot_config.radius,
            occupied_points,
            map_resolution,
        )
        dynamic_clearance = minimum_dynamic_clearance(
            robot_pose[:2],
            self.config.robot_config.radius,
            dynamic_obstacles,
        )
        total_clearance = float(min(static_clearance, dynamic_clearance))
        return {
            f"subgoal_world_{suffix}": np.asarray(waypoint["selected_subgoal_world"], dtype=float).copy(),
            f"nearest_path_index_{suffix}": int(waypoint["nearest_path_index"]),
            f"selected_subgoal_index_{suffix}": int(waypoint["selected_subgoal_index"]),
            f"distance_to_subgoal_{suffix}": float(goal_features[2]),
            f"distance_to_path_{suffix}": float(distance_point_to_polyline(robot_pose[:2], path_world)),
            f"heading_error_{suffix}": heading_error,
            f"heading_error_cos_{suffix}": float(np.cos(heading_error)),
            f"heading_error_sin_{suffix}": float(np.sin(heading_error)),
            f"min_static_clearance_{suffix}": float(static_clearance),
            f"min_dynamic_clearance_{suffix}": float(dynamic_clearance),
            f"min_total_clearance_{suffix}": total_clearance,
        }

    def _risk_from_clearance(self, clearance: float) -> float:
        """Return a convenience risk proxy from total clearance."""
        if clearance <= 0.0:
            return 1.0
        tau = max(1e-6, float(self.config.risk_tau))
        return float(np.exp(-clearance / tau))

    def _dynamic_risk_from_clearance(self, clearance: float) -> float:
        """Return the V4 dynamic-only risk proxy from dynamic clearance."""
        if clearance <= 0.0:
            return 1.0
        tau = max(1e-6, float(self.config.dynamic_risk_tau))
        return float(np.exp(-clearance / tau))

    def _dynamic_visible_from_clearance(self, clearance: float) -> bool:
        """Return the V4 structured-state visibility flag used for events and acceptance."""
        return bool(float(clearance) < float(self.config.dynamic_visibility_radius))

    def _compute_lidar_fields(
        self,
        *,
        robot_pose: np.ndarray,
        occupancy: np.ndarray,
        map_bundle,
        dynamic_obstacles,
        suffix: str,
    ) -> dict[str, object]:
        """Compute real LiDAR fields only when this dataset is configured to keep LiDAR."""
        should_cast = bool(
            self.config.save_static_lidar
            or self.config.save_dynamic_residual
            or self.config.visualization_config.draw_lidar_in_gif
        )
        if should_cast:
            lidar_full = cast_lidar_scan(
                robot_pose,
                occupancy,
                map_bundle.resolution,
                map_bundle.origin,
                self.config.lidar_config,
                dynamic_obstacles,
            )
            lidar_static = cast_lidar_scan(
                robot_pose,
                occupancy,
                map_bundle.resolution,
                map_bundle.origin,
                self.config.lidar_config,
                (),
            )
        else:
            lidar_full = np.full(
                (int(self.config.lidar_config.num_beams),),
                float(self.config.lidar_config.max_range),
                dtype=np.float32,
            )
            lidar_static = lidar_full.copy()

        return {
            f"lidar_{suffix}": lidar_full.astype(np.float32),
            f"lidar_static_{suffix}": lidar_static.astype(np.float32),
            **self._compute_dynamic_lidar_fields(lidar_static, lidar_full, suffix=suffix),
        }

    def _compute_dynamic_lidar_fields(
        self,
        lidar_static: np.ndarray,
        lidar_full: np.ndarray,
        suffix: str,
    ) -> dict[str, object]:
        """Compute residual LiDAR observation fields for one timestep."""
        residual = np.maximum(0.0, np.asarray(lidar_static, dtype=np.float32) - np.asarray(lidar_full, dtype=np.float32))
        hit_mask = residual > float(self.config.dynamic_residual_epsilon)
        visible = int(np.sum(hit_mask)) >= int(self.config.dynamic_visible_min_beams)
        return {
            f"lidar_dynamic_residual_{suffix}": residual.astype(np.float32),
            f"dynamic_hit_mask_{suffix}": hit_mask.astype(np.uint8),
            f"lidar_dynamic_visible_{suffix}": bool(visible),
            f"dynamic_hit_fraction_{suffix}": float(np.mean(hit_mask)) if hit_mask.size else 0.0,
            f"dynamic_residual_energy_{suffix}": float(np.mean(residual)) if residual.size else 0.0,
        }

    def _episode_has_dynamic_interaction(self, episode: dict[str, object]) -> bool:
        """Return whether a candidate satisfies V4 dynamic richness thresholds."""
        stats = episode["episode_stats"]
        return bool(
            float(stats["dynamic_visible_fraction"]) >= float(self.config.min_dynamic_visible_fraction)
            or float(stats["dynamic_risk_fraction"]) >= float(self.config.min_dynamic_risk_fraction)
            or int(stats["dynamic_event_steps"]) >= int(self.config.min_dynamic_event_steps)
        )

    def _compute_dynamic_episode_stats(self, steps: list[dict[str, object]]) -> dict[str, object]:
        """Summarize V4 dynamic supervision richness for one episode."""
        total = len(steps)
        if total == 0:
            return {
                "dynamic_visible_fraction": 0.0,
                "lidar_dynamic_visible_fraction": 0.0,
                "dynamic_risk_fraction": 0.0,
                "dynamic_event_steps": 0,
                "dynamic_collision_t1_count": 0,
                "dynamic_collision_within_3_count": 0,
                "dynamic_collision_within_5_count": 0,
            }
        risky = [
            float(step.get("dynamic_risk_t1", 0.0)) >= float(self.config.dynamic_risk_positive_threshold)
            for step in steps
        ]
        return {
            "dynamic_visible_fraction": float(np.mean([bool(step.get("dynamic_visible_t", False)) for step in steps])),
            "lidar_dynamic_visible_fraction": float(
                np.mean([bool(step.get("lidar_dynamic_visible_t", False)) for step in steps])
            ),
            "dynamic_risk_fraction": float(np.mean(risky)),
            "dynamic_event_steps": int(sum(bool(step.get("dynamic_event_step_t", False)) for step in steps)),
            "dynamic_collision_t1_count": int(sum(bool(step.get("dynamic_collision_t1", False)) for step in steps)),
            "dynamic_collision_within_3_count": int(
                sum(bool(step.get("dynamic_collision_within_3", False)) for step in steps)
            ),
            "dynamic_collision_within_5_count": int(
                sum(bool(step.get("dynamic_collision_within_5", False)) for step in steps)
            ),
        }

    @staticmethod
    def _add_collision_horizon_labels(steps: list[dict[str, object]], horizons: tuple[int, ...]) -> None:
        """Add collision-within-H labels using future collision_t1 values.

        For step index i, collision_t1 at index i is the event at t+1, so
        collision_within_H checks indices [i, i + H - 1].
        """
        for index, step in enumerate(steps):
            for horizon in horizons:
                end = min(len(steps), index + horizon)
                step[f"collision_within_{horizon}"] = bool(
                    any(bool(future_step.get("collision_t1", False)) for future_step in steps[index:end])
                )

    def _add_dynamic_horizon_and_event_labels(self, steps: list[dict[str, object]], horizons: tuple[int, ...]) -> None:
        """Add dynamic-only horizon labels and the explicit V4 event-step flag."""
        for index, step in enumerate(steps):
            for horizon in horizons:
                end = min(len(steps), index + horizon)
                step[f"dynamic_collision_within_{horizon}"] = bool(
                    any(bool(future_step.get("dynamic_collision_t1", False)) for future_step in steps[index:end])
                )
            event_visible = bool(step.get("dynamic_visible_t", False))
            event_risky = float(step.get("dynamic_risk_t1", 0.0)) >= float(self.config.dynamic_risk_positive_threshold)
            event_h3 = bool(step.get("dynamic_collision_within_3", False))
            event_h5 = bool(step.get("dynamic_collision_within_5", False))
            step["dynamic_event_step_t"] = bool(event_visible or event_risky or event_h3 or event_h5)
