"""DWA-inspired discrete-action expert for local navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ACTIONS, ExpertConfig, RobotConfig
from .dynamics import (
    disk_collides_with_dynamic,
    disk_collides_with_occupancy,
    holonomic_step,
    minimum_combined_clearance,
)
from .utils import distance_point_to_polyline, nearest_point_index, world_to_grid, wrap_angle


@dataclass
class ExpertDecision:
    action_index: int
    action_name: str
    vx: float
    vy: float
    omega: float
    expert_debug: dict[str, object]
    chosen_action_min_clearance: float
    emergency_override_used: bool

    @property
    def v(self) -> float:
        return self.vx

    @property
    def wz(self) -> float:
        return self.omega


class DWAExpertPolicy:
    """Evaluate the fixed discrete action set with a DWA-like rollout objective."""

    def __init__(self, config: ExpertConfig, robot_config: RobotConfig):
        self.config = config
        self.robot_config = robot_config

    def select_action(
        self,
        robot_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        planning_occupancy: np.ndarray,
        true_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
        occupied_points: np.ndarray,
        dynamic_obstacles: Sequence[object],
    ) -> ExpertDecision:
        """Choose the minimum-cost feasible action."""
        candidate_logs: list[dict[str, object]] = []
        feasible_non_stop: list[dict[str, object]] = []
        feasible_all: list[dict[str, object]] = []

        for action_index in self.config.action_indices:
            if int(action_index) not in ACTIONS:
                raise ValueError(f"Unknown expert action index: {action_index}")
            candidate_log = self._evaluate_action(
                action_index=int(action_index),
                robot_pose=robot_pose,
                current_subgoal_world=current_subgoal_world,
                path_world=path_world,
                planning_occupancy=planning_occupancy,
                true_occupancy=true_occupancy,
                resolution=resolution,
                origin=origin,
                occupied_points=occupied_points,
                dynamic_obstacles=dynamic_obstacles,
            )
            candidate_logs.append(candidate_log)
            if candidate_log["feasible"]:
                feasible_all.append(candidate_log)
                if action_index != 0:
                    feasible_non_stop.append(candidate_log)

        emergency_override_used = False
        if feasible_non_stop:
            chosen = min(feasible_non_stop, key=lambda item: float(item["total_score"]))
        elif feasible_all:
            chosen = min(feasible_all, key=lambda item: float(item["total_score"]))
        else:
            emergency_override_used = True
            chosen = min(candidate_logs, key=lambda item: float(item["total_score"]))

        action = ACTIONS[int(chosen["action_index"])]
        return ExpertDecision(
            action_index=int(chosen["action_index"]),
            action_name=str(action["name"]),
            vx=float(action["vx"]),
            vy=float(action["vy"]),
            omega=float(action["omega"]),
            expert_debug={
                "candidates": candidate_logs,
                "chosen": chosen,
            },
            chosen_action_min_clearance=float(chosen["min_clearance"]),
            emergency_override_used=emergency_override_used or bool(chosen.get("emergency_choice", False)),
        )

    def _evaluate_action(
        self,
        action_index: int,
        robot_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        planning_occupancy: np.ndarray,
        true_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
        occupied_points: np.ndarray,
        dynamic_obstacles: Sequence[object],
    ) -> dict[str, object]:
        action = ACTIONS[action_index]
        pose = np.asarray(robot_pose, dtype=float).copy()
        obstacles = [obstacle.clone() for obstacle in dynamic_obstacles]
        min_clearance = float("inf")
        feasible = True
        collision_reason = ""
        rollout_positions: list[np.ndarray] = [pose[:2].copy()]
        rollout_headings: list[float] = [float(pose[2])]
        clearance_history: list[float] = []
        collision_step_index: int | None = None
        collision_position: np.ndarray | None = None
        collision_map_type = ""

        true_map_one_step_collision, true_map_one_step_collision_reason = self._evaluate_true_map_one_step(
            pose=pose,
            action_vx=float(action["vx"]),
            action_vy=float(action["vy"]),
            action_omega=float(action["omega"]),
            true_occupancy=true_occupancy,
            resolution=resolution,
            origin=origin,
            dynamic_obstacles=dynamic_obstacles,
        )

        for step_index in range(self.config.rollout_steps):
            pose = self._step_pose(pose, float(action["vx"]), float(action["vy"]), float(action["omega"]))
            for obstacle in obstacles:
                obstacle.step(self.robot_config.dt)

            rollout_positions.append(pose[:2].copy())
            rollout_headings.append(float(pose[2]))

            clearance = minimum_combined_clearance(
                pose[:2],
                self.robot_config.radius,
                occupied_points,
                resolution,
                obstacles,
            )
            clearance_history.append(float(clearance))
            min_clearance = min(min_clearance, clearance)

            if self._planning_point_collision(pose[:2], planning_occupancy, resolution, origin):
                feasible = False
                collision_reason = "static_collision"
                collision_step_index = step_index
                collision_position = pose[:2].copy()
                collision_map_type = "planning_static"
                break
            if disk_collides_with_dynamic(pose[:2], self.robot_config.radius, obstacles):
                feasible = False
                collision_reason = "dynamic_collision"
                collision_step_index = step_index
                collision_position = pose[:2].copy()
                collision_map_type = "dynamic"
                break

        if min_clearance == float("inf"):
            min_clearance = minimum_combined_clearance(
                pose[:2],
                self.robot_config.radius,
                occupied_points,
                resolution,
                obstacles,
            )

        goal_distance = float(np.linalg.norm(pose[:2] - current_subgoal_world))
        heading_to_goal = math.atan2(
            float(current_subgoal_world[1] - pose[1]),
            float(current_subgoal_world[0] - pose[0]),
        )
        heading_alignment = abs(wrap_angle(heading_to_goal - float(pose[2])))
        path_distance = distance_point_to_polyline(pose[:2], path_world)
        tangent_penalty = self._path_tangent_penalty(pose, path_world)
        clipped_clearance = max(min_clearance, self.config.min_clearance_clip)
        clearance_cost = 1.0 / clipped_clearance
        velocity_cost = self._velocity_cost(float(action["vx"]), float(action["vy"]))

        total_score = (
            self.config.goal_progress_weight * goal_distance
            + self.config.heading_weight * heading_alignment
            + self.config.clearance_weight * clearance_cost
            + self.config.path_adherence_weight * (path_distance + tangent_penalty)
            + self.config.velocity_weight * velocity_cost
        )
        if not feasible:
            total_score += self.config.collision_penalty

        return {
            "action_index": int(action_index),
            "action_name": str(action["name"]),
            "vx": float(action["vx"]),
            "vy": float(action["vy"]),
            "v": float(action["vx"]),
            "omega": float(action["omega"]),
            "wz": float(action["omega"]),
            "feasible": bool(feasible),
            "collision_reason": collision_reason,
            "goal_progress_cost": float(goal_distance),
            "heading_cost": float(heading_alignment),
            "clearance_cost": float(clearance_cost),
            "path_cost": float(path_distance + tangent_penalty),
            "velocity_cost": float(velocity_cost),
            "total_score": float(total_score),
            "min_clearance": float(min_clearance),
            "rollout_endpoint": pose[:2].copy(),
            "rollout_heading": float(pose[2]),
            "rollout_num_positions": len(rollout_positions),
            "rollout_positions": np.array(rollout_positions, dtype=np.float32),
            "rollout_headings": np.array(rollout_headings, dtype=np.float32),
            "clearance_history": np.array(clearance_history, dtype=np.float32),
            "collision_step_index": None if collision_step_index is None else int(collision_step_index),
            "collision_position": None if collision_position is None else collision_position.astype(np.float32),
            "collision_map_type": collision_map_type,
            "true_map_one_step_collision": bool(true_map_one_step_collision),
            "true_map_one_step_collision_reason": true_map_one_step_collision_reason,
        }

    def _step_pose(self, pose: np.ndarray, vx: float, vy: float, omega: float) -> np.ndarray:
        return holonomic_step(pose, vx, vy, omega, self.robot_config.dt)

    def _evaluate_true_map_one_step(
        self,
        pose: np.ndarray,
        action_vx: float,
        action_vy: float,
        action_omega: float,
        true_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
        dynamic_obstacles: Sequence[object],
    ) -> tuple[bool, str]:
        """Check whether one environment step would collide in the true map."""
        next_pose = self._step_pose(pose, action_vx, action_vy, action_omega)
        simulated_obstacles = [obstacle.clone() for obstacle in dynamic_obstacles]
        for obstacle in simulated_obstacles:
            obstacle.step(self.robot_config.dt)

        if disk_collides_with_occupancy(next_pose[:2], self.robot_config.radius, true_occupancy, resolution, origin):
            return True, "static_collision"
        if disk_collides_with_dynamic(next_pose[:2], self.robot_config.radius, simulated_obstacles):
            return True, "dynamic_collision"
        return False, ""

    def _planning_point_collision(
        self,
        position_xy: np.ndarray,
        planning_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
    ) -> bool:
        """Check point-mass collision on the inflated planning map."""
        row, col = world_to_grid(position_xy, resolution, origin)
        if row < 0 or row >= planning_occupancy.shape[0] or col < 0 or col >= planning_occupancy.shape[1]:
            return True
        return bool(planning_occupancy[row, col] != 0)

    def _path_tangent_penalty(self, pose: np.ndarray, path_world: np.ndarray) -> float:
        if len(path_world) < 2:
            return 0.0
        index = nearest_point_index(path_world, pose[:2])
        next_index = min(index + 1, len(path_world) - 1)
        prev_index = max(index - 1, 0)
        if next_index == prev_index:
            return 0.0
        tangent = path_world[next_index] - path_world[prev_index]
        if np.linalg.norm(tangent) < 1e-8:
            return 0.0
        tangent_heading = math.atan2(float(tangent[1]), float(tangent[0]))
        return abs(wrap_angle(tangent_heading - float(pose[2])))

    @staticmethod
    def _velocity_cost(vx: float, vy: float) -> float:
        translation_speed = float(np.hypot(float(vx), float(vy)))
        reverse_cost = max(0.0, -float(vx))
        return max(0.0, 0.4 - translation_speed) + reverse_cost
