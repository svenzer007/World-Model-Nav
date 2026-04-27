"""Baseline V4 controller with true cloned dynamic-obstacle rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import math
import numpy as np

from sim2d.config import ACTIONS, DatasetConfig
from sim2d.dynamics import disk_collides_with_occupancy, minimum_static_clearance, unicycle_step
from sim2d.obstacles import DynamicObstacle
from sim2d.utils import distance_point_to_polyline, nearest_point_index, world_to_grid, wrap_angle


@dataclass
class LearnedClearanceCalibrationConfig:
    enabled: bool = False
    summary_json_path: str | None = None
    alpha_bias: float = 1.0
    alpha_std: float = 0.5
    use_action_family_specific: bool = True
    fallback_to_global: bool = True
    clamp_min_clearance_adjustment: float | None = None
    clamp_max_clearance_adjustment: float | None = None


@dataclass
class StructuredControllerConfig:
    horizon: int = 10
    w_progress: float = 2.0
    w_heading: float = 0.8
    w_path: float = 1.1
    w_velocity: float = 0.3
    w_static: float = 0.5
    w_dynamic: float = 2.0
    static_collision_penalty: float = 1e6
    dynamic_collision_penalty: float = 1e6
    dynamic_risk_tau: float = 0.5
    min_clearance_clip: float = 0.05
    learned_clearance_calibration: LearnedClearanceCalibrationConfig = field(
        default_factory=LearnedClearanceCalibrationConfig
    )


def robot_frame_points(robot_pose: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(robot_pose, dtype=float)
    points = np.asarray(points_world, dtype=float)
    translated = points - pose[:2][None, :]
    cos_theta = float(np.cos(pose[2]))
    sin_theta = float(np.sin(pose[2]))
    rotation = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=float)
    return translated @ rotation.T


def planning_point_collision(
    position_xy: np.ndarray,
    planning_occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
) -> bool:
    """Match the expert's inflated planning-map point-collision feasibility check."""
    row, col = world_to_grid(position_xy, resolution, origin)
    if row < 0 or row >= planning_occupancy.shape[0] or col < 0 or col >= planning_occupancy.shape[1]:
        return True
    return bool(planning_occupancy[row, col] != 0)


def dynamic_clearances_from_positions(
    rel_positions: np.ndarray,
    radii: np.ndarray,
    robot_radius: float,
) -> np.ndarray:
    positions = np.asarray(rel_positions, dtype=float)
    obstacle_radii = np.asarray(radii, dtype=float)
    if positions.size == 0:
        return np.zeros((0,), dtype=float)
    distances = np.linalg.norm(positions + 1e-6, axis=1)
    return distances - float(robot_radius) - obstacle_radii


def velocity_penalty_from_action(v: float) -> float:
    return max(0.0, 0.4 - float(v))


def path_tangent_penalty(pose: np.ndarray, path_world: np.ndarray) -> float:
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


def expert_style_rollout_cost(
    *,
    rollout_endpoint_pose: np.ndarray,
    current_subgoal_world: np.ndarray,
    path_world: np.ndarray,
    min_combined_clearance: float,
    velocity_penalty: float,
    feasible: bool,
    controller_cfg: StructuredControllerConfig,
) -> dict[str, Any]:
    endpoint = np.asarray(rollout_endpoint_pose, dtype=float)
    subgoal = np.asarray(current_subgoal_world, dtype=float)
    goal_progress_cost = float(np.linalg.norm(endpoint[:2] - subgoal))
    heading_to_goal = math.atan2(float(subgoal[1] - endpoint[1]), float(subgoal[0] - endpoint[0]))
    heading_cost = abs(wrap_angle(heading_to_goal - float(endpoint[2])))
    path_distance_cost = float(distance_point_to_polyline(endpoint[:2], path_world))
    tangent_cost = float(path_tangent_penalty(endpoint, path_world))
    path_cost = float(path_distance_cost + tangent_cost)
    clearance_cost = 1.0 / max(float(min_combined_clearance), float(controller_cfg.min_clearance_clip))
    infeasibility_penalty = float(controller_cfg.static_collision_penalty) if not feasible else 0.0
    total_cost = (
        float(controller_cfg.w_progress) * goal_progress_cost
        + float(controller_cfg.w_heading) * heading_cost
        + float(controller_cfg.w_path) * path_cost
        + float(controller_cfg.w_velocity) * float(velocity_penalty)
        + float(controller_cfg.w_static) * float(clearance_cost)
        + float(infeasibility_penalty)
    )
    return {
        "goal_progress_cost": float(goal_progress_cost),
        "heading_cost": float(heading_cost),
        "path_distance_cost": float(path_distance_cost),
        "path_tangent_penalty": float(tangent_cost),
        "path_cost": float(path_cost),
        "velocity_cost": float(velocity_penalty),
        "clearance_cost": float(clearance_cost),
        "infeasibility_penalty": float(infeasibility_penalty),
        "total_cost": float(total_cost),
    }


class BaselineStructuredController:
    """Use true dynamic-obstacle rollout with analytic scoring."""

    name = "baseline_structured"

    def __init__(self, config: DatasetConfig, controller_cfg: StructuredControllerConfig):
        if controller_cfg.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.config = config
        self.controller_cfg = controller_cfg

    def reset(self) -> None:
        """Reset episode state. The baseline controller is memoryless."""

    def commit_step(
        self,
        *,
        robot_pose_t,
        robot_pose_t1,
        dynamic_obstacles_t,
        dynamic_obstacles_t1,
        action_index,
        action_cont,
        path_world,
    ) -> None:
        del robot_pose_t, robot_pose_t1, dynamic_obstacles_t, dynamic_obstacles_t1, action_index, action_cont, path_world

    def select_action(
        self,
        *,
        robot_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        planning_occupancy: np.ndarray,
        true_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
        occupied_points: np.ndarray,
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> dict[str, object]:
        candidates = [
            self._evaluate_candidate(
                action_index=action_index,
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
            for action_index in ACTIONS
        ]
        feasible_non_stop = [
            candidate for candidate in candidates if bool(candidate["feasible"]) and int(candidate["action_index"]) != 0
        ]
        feasible_all = [candidate for candidate in candidates if bool(candidate["feasible"])]
        if feasible_non_stop:
            chosen = min(feasible_non_stop, key=lambda item: float(item["total_cost"]))
            selection_mode = "feasible_non_stop"
        elif feasible_all:
            chosen = min(feasible_all, key=lambda item: float(item["total_cost"]))
            selection_mode = "feasible_all"
        else:
            chosen = min(candidates, key=lambda item: float(item["total_cost"]))
            selection_mode = "fallback_all"
        action = ACTIONS[int(chosen["action_index"])]
        return {
            "action_index": int(chosen["action_index"]),
            "action_name": str(action["name"]),
            "v": float(action["v"]),
            "omega": float(action["omega"]),
            "debug": {
                "mode": "baseline_structured_rollout",
                "horizon": int(self.controller_cfg.horizon),
                "weights": {
                    "progress": float(self.controller_cfg.w_progress),
                    "heading": float(self.controller_cfg.w_heading),
                    "path": float(self.controller_cfg.w_path),
                    "velocity": float(self.controller_cfg.w_velocity),
                    "static": float(self.controller_cfg.w_static),
                    "dynamic": float(self.controller_cfg.w_dynamic),
                },
                "tau": float(self.controller_cfg.dynamic_risk_tau),
                "min_clearance_clip": float(self.controller_cfg.min_clearance_clip),
                "static_collision_penalty": float(self.controller_cfg.static_collision_penalty),
                "dynamic_collision_penalty": float(self.controller_cfg.dynamic_collision_penalty),
                "num_feasible_non_stop": int(len(feasible_non_stop)),
                "num_feasible_all": int(len(feasible_all)),
                "selection_mode": selection_mode,
                "chosen": chosen,
                "candidates": candidates,
            },
            "chosen_action_min_clearance": float(min(chosen["min_static_clearance"], chosen["min_dynamic_clearance"])),
            "emergency_override_used": False,
        }

    def _evaluate_candidate(
        self,
        *,
        action_index: int,
        robot_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        planning_occupancy: np.ndarray,
        true_occupancy: np.ndarray,
        resolution: float,
        origin: Sequence[float],
        occupied_points: np.ndarray,
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> dict[str, Any]:
        action = ACTIONS[int(action_index)]
        pose = np.asarray(robot_pose, dtype=float).copy()
        rolled_obstacles = [obstacle.clone() for obstacle in dynamic_obstacles]
        robot_rollout_world: list[list[float]] = []
        static_clearances: list[float] = []
        static_collisions: list[bool] = []
        dynamic_clearances: list[float] = []
        dynamic_collisions: list[bool] = []
        planning_collisions: list[bool] = []
        combined_clearances: list[float] = []
        dynamic_positions_robot: list[list[list[float]]] = []
        infeasible_reasons: list[str] = []

        for _ in range(int(self.controller_cfg.horizon)):
            pose = unicycle_step(
                pose,
                float(action["v"]),
                float(action["omega"]),
                float(self.config.robot_config.dt),
            )
            for obstacle in rolled_obstacles:
                obstacle.step(float(self.config.robot_config.dt))

            robot_rollout_world.append([float(value) for value in pose])

            static_clearance = float(
                minimum_static_clearance(
                    pose[:2],
                    self.config.robot_config.radius,
                    occupied_points,
                    resolution,
                )
            )
            static_collision = bool(
                disk_collides_with_occupancy(
                    pose[:2],
                    self.config.robot_config.radius,
                    true_occupancy,
                    resolution,
                    origin,
                )
            )
            positions_world = np.asarray([obstacle.position.copy() for obstacle in rolled_obstacles], dtype=float)
            radii = np.asarray([float(obstacle.radius) for obstacle in rolled_obstacles], dtype=float)
            positions_robot = robot_frame_points(pose, positions_world)
            dyn_clearances = dynamic_clearances_from_positions(positions_robot, radii, self.config.robot_config.radius)
            planning_collision = bool(planning_point_collision(pose[:2], planning_occupancy, resolution, origin))

            static_clearances.append(static_clearance)
            static_collisions.append(static_collision)
            planning_collisions.append(planning_collision)
            dynamic_positions_robot.append(positions_robot.astype(float).tolist())
            dynamic_clearance = float(np.min(dyn_clearances)) if dyn_clearances.size else float("inf")
            dynamic_clearances.append(dynamic_clearance)
            dynamic_collisions.append(bool(np.any(dyn_clearances <= 0.0)))
            combined_clearances.append(float(min(static_clearance, dynamic_clearance)))

        if any(planning_collisions):
            infeasible_reasons.append("planning_collision")
        if any(static_collisions):
            infeasible_reasons.append("static_collision")
        if any(dynamic_collisions):
            infeasible_reasons.append("dynamic_collision")
        feasible = not bool(infeasible_reasons)

        score = expert_style_rollout_cost(
            rollout_endpoint_pose=pose,
            current_subgoal_world=np.asarray(current_subgoal_world, dtype=float),
            path_world=path_world,
            min_combined_clearance=float(np.min(combined_clearances)) if combined_clearances else float("inf"),
            velocity_penalty=velocity_penalty_from_action(float(action["v"])),
            feasible=feasible,
            controller_cfg=self.controller_cfg,
        )
        return {
            "action_index": int(action_index),
            "action_name": str(action["name"]),
            "v": float(action["v"]),
            "omega": float(action["omega"]),
            "total_cost": float(score["total_cost"]),
            "goal_progress_cost": float(score["goal_progress_cost"]),
            "heading_cost": float(score["heading_cost"]),
            "path_distance_cost": float(score["path_distance_cost"]),
            "path_tangent_penalty": float(score["path_tangent_penalty"]),
            "path_cost": float(score["path_cost"]),
            "velocity_cost": float(score["velocity_cost"]),
            "clearance_cost": float(score["clearance_cost"]),
            "infeasibility_penalty": float(score["infeasibility_penalty"]),
            "robot_rollout_world": robot_rollout_world,
            "static_clearances": static_clearances,
            "static_collisions": static_collisions,
            "planning_collisions": planning_collisions,
            "dynamic_clearances": dynamic_clearances,
            "dynamic_collisions": dynamic_collisions,
            "combined_clearances": combined_clearances,
            "dynamic_positions_robot": dynamic_positions_robot,
            "feasible": bool(feasible),
            "infeasible_reasons": infeasible_reasons,
            "min_static_clearance": float(np.min(static_clearances)) if static_clearances else float("inf"),
            "min_dynamic_clearance": float(np.min(dynamic_clearances)) if dynamic_clearances else float("inf"),
            "min_combined_clearance": float(np.min(combined_clearances)) if combined_clearances else float("inf"),
        }
