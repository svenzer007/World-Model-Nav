"""Baseline V4 controller with true cloned dynamic-obstacle rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import numpy as np

from sim2d.config import ACTIONS, DatasetConfig
from sim2d.dynamics import disk_collides_with_occupancy, holonomic_step, minimum_static_clearance
from sim2d.obstacles import DynamicObstacle
from sim2d.utils import distance_point_to_polyline, nearest_point_index, world_to_grid, wrap_angle


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
    backward_penalty_enabled: bool = True
    backward_action_name: str = "backward"
    backward_penalty: float = 0.25
    backward_gate_enabled: bool = True
    dynamic_stop_clearance_threshold: float = 0.20
    backward_dynamic_margin: float = 0.10
    action_indices: tuple[int, ...] = tuple(ACTIONS.keys())


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


def velocity_penalty_from_action(vx: float, vy: float) -> float:
    translation_speed = float(np.hypot(float(vx), float(vy)))
    reverse_cost = max(0.0, -float(vx))
    return max(0.0, 0.4 - translation_speed) + reverse_cost


def effective_runtime_actions(controller_cfg: StructuredControllerConfig) -> dict[int, dict[str, float | str]]:
    return {int(action_index): ACTIONS[int(action_index)] for action_index in controller_cfg.action_indices}


def _matches_action_name(candidate: dict[str, Any], expected_name: str) -> bool:
    return str(candidate.get("action_name", "")) == str(expected_name)


def _is_stop_candidate(candidate: dict[str, Any]) -> bool:
    return _matches_action_name(candidate, "stop")


def _is_backward_candidate(candidate: dict[str, Any], controller_cfg: StructuredControllerConfig) -> bool:
    if _matches_action_name(candidate, controller_cfg.backward_action_name):
        return True
    try:
        return float(candidate.get("vx", candidate.get("v", 0.0))) < 0.0
    except (TypeError, ValueError):
        return False


def apply_backward_policy(
    candidates: list[dict[str, Any]],
    controller_cfg: StructuredControllerConfig,
) -> dict[str, object]:
    backward_candidates = [candidate for candidate in candidates if _is_backward_candidate(candidate, controller_cfg)]
    backward_candidate = min(
        backward_candidates,
        key=lambda candidate: float(candidate.get("total_cost", float("inf"))),
        default=None,
    )
    stop_candidate = next((candidate for candidate in candidates if _is_stop_candidate(candidate)), None)
    normal_non_stop_candidates = [
        candidate
        for candidate in candidates
        if not _is_stop_candidate(candidate) and not _is_backward_candidate(candidate, controller_cfg)
    ]

    normal_non_stop_static_or_planning_feasible_count = sum(
        not (any(bool(value) for value in candidate.get("planning_collisions", [])) or any(bool(value) for value in candidate.get("static_collisions", [])))
        for candidate in normal_non_stop_candidates
    )
    normal_non_stop_dynamic_feasible_count = sum(
        not any(bool(value) for value in candidate.get("dynamic_collisions", []))
        for candidate in normal_non_stop_candidates
    )

    stop_min_dynamic_clearance = float(stop_candidate.get("min_dynamic_clearance", float("inf"))) if stop_candidate else float("nan")
    stop_dynamic_collision = bool(any(bool(value) for value in stop_candidate.get("dynamic_collisions", []))) if stop_candidate else False

    backward_allowed = True
    backward_gate_reason = "not_backward_action"
    backward_selected = False
    backward_candidate_cost_before_penalty = float("nan")
    backward_candidate_cost_after_penalty = float("nan")

    if backward_candidate is not None:
        for reverse_candidate in backward_candidates:
            reverse_cost_before_penalty = float(
                reverse_candidate.get(
                    "total_cost_before_backward_penalty",
                    reverse_candidate.get("total_cost", float("nan")),
                )
            )
            reverse_allowed = True
            reverse_gate_reason = "gate_disabled"
            if controller_cfg.backward_gate_enabled:
                static_escape_allowed = normal_non_stop_static_or_planning_feasible_count == 0
                stop_is_dynamically_unsafe = bool(
                    stop_dynamic_collision
                    or stop_min_dynamic_clearance < float(controller_cfg.dynamic_stop_clearance_threshold)
                )
                reverse_min_dynamic_clearance = float(reverse_candidate.get("min_dynamic_clearance", float("-inf")))
                dynamic_escape_allowed = bool(
                    stop_is_dynamically_unsafe
                    and reverse_min_dynamic_clearance
                    > stop_min_dynamic_clearance + float(controller_cfg.backward_dynamic_margin)
                )
                if static_escape_allowed:
                    reverse_allowed = True
                    reverse_gate_reason = "static_escape"
                elif dynamic_escape_allowed:
                    reverse_allowed = True
                    reverse_gate_reason = "dynamic_escape"
                elif stop_is_dynamically_unsafe:
                    reverse_allowed = False
                    reverse_gate_reason = "dynamic_gate_failed"
                else:
                    reverse_allowed = False
                    reverse_gate_reason = "not_escape_condition"
            reverse_candidate["backward_allowed"] = bool(reverse_allowed)
            reverse_candidate["backward_gated"] = bool(not reverse_allowed)
            reverse_candidate["backward_gate_reason"] = str(reverse_gate_reason)
            if not reverse_allowed:
                infeasible_reasons = list(reverse_candidate.get("infeasible_reasons", []))
                if "backward_gated" not in infeasible_reasons:
                    infeasible_reasons.append("backward_gated")
                reverse_candidate["infeasible_reasons"] = infeasible_reasons
                reverse_candidate["feasible"] = False
            penalty_value = float(controller_cfg.backward_penalty) if controller_cfg.backward_penalty_enabled else 0.0
            penalty_applied = bool(controller_cfg.backward_penalty_enabled)
            reverse_candidate["backward_penalty_applied"] = penalty_applied
            reverse_candidate["backward_penalty_value"] = penalty_value
            reverse_candidate["total_cost_after_backward_penalty"] = float(reverse_cost_before_penalty + penalty_value)
            reverse_candidate["total_cost"] = float(reverse_candidate["total_cost_after_backward_penalty"])
            if reverse_candidate is backward_candidate:
                backward_candidate_cost_before_penalty = reverse_cost_before_penalty
                backward_candidate_cost_after_penalty = float(reverse_candidate["total_cost_after_backward_penalty"])
                backward_allowed = bool(reverse_allowed)
                backward_gate_reason = str(reverse_gate_reason)

    for candidate in candidates:
        is_backward_action = _is_backward_candidate(candidate, controller_cfg)
        candidate.setdefault("is_backward_action", bool(is_backward_action))
        candidate.setdefault("backward_penalty_applied", False)
        candidate.setdefault("backward_penalty_value", 0.0)
        candidate.setdefault("backward_allowed", True if not is_backward_action else bool(backward_allowed))
        candidate.setdefault("backward_gated", False)
        candidate.setdefault("backward_gate_reason", "not_backward_action")
        candidate.setdefault(
            "total_cost_before_backward_penalty",
            float(candidate.get("total_cost", float("nan"))),
        )
        candidate.setdefault(
            "total_cost_after_backward_penalty",
            float(candidate.get("total_cost", float("nan"))),
        )

    return {
        "backward_policy_config": {
            "backward_penalty_enabled": bool(controller_cfg.backward_penalty_enabled),
            "backward_action_name": str(controller_cfg.backward_action_name),
            "backward_penalty": float(controller_cfg.backward_penalty),
            "backward_gate_enabled": bool(controller_cfg.backward_gate_enabled),
            "dynamic_stop_clearance_threshold": float(controller_cfg.dynamic_stop_clearance_threshold),
            "backward_dynamic_margin": float(controller_cfg.backward_dynamic_margin),
        },
        "backward_candidate_cost_before_penalty": backward_candidate_cost_before_penalty,
        "backward_candidate_cost_after_penalty": backward_candidate_cost_after_penalty,
        "backward_allowed": bool(backward_allowed) if backward_candidate is not None else False,
        "backward_gate_reason": str(backward_gate_reason),
        "stop_min_dynamic_clearance": float(stop_min_dynamic_clearance),
        "stop_dynamic_collision": bool(stop_dynamic_collision),
        "normal_non_stop_static_or_planning_feasible_count": int(normal_non_stop_static_or_planning_feasible_count),
        "normal_non_stop_dynamic_feasible_count": int(normal_non_stop_dynamic_feasible_count),
        "backward_selected": bool(backward_selected),
    }


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
        + float(clearance_cost)
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
            for action_index in effective_runtime_actions(self.controller_cfg)
        ]
        backward_debug = apply_backward_policy(candidates, self.controller_cfg)
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
        backward_debug["backward_selected"] = bool(_is_backward_candidate(chosen, self.controller_cfg))
        action = effective_runtime_actions(self.controller_cfg)[int(chosen["action_index"])]
        return {
            "action_index": int(chosen["action_index"]),
            "action_name": str(action["name"]),
            "vx": float(action["vx"]),
            "vy": float(action["vy"]),
            "v": float(action["vx"]),
            "omega": float(action["omega"]),
            "wz": float(action["omega"]),
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
                **backward_debug,
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
            pose = holonomic_step(
                pose,
                float(action["vx"]),
                float(action["vy"]),
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
            velocity_penalty=velocity_penalty_from_action(float(action["vx"]), float(action["vy"])),
            feasible=feasible,
            controller_cfg=self.controller_cfg,
        )
        return {
            "action_index": int(action_index),
            "action_name": str(action["name"]),
            "vx": float(action["vx"]),
            "vy": float(action["vy"]),
            "v": float(action["vx"]),
            "omega": float(action["omega"]),
            "wz": float(action["omega"]),
            "total_cost_before_backward_penalty": float(score["total_cost"]),
            "total_cost_after_backward_penalty": float(score["total_cost"]),
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
            "backward_penalty_applied": False,
            "backward_penalty_value": 0.0,
            "backward_gated": False,
            "backward_allowed": True,
            "backward_gate_reason": "not_backward_action",
            "is_backward_action": bool(_is_backward_candidate({"action_name": str(action["name"])}, self.controller_cfg)),
        }
