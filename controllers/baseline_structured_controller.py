"""Baseline V4 controller with true cloned dynamic-obstacle rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import numpy as np

from policy_eval.robustness import RobotPoseEKF, covariance_summary, generate_rollout_noise_sequences
from sim2d.config import ACTIONS, DatasetConfig
from sim2d.dynamics import disk_collides_with_occupancy, minimum_static_clearance, unicycle_step
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
    execution_noise_enabled: bool = False
    sigma_v: float = 0.05
    sigma_omega: float = 0.1
    pose_observation_noise_enabled: bool = False
    sigma_obs_x: float = 0.03
    sigma_obs_y: float = 0.03
    sigma_obs_theta: float = 0.05
    num_stochastic_rollouts: int = 8
    risk_beta: float = 1.0
    collision_rate_threshold: float = 0.15


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


def effective_runtime_actions(controller_cfg: StructuredControllerConfig) -> dict[int, dict[str, float | str]]:
    return {int(action_index): ACTIONS[int(action_index)] for action_index in controller_cfg.action_indices}


def _matches_action_name(candidate: dict[str, Any], expected_name: str) -> bool:
    return str(candidate.get("action_name", "")) == str(expected_name)


def _is_stop_candidate(candidate: dict[str, Any]) -> bool:
    return _matches_action_name(candidate, "stop")


def _is_backward_candidate(candidate: dict[str, Any], controller_cfg: StructuredControllerConfig) -> bool:
    return _matches_action_name(candidate, controller_cfg.backward_action_name)


def apply_backward_policy(
    candidates: list[dict[str, Any]],
    controller_cfg: StructuredControllerConfig,
) -> dict[str, object]:
    backward_candidate = next(
        (candidate for candidate in candidates if _is_backward_candidate(candidate, controller_cfg)),
        None,
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
        backward_candidate_cost_before_penalty = float(backward_candidate.get("total_cost_before_backward_penalty", backward_candidate.get("total_cost", float("nan"))))
        backward_allowed = True
        backward_gate_reason = "gate_disabled"
        if controller_cfg.backward_gate_enabled:
            static_escape_allowed = normal_non_stop_static_or_planning_feasible_count == 0
            stop_is_dynamically_unsafe = bool(
                stop_dynamic_collision
                or stop_min_dynamic_clearance < float(controller_cfg.dynamic_stop_clearance_threshold)
            )
            backward_min_dynamic_clearance = float(backward_candidate.get("min_dynamic_clearance", float("-inf")))
            dynamic_escape_allowed = bool(
                stop_is_dynamically_unsafe
                and backward_min_dynamic_clearance
                > stop_min_dynamic_clearance + float(controller_cfg.backward_dynamic_margin)
            )
            if static_escape_allowed:
                backward_allowed = True
                backward_gate_reason = "static_escape"
            elif dynamic_escape_allowed:
                backward_allowed = True
                backward_gate_reason = "dynamic_escape"
            elif stop_is_dynamically_unsafe:
                backward_allowed = False
                backward_gate_reason = "dynamic_gate_failed"
            else:
                backward_allowed = False
                backward_gate_reason = "not_escape_condition"
        backward_candidate["backward_allowed"] = bool(backward_allowed)
        backward_candidate["backward_gated"] = bool(not backward_allowed)
        backward_candidate["backward_gate_reason"] = str(backward_gate_reason)
        if not backward_allowed:
            infeasible_reasons = list(backward_candidate.get("infeasible_reasons", []))
            if "backward_gated" not in infeasible_reasons:
                infeasible_reasons.append("backward_gated")
            backward_candidate["infeasible_reasons"] = infeasible_reasons
            backward_candidate["feasible"] = False
        penalty_value = float(controller_cfg.backward_penalty) if controller_cfg.backward_penalty_enabled else 0.0
        penalty_applied = bool(controller_cfg.backward_penalty_enabled)
        backward_candidate["backward_penalty_applied"] = penalty_applied
        backward_candidate["backward_penalty_value"] = penalty_value
        backward_candidate["total_cost_after_backward_penalty"] = float(backward_candidate_cost_before_penalty + penalty_value)
        backward_candidate["total_cost"] = float(backward_candidate["total_cost_after_backward_penalty"])
        backward_candidate_cost_after_penalty = float(backward_candidate["total_cost_after_backward_penalty"])

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
        self.ekf = RobotPoseEKF.from_pose(np.zeros((3,), dtype=float))
        self.last_ekf_debug: dict[str, object] = {
            "estimated_pose_mean": self.ekf.mean.copy().tolist(),
            "ekf_covariance": covariance_summary(self.ekf.covariance),
        }
        self.rollout_rng = np.random.default_rng()

    def reset(self, *, initial_pose: np.ndarray | None = None, seed: int | None = None) -> None:
        """Reset episode state. The baseline controller is memoryless."""
        pose = np.zeros((3,), dtype=float) if initial_pose is None else np.asarray(initial_pose, dtype=float)
        self.ekf = RobotPoseEKF.from_pose(pose)
        self.last_ekf_debug = {
            "estimated_pose_mean": self.ekf.mean.copy().tolist(),
            "ekf_covariance": covariance_summary(self.ekf.covariance),
            "commanded_control": None,
            "pose_observation": pose.astype(float).tolist(),
            "innovation": [0.0, 0.0, 0.0],
        }
        self.rollout_rng = np.random.default_rng(seed)

    def current_pose_estimate(self) -> np.ndarray:
        return np.asarray(self.ekf.mean, dtype=float).copy()

    def ekf_debug(self) -> dict[str, object]:
        return dict(self.last_ekf_debug)

    def commit_step(
        self,
        *,
        robot_pose_t,
        robot_pose_t1,
        robot_pose_est_t,
        pose_observation,
        dynamic_obstacles_t,
        dynamic_obstacles_t1,
        action_index,
        action_cont,
        path_world,
    ) -> None:
        del robot_pose_t, robot_pose_t1, dynamic_obstacles_t, dynamic_obstacles_t1, action_index, path_world
        self.ekf.mean = np.asarray(robot_pose_est_t, dtype=float).copy()
        prediction_debug = self.ekf.predict(
            commanded_control=np.asarray(action_cont, dtype=float),
            dt=float(self.config.robot_config.dt),
            execution_noise_enabled=bool(self.controller_cfg.execution_noise_enabled),
            sigma_v=float(self.controller_cfg.sigma_v),
            sigma_omega=float(self.controller_cfg.sigma_omega),
        )
        correction_debug = self.ekf.correct(
            pose_observation=np.asarray(pose_observation, dtype=float),
            pose_observation_noise_enabled=bool(self.controller_cfg.pose_observation_noise_enabled),
            sigma_obs_x=float(self.controller_cfg.sigma_obs_x),
            sigma_obs_y=float(self.controller_cfg.sigma_obs_y),
            sigma_obs_theta=float(self.controller_cfg.sigma_obs_theta),
        )
        self.last_ekf_debug = {
            "estimated_pose_mean": self.ekf.mean.copy().tolist(),
            "ekf_covariance": covariance_summary(self.ekf.covariance),
            "predicted_pose_mean": np.asarray(prediction_debug["predicted_mean"], dtype=float).tolist(),
            "pose_observation": np.asarray(correction_debug["observation"], dtype=float).tolist(),
            "innovation": np.asarray(correction_debug["innovation"], dtype=float).tolist(),
            "commanded_control": np.asarray(action_cont, dtype=float).tolist(),
        }

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
        rollout_noise_sequences = None
        if self.controller_cfg.execution_noise_enabled:
            rollout_noise_sequences = generate_rollout_noise_sequences(
                self.rollout_rng,
                num_stochastic_rollouts=int(self.controller_cfg.num_stochastic_rollouts),
                horizon=int(self.controller_cfg.horizon),
                execution_noise_enabled=True,
                sigma_v=float(self.controller_cfg.sigma_v),
                sigma_omega=float(self.controller_cfg.sigma_omega),
            )
        candidates = []
        for action_index in effective_runtime_actions(self.controller_cfg):
            if rollout_noise_sequences is None:
                candidate = self._evaluate_candidate(
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
            else:
                candidate = self._evaluate_candidate_stochastic(
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
                    rollout_noise_sequences=rollout_noise_sequences,
                )
            candidates.append(candidate)
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
                "stochastic_planning_active": bool(rollout_noise_sequences is not None),
                "num_stochastic_rollouts": int(self.controller_cfg.num_stochastic_rollouts),
                "risk_beta": float(self.controller_cfg.risk_beta),
                "collision_rate_threshold": float(self.controller_cfg.collision_rate_threshold),
                "common_random_rollout_control_noise_sequences": rollout_noise_sequences.tolist()
                if rollout_noise_sequences is not None
                else None,
                "estimated_pose_mean": self.current_pose_estimate().tolist(),
                "ekf_covariance": covariance_summary(self.ekf.covariance),
                "last_ekf_update": self.ekf_debug(),
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
        control_noise_sequence: np.ndarray | None = None,
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

        for horizon_index in range(int(self.controller_cfg.horizon)):
            step_noise = (
                np.asarray(control_noise_sequence[horizon_index], dtype=float)
                if control_noise_sequence is not None
                else np.zeros((2,), dtype=float)
            )
            pose = unicycle_step(
                pose,
                float(action["v"]) + float(step_noise[0]),
                float(action["omega"]) + float(step_noise[1]),
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

    def _evaluate_candidate_stochastic(
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
        rollout_noise_sequences: np.ndarray,
    ) -> dict[str, Any]:
        sample_payloads = [
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
                control_noise_sequence=np.asarray(rollout_noise_sequences[sample_index], dtype=float),
            )
            for sample_index in range(int(rollout_noise_sequences.shape[0]))
        ]
        sample_costs = [float(sample["total_cost"]) for sample in sample_payloads]
        collision_flags = [bool(any(reason in {"planning_collision", "static_collision", "dynamic_collision"} for reason in sample["infeasible_reasons"])) for sample in sample_payloads]
        collision_rate = float(np.mean(collision_flags)) if collision_flags else 0.0
        mean_cost = float(np.mean(sample_costs)) if sample_costs else float("nan")
        std_cost = float(np.std(sample_costs)) if sample_costs else float("nan")
        stochastic_score = float(mean_cost + float(self.controller_cfg.risk_beta) * std_cost)
        accepted = bool(collision_rate <= float(self.controller_cfg.collision_rate_threshold))
        first_sample = sample_payloads[0]
        horizon = int(self.controller_cfg.horizon)
        planning_collisions = [
            bool(any(bool(sample["planning_collisions"][step_index]) for sample in sample_payloads))
            for step_index in range(horizon)
        ]
        static_collisions = [
            bool(any(bool(sample["static_collisions"][step_index]) for sample in sample_payloads))
            for step_index in range(horizon)
        ]
        dynamic_collisions = [
            bool(any(bool(sample["dynamic_collisions"][step_index]) for sample in sample_payloads))
            for step_index in range(horizon)
        ]
        min_static_clearance = float(min(float(sample["min_static_clearance"]) for sample in sample_payloads))
        min_dynamic_clearance = float(min(float(sample["min_dynamic_clearance"]) for sample in sample_payloads))
        min_combined_clearance = float(min(float(sample["min_combined_clearance"]) for sample in sample_payloads))
        return {
            "action_index": int(first_sample["action_index"]),
            "action_name": str(first_sample["action_name"]),
            "v": float(first_sample["v"]),
            "omega": float(first_sample["omega"]),
            "total_cost_before_backward_penalty": float(stochastic_score),
            "total_cost_after_backward_penalty": float(stochastic_score),
            "total_cost": float(stochastic_score),
            "goal_progress_cost": float(np.mean([float(sample["goal_progress_cost"]) for sample in sample_payloads])),
            "heading_cost": float(np.mean([float(sample["heading_cost"]) for sample in sample_payloads])),
            "path_distance_cost": float(np.mean([float(sample["path_distance_cost"]) for sample in sample_payloads])),
            "path_tangent_penalty": float(np.mean([float(sample["path_tangent_penalty"]) for sample in sample_payloads])),
            "path_cost": float(np.mean([float(sample["path_cost"]) for sample in sample_payloads])),
            "velocity_cost": float(np.mean([float(sample["velocity_cost"]) for sample in sample_payloads])),
            "clearance_cost": float(np.mean([float(sample["clearance_cost"]) for sample in sample_payloads])),
            "infeasibility_penalty": float(np.mean([float(sample["infeasibility_penalty"]) for sample in sample_payloads])),
            "planning_collisions": planning_collisions,
            "static_collisions": static_collisions,
            "dynamic_collisions": dynamic_collisions,
            "min_static_clearance": min_static_clearance,
            "min_dynamic_clearance": min_dynamic_clearance,
            "min_combined_clearance": min_combined_clearance,
            "feasible": bool(accepted),
            "accepted": bool(accepted),
            "rejection_reason": None if accepted else "stochastic_collision_rate",
            "infeasible_reasons": [] if accepted else ["stochastic_collision_rate"],
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "collision_rate": collision_rate,
            "stochastic_score": float(stochastic_score),
            "num_stochastic_rollouts": int(len(sample_payloads)),
            "stochastic_sample_costs": [float(value) for value in sample_costs],
            "stochastic_sample_collision_flags": [bool(value) for value in collision_flags],
            "backward_penalty_applied": False,
            "backward_penalty_value": 0.0,
            "backward_gated": False,
            "backward_allowed": True,
            "backward_gate_reason": "not_backward_action",
            "is_backward_action": bool(_is_backward_candidate({"action_name": str(first_sample["action_name"])}, self.controller_cfg)),
        }
