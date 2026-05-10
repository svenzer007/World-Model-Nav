"""Continuous-command CBF-QP baseline controller for policy eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import numpy as np

from controllers.baseline_structured_controller import (
    StructuredControllerConfig,
    expert_style_rollout_cost,
    path_tangent_penalty,
    simulate_constant_command_rollout,
    velocity_penalty_from_action,
)
from sim2d.config import ACTIONS, DatasetConfig
from sim2d.obstacles import DynamicObstacle
from sim2d.utils import distance_point_to_polyline, wrap_angle


@dataclass
class CBFQPControllerConfig:
    horizon: int = 10
    cbf_omega_scan_enabled: bool = True
    cbf_omega_candidates: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    cbf_static_rollout_steps: int = 10
    cbf_heading_lookahead_enabled: bool = False
    cbf_heading_lookahead_time: float = 1.0
    cbf_dynamic_margin_priority_enabled: bool = False
    cbf_margin_priority_use_margin_trigger: bool = False
    cbf_margin_priority_threshold: float = 0.17
    cbf_margin_priority_use_distance_trigger: bool = False
    cbf_dynamic_priority_distance_threshold: float = 0.25
    cbf_alpha: float = 1.0
    cbf_safety_margin: float = 0.08
    cbf_k_v: float = 0.4
    cbf_k_omega: float = 1.2
    cbf_w_v: float = 1.0
    cbf_w_omega: float = 0.2
    cbf_v_min: float = -0.2
    cbf_v_max: float = 0.4
    cbf_omega_min: float = -1.0
    cbf_omega_max: float = 1.0
    cbf_constraint_eps: float = 1e-8
    cbf_use_reverse_fallback: bool = False
    cbf_reverse_fallback_speed: float = 0.1
    cbf_debug: bool = False
    cbf_w_goal: float = 2.0
    cbf_w_heading: float = 0.8
    cbf_w_velocity: float = 0.3
    cbf_w_omega_dev: float = 0.0
    cbf_w_cbf_margin: float = 0.5
    cbf_static_collision_penalty: float = 1e6
    cbf_dynamic_infeasible_penalty: float = 1e6
    score_cfg: StructuredControllerConfig | None = None


def infer_cbf_command_bounds() -> tuple[float, float, float, float]:
    velocities = [float(action["v"]) for action in ACTIONS.values()]
    omegas = [float(action["omega"]) for action in ACTIONS.values()]
    if not velocities or not omegas:
        return (-0.2, 0.4, -1.0, 1.0)
    return (min(velocities), max(velocities), min(omegas), max(omegas))


class CBFQPController:
    """Nominal unicycle controller filtered by a distance-based dynamic-obstacle CBF."""

    name = "cbf_qp"

    def __init__(self, config: DatasetConfig, controller_cfg: CBFQPControllerConfig):
        if int(controller_cfg.horizon) <= 0:
            raise ValueError("CBF controller horizon must be positive")
        if int(controller_cfg.cbf_static_rollout_steps) <= 0:
            raise ValueError("CBF static rollout steps must be positive")
        if float(controller_cfg.cbf_heading_lookahead_time) <= 0.0:
            raise ValueError("CBF heading lookahead time must be positive")
        self.config = config
        self.controller_cfg = controller_cfg
        self._previous_obstacle_positions: dict[object, np.ndarray] = {}
        self._obstacle_key_mode = "uninitialized"

    def reset(self) -> None:
        self._previous_obstacle_positions = {}
        self._obstacle_key_mode = "uninitialized"

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
        del robot_pose_t, robot_pose_t1, dynamic_obstacles_t, action_index, action_cont, path_world
        observations = self._snapshot_observations(dynamic_obstacles_t1)
        self._previous_obstacle_positions = {
            observation["key"]: np.asarray(observation["position"], dtype=float).copy()
            for observation in observations
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
        robot_pose = np.asarray(robot_pose, dtype=float)
        current_subgoal_world = np.asarray(current_subgoal_world, dtype=float)
        obstacle_observations = self._object_observations(dynamic_obstacles)
        min_dynamic_boundary_distance = self._min_dynamic_boundary_distance(
            robot_pose=robot_pose,
            dynamic_observations=obstacle_observations,
        )

        u_nom = self._nominal_command(
            robot_pose=robot_pose,
            current_subgoal_world=current_subgoal_world,
        )
        omega_candidates = self._omega_candidates(float(u_nom[1]))
        candidate_debug: list[dict[str, Any]] = []
        selected_candidate: dict[str, Any] | None = None
        feasible_candidates: list[dict[str, Any]] = []
        normal_selected_candidate: dict[str, Any] | None = None
        normal_selected_candidate_index = math.nan
        normal_selected_min_cbf_margin = math.nan
        best_candidate_min_cbf_margin = math.nan
        worst_candidate_min_cbf_margin = math.nan
        cbf_margin_range = math.nan
        margin_trigger_active = False
        distance_trigger_active = False
        priority_active = False
        theta_eff_source = (
            "heading_lookahead_time"
            if bool(self.controller_cfg.cbf_heading_lookahead_enabled)
            else "dt"
        )

        for candidate_index, omega_candidate_raw in enumerate(omega_candidates):
            omega_candidate = float(
                np.clip(
                    float(omega_candidate_raw),
                    float(self.controller_cfg.cbf_omega_min),
                    float(self.controller_cfg.cbf_omega_max),
                )
            )
            theta_eff_horizon = (
                float(self.controller_cfg.cbf_heading_lookahead_time)
                if bool(self.controller_cfg.cbf_heading_lookahead_enabled)
                else float(self.config.robot_config.dt)
            )
            theta_eff = float(robot_pose[2] + omega_candidate * theta_eff_horizon)
            projection = self._project_nominal_command(
                robot_pose=robot_pose,
                theta_eff=theta_eff,
                dynamic_observations=obstacle_observations,
                v_nom=float(u_nom[0]),
            )
            dynamic_cbf_feasible = bool(projection["qp_success"])
            if dynamic_cbf_feasible:
                v_candidate = float(
                    np.clip(
                        float(u_nom[0]),
                        float(projection["v_feasible_lower"]),
                        float(projection["v_feasible_upper"]),
                    )
                )
                dynamic_infeasible_reason = ""
            else:
                v_candidate, _ = self._fallback_command(omega_candidate)
                dynamic_infeasible_reason = str(projection["qp_status"])

            static_rollout = simulate_constant_command_rollout(
                robot_pose=robot_pose,
                v=float(v_candidate),
                omega=float(omega_candidate),
                horizon=int(self.controller_cfg.cbf_static_rollout_steps),
                config=self.config,
                planning_occupancy=planning_occupancy,
                true_occupancy=true_occupancy,
                resolution=resolution,
                origin=origin,
                occupied_points=occupied_points,
                dynamic_obstacles=dynamic_obstacles,
            )
            static_rollout_collision = bool(
                any(bool(value) for value in static_rollout["planning_collisions"])
                or any(bool(value) for value in static_rollout["static_collisions"])
            )
            final_pose = np.asarray(static_rollout["rollout_endpoint_pose"], dtype=float)
            distance_to_goal_after_rollout = float(
                np.linalg.norm(final_pose[:2] - current_subgoal_world[:2])
            )
            desired_heading_after_rollout = math.atan2(
                float(current_subgoal_world[1] - final_pose[1]),
                float(current_subgoal_world[0] - final_pose[0]),
            )
            heading_error_after_rollout = abs(
                wrap_angle(desired_heading_after_rollout - float(final_pose[2]))
            )
            min_cbf_margin = float(projection["min_hdot_plus_alpha_h_at_u_star"])
            candidate_cost = self._candidate_score(
                rollout_endpoint_pose=final_pose,
                current_subgoal_world=current_subgoal_world,
                path_world=np.asarray(path_world, dtype=float),
                static_rollout_collision=static_rollout_collision,
                dynamic_cbf_feasible=dynamic_cbf_feasible,
                min_static_clearance=float(static_rollout["min_static_clearance"]),
                v_star=float(v_candidate),
                omega_candidate=float(omega_candidate),
                omega_nom=float(u_nom[1]),
                min_cbf_margin=min_cbf_margin,
            )
            candidate = {
                "candidate_index": int(candidate_index),
                "action_index": int(0 if self._is_exact_stop(v_candidate, omega_candidate) else -1),
                "action_name": "stop" if self._is_exact_stop(v_candidate, omega_candidate) else "cbf_qp",
                "omega_candidate": float(omega_candidate),
                "theta_eff": float(theta_eff),
                "v_lower": float(projection["v_feasible_lower"]),
                "v_upper": float(projection["v_feasible_upper"]),
                "v_star": float(v_candidate),
                "dynamic_cbf_feasible": bool(dynamic_cbf_feasible),
                "dynamic_infeasible_reason": str(dynamic_infeasible_reason),
                "min_h": float(projection["min_h"]),
                "min_cbf_margin": float(min_cbf_margin),
                "most_critical_obstacle_id": projection["most_critical_obstacle_id"],
                "most_critical_obstacle_position": projection["most_critical_obstacle_position"],
                "most_critical_obstacle_velocity": projection["most_critical_obstacle_velocity"],
                "most_restrictive_obstacle_id": projection["most_restrictive_obstacle_id"],
                "static_rollout_collision": bool(static_rollout_collision),
                "static_rollout_min_clearance": float(static_rollout["min_static_clearance"]),
                "final_rollout_pose": [float(value) for value in final_pose],
                "distance_to_goal_after_rollout": float(distance_to_goal_after_rollout),
                "heading_error_after_rollout": float(heading_error_after_rollout),
                "goal_progress_cost": float(candidate_cost["goal_progress_cost"]),
                "heading_cost": float(candidate_cost["heading_cost"]),
                "path_cost": float(candidate_cost["path_cost"]),
                "velocity_penalty": float(candidate_cost["velocity_penalty"]),
                "static_clearance_cost": float(candidate_cost["static_clearance_cost"]),
                "infeasibility_penalty": float(candidate_cost["infeasibility_penalty"]),
                "static_collision_penalty_applied": float(candidate_cost["static_collision_penalty_applied"]),
                "dynamic_infeasibility_penalty_applied": float(candidate_cost["dynamic_infeasibility_penalty_applied"]),
                "score": float(candidate_cost["final_candidate_score"]),
                "final_candidate_score": float(candidate_cost["final_candidate_score"]),
                "selected": False,
                "robot_rollout_world": static_rollout["robot_rollout_world"],
                "static_clearances": static_rollout["static_clearances"],
                "static_collisions": static_rollout["static_collisions"],
                "planning_collisions": static_rollout["planning_collisions"],
                "dynamic_clearances": static_rollout["dynamic_clearances"],
                "dynamic_collisions": static_rollout["dynamic_collisions"],
                "combined_clearances": static_rollout["combined_clearances"],
                "dynamic_positions_robot": static_rollout["dynamic_positions_robot"],
                "min_static_clearance": float(static_rollout["min_static_clearance"]),
                "min_dynamic_clearance": float(static_rollout["min_dynamic_clearance"]),
                "min_combined_clearance": float(static_rollout["min_combined_clearance"]),
                "feasible": bool(dynamic_cbf_feasible and not static_rollout_collision),
                "infeasible_reasons": (
                    ([] if dynamic_cbf_feasible else ["dynamic_cbf_infeasible"])
                    + (["static_collision"] if static_rollout_collision else [])
                ),
                "qp_success": bool(dynamic_cbf_feasible),
                "qp_status": str(projection["qp_status"]),
                "qp_fallback_used": bool(not dynamic_cbf_feasible),
                "qp_fallback_reason": str(dynamic_infeasible_reason),
                "cbf_constraints_summary": projection["constraints_summary"] if self.controller_cfg.cbf_debug else [],
            }
            candidate_debug.append(candidate)
            if bool(candidate["feasible"]):
                feasible_candidates.append(candidate)

        if feasible_candidates:
            normal_selected_candidate = min(feasible_candidates, key=lambda item: float(item["score"]))
            normal_selected_candidate_index = int(normal_selected_candidate["candidate_index"])
            normal_selected_min_cbf_margin = float(normal_selected_candidate.get("min_cbf_margin", math.nan))
            best_candidate_min_cbf_margin = max(
                float(item["min_cbf_margin"]) for item in feasible_candidates
            )
            worst_candidate_min_cbf_margin = min(
                float(item["min_cbf_margin"]) for item in feasible_candidates
            )
            cbf_margin_range = float(best_candidate_min_cbf_margin - worst_candidate_min_cbf_margin)
            margin_trigger_active = bool(
                bool(self.controller_cfg.cbf_margin_priority_use_margin_trigger)
                and float(normal_selected_min_cbf_margin)
                < float(self.controller_cfg.cbf_margin_priority_threshold)
            )
            distance_trigger_active = bool(
                bool(self.controller_cfg.cbf_margin_priority_use_distance_trigger)
                and float(min_dynamic_boundary_distance)
                < float(self.controller_cfg.cbf_dynamic_priority_distance_threshold)
            )
            priority_active = bool(margin_trigger_active or distance_trigger_active)

        if feasible_candidates and priority_active:
            selected_candidate = min(
                feasible_candidates,
                key=lambda item: (
                    -float(item["min_cbf_margin"]),
                    float(item["static_clearance_cost"]),
                    float(item["path_cost"]),
                    float(item["heading_cost"]),
                    abs(float(item["omega_candidate"]) - float(u_nom[1])),
                ),
            )
            if margin_trigger_active and distance_trigger_active:
                selection_mode = "dynamic_margin_priority_both_triggers"
            elif margin_trigger_active:
                selection_mode = "dynamic_margin_priority_margin_trigger"
            else:
                selection_mode = "dynamic_margin_priority_distance_trigger"
        elif feasible_candidates:
            selected_candidate = normal_selected_candidate
            selection_mode = "feasible_candidates"
        else:
            selected_candidate = min(candidate_debug, key=lambda item: float(item["score"]))
            selection_mode = "fallback_candidates"
        selected_candidate["selected"] = True

        v_star = float(selected_candidate["v_star"])
        omega_star = float(selected_candidate["omega_candidate"])
        qp_success = bool(selected_candidate["dynamic_cbf_feasible"]) and not bool(
            selected_candidate["static_rollout_collision"]
        )
        qp_status = str(
            "selected_feasible"
            if qp_success
            else (
                "static_infeasible"
                if bool(selected_candidate["static_rollout_collision"])
                else str(selected_candidate["qp_status"])
            )
        )
        qp_fallback_used = not qp_success
        qp_fallback_reason = (
            "selected_candidate_infeasible"
            if qp_fallback_used and not str(selected_candidate.get("qp_fallback_reason", ""))
            else str(selected_candidate.get("qp_fallback_reason", ""))
        )

        rollout = {
            "rollout_endpoint_pose": np.asarray(selected_candidate["final_rollout_pose"], dtype=float),
            "robot_rollout_world": selected_candidate["robot_rollout_world"],
            "static_clearances": selected_candidate["static_clearances"],
            "static_collisions": selected_candidate["static_collisions"],
            "planning_collisions": selected_candidate["planning_collisions"],
            "dynamic_clearances": selected_candidate["dynamic_clearances"],
            "dynamic_collisions": selected_candidate["dynamic_collisions"],
            "combined_clearances": selected_candidate["combined_clearances"],
            "dynamic_positions_robot": selected_candidate["dynamic_positions_robot"],
            "feasible": bool(selected_candidate["feasible"]),
            "infeasible_reasons": list(selected_candidate["infeasible_reasons"]),
            "min_static_clearance": float(selected_candidate["min_static_clearance"]),
            "min_dynamic_clearance": float(selected_candidate["min_dynamic_clearance"]),
            "min_combined_clearance": float(selected_candidate["min_combined_clearance"]),
        }
        diagnostic_score = self._diagnostic_score(
            rollout_endpoint_pose=np.asarray(rollout["rollout_endpoint_pose"], dtype=float),
            current_subgoal_world=current_subgoal_world,
            path_world=np.asarray(path_world, dtype=float),
            min_combined_clearance=float(rollout["min_combined_clearance"]),
            v=float(v_star),
            feasible=bool(rollout["feasible"]),
        )
        control_deviation_norm = float(
            np.linalg.norm(np.asarray([v_star, omega_star]) - np.asarray(u_nom))
        )
        qp_objective = 0.5 * (
            float(self.controller_cfg.cbf_w_v) * float((v_star - float(u_nom[0])) ** 2)
            + float(self.controller_cfg.cbf_w_omega) * float((omega_star - float(u_nom[1])) ** 2)
        )
        chosen_debug = {
            "action_index": int(0 if self._is_exact_stop(v_star, omega_star) else -1),
            "action_name": "stop" if self._is_exact_stop(v_star, omega_star) else "cbf_qp",
            "v": float(v_star),
            "omega": float(omega_star),
            "total_cost": float(diagnostic_score["total_cost"]),
            "goal_progress_cost": float(diagnostic_score["goal_progress_cost"]),
            "heading_cost": float(diagnostic_score["heading_cost"]),
            "path_distance_cost": float(diagnostic_score["path_distance_cost"]),
            "path_tangent_penalty": float(diagnostic_score["path_tangent_penalty"]),
            "path_cost": float(diagnostic_score["path_cost"]),
            "velocity_cost": float(diagnostic_score["velocity_cost"]),
            "clearance_cost": float(diagnostic_score["clearance_cost"]),
            "infeasibility_penalty": float(diagnostic_score["infeasibility_penalty"]),
            "robot_rollout_world": rollout["robot_rollout_world"],
            "static_clearances": rollout["static_clearances"],
            "static_collisions": rollout["static_collisions"],
            "planning_collisions": rollout["planning_collisions"],
            "dynamic_clearances": rollout["dynamic_clearances"],
            "dynamic_collisions": rollout["dynamic_collisions"],
            "combined_clearances": rollout["combined_clearances"],
            "dynamic_positions_robot": rollout["dynamic_positions_robot"],
            "feasible": bool(rollout["feasible"]),
            "infeasible_reasons": list(rollout["infeasible_reasons"]),
            "min_static_clearance": float(rollout["min_static_clearance"]),
            "min_dynamic_clearance": float(rollout["min_dynamic_clearance"]),
            "min_combined_clearance": float(rollout["min_combined_clearance"]),
            "control_deviation_norm": control_deviation_norm,
            "cbf_margin": float(selected_candidate["min_cbf_margin"]),
            "static_rollout_collision": bool(selected_candidate["static_rollout_collision"]),
            "theta_eff": float(selected_candidate["theta_eff"]),
            "v_lower": float(selected_candidate["v_lower"]),
            "v_upper": float(selected_candidate["v_upper"]),
            "qp_success": bool(qp_success),
            "qp_status": qp_status,
            "qp_fallback_used": bool(qp_fallback_used),
            "qp_fallback_reason": str(qp_fallback_reason),
            "score": float(selected_candidate["score"]),
        }
        debug = {
            "mode": "cbf_qp",
            "controller_type": "cbf_qp",
            "obstacle_key_mode": str(self._obstacle_key_mode),
            "cbf_omega_scan_enabled": bool(self.controller_cfg.cbf_omega_scan_enabled),
            "cbf_omega_candidates": [float(value) for value in omega_candidates],
            "cbf_heading_lookahead_enabled": bool(self.controller_cfg.cbf_heading_lookahead_enabled),
            "cbf_heading_lookahead_time": float(self.controller_cfg.cbf_heading_lookahead_time),
            "theta_eff_source": theta_eff_source,
            "cbf_dynamic_margin_priority_enabled_deprecated": bool(self.controller_cfg.cbf_dynamic_margin_priority_enabled),
            "cbf_margin_priority_use_margin_trigger": bool(self.controller_cfg.cbf_margin_priority_use_margin_trigger),
            "cbf_margin_priority_threshold": float(self.controller_cfg.cbf_margin_priority_threshold),
            "cbf_margin_priority_use_distance_trigger": bool(self.controller_cfg.cbf_margin_priority_use_distance_trigger),
            "cbf_dynamic_priority_distance_threshold": float(self.controller_cfg.cbf_dynamic_priority_distance_threshold),
            "normal_selected_candidate_index": normal_selected_candidate_index,
            "normal_selected_min_cbf_margin": normal_selected_min_cbf_margin,
            "best_candidate_min_cbf_margin": best_candidate_min_cbf_margin,
            "worst_candidate_min_cbf_margin": worst_candidate_min_cbf_margin,
            "cbf_margin_range": cbf_margin_range,
            "min_dynamic_boundary_distance": float(min_dynamic_boundary_distance),
            "margin_trigger_active": bool(margin_trigger_active),
            "distance_trigger_active": bool(distance_trigger_active),
            "priority_active": bool(priority_active),
            "high_dynamic_risk": bool(priority_active),
            "selected_omega_candidate": float(omega_star),
            "selected_candidate_index": int(selected_candidate["candidate_index"]),
            "candidate_count": int(len(candidate_debug)),
            "omega_nom": float(u_nom[1]),
            "v_nom": float(u_nom[0]),
            "v_star": float(v_star),
            "selected_score": float(selected_candidate["score"]),
            "robot_pose": [float(value) for value in robot_pose],
            "local_goal": [float(value) for value in current_subgoal_world],
            "u_nom": {"v": float(u_nom[0]), "omega": float(u_nom[1])},
            "u_star": {"v": float(v_star), "omega": float(omega_star)},
            "qp_success": bool(qp_success),
            "qp_status": qp_status,
            "qp_objective": float(qp_objective),
            "qp_fallback_used": bool(qp_fallback_used),
            "qp_fallback_reason": str(qp_fallback_reason),
            "selection_mode": selection_mode,
            "num_cbf_constraints": int(len(obstacle_observations)),
            "v_feasible_lower": float(selected_candidate["v_lower"]),
            "v_feasible_upper": float(selected_candidate["v_upper"]),
            "min_h": float(selected_candidate["min_h"]),
            "min_hdot_plus_alpha_h": float(selected_candidate["min_cbf_margin"]),
            "most_critical_obstacle_id": selected_candidate["most_critical_obstacle_id"],
            "most_critical_obstacle_position": selected_candidate["most_critical_obstacle_position"],
            "most_critical_obstacle_velocity": selected_candidate["most_critical_obstacle_velocity"],
            "most_restrictive_obstacle_id": selected_candidate["most_restrictive_obstacle_id"],
            "control_deviation_norm": control_deviation_norm,
            "cbf_constraints_summary": (
                selected_candidate["cbf_constraints_summary"] if self.controller_cfg.cbf_debug else []
            ),
            "candidate_debug": candidate_debug,
            "chosen": chosen_debug,
            "candidates": candidate_debug,
        }
        return {
            "action_index": int(chosen_debug["action_index"]),
            "action_name": str(chosen_debug["action_name"]),
            "v": float(v_star),
            "omega": float(omega_star),
            "debug": debug,
            "chosen_action_min_clearance": float(rollout["min_combined_clearance"]),
            "emergency_override_used": bool(qp_fallback_used),
        }

    def _omega_candidates(self, omega_nom: float) -> list[float]:
        if not bool(self.controller_cfg.cbf_omega_scan_enabled):
            return [
                float(
                    np.clip(
                        float(omega_nom),
                        float(self.controller_cfg.cbf_omega_min),
                        float(self.controller_cfg.cbf_omega_max),
                    )
                )
            ]
        return [
            float(
                np.clip(
                    float(candidate),
                    float(self.controller_cfg.cbf_omega_min),
                    float(self.controller_cfg.cbf_omega_max),
                )
            )
            for candidate in self.controller_cfg.cbf_omega_candidates
        ]

    def _min_dynamic_boundary_distance(
        self,
        *,
        robot_pose: np.ndarray,
        dynamic_observations: list[dict[str, Any]],
    ) -> float:
        if not dynamic_observations:
            return float("inf")
        robot_position = np.asarray(robot_pose[:2], dtype=float)
        boundary_distances = []
        for observation in dynamic_observations:
            p_obs = np.asarray(observation["position"], dtype=float)
            d_safe = (
                float(self.config.robot_config.radius)
                + float(observation["radius"])
                + float(self.controller_cfg.cbf_safety_margin)
            )
            boundary_distances.append(float(np.linalg.norm(robot_position - p_obs) - d_safe))
        return float(min(boundary_distances)) if boundary_distances else float("inf")

    def _nominal_command(
        self,
        *,
        robot_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
    ) -> np.ndarray:
        goal_delta = np.asarray(current_subgoal_world[:2], dtype=float) - np.asarray(
            robot_pose[:2], dtype=float
        )
        distance_to_goal = float(np.linalg.norm(goal_delta))
        desired_heading = math.atan2(float(goal_delta[1]), float(goal_delta[0]))
        heading_error = wrap_angle(desired_heading - float(robot_pose[2]))
        v_nom = float(
            np.clip(
                float(self.controller_cfg.cbf_k_v) * distance_to_goal,
                float(self.controller_cfg.cbf_v_min),
                float(self.controller_cfg.cbf_v_max),
            )
        )
        omega_nom = float(
            np.clip(
                float(self.controller_cfg.cbf_k_omega) * float(heading_error),
                float(self.controller_cfg.cbf_omega_min),
                float(self.controller_cfg.cbf_omega_max),
            )
        )
        return np.asarray([v_nom, omega_nom], dtype=float)

    def _project_nominal_command(
        self,
        *,
        robot_pose: np.ndarray,
        theta_eff: float,
        dynamic_observations: list[dict[str, Any]],
        v_nom: float,
    ) -> dict[str, Any]:
        cos_theta = float(np.cos(theta_eff))
        sin_theta = float(np.sin(theta_eff))
        heading = np.asarray([cos_theta, sin_theta], dtype=float)
        v_lower = float(self.controller_cfg.cbf_v_min)
        v_upper = float(self.controller_cfg.cbf_v_max)
        num_constraints = 0
        min_h = float("inf")
        min_margin_nom = float("inf")
        min_margin_star = float("inf")
        most_critical_obstacle_id: object | None = None
        most_critical_obstacle_position: list[float] | None = None
        most_critical_obstacle_velocity: list[float] | None = None
        most_restrictive_obstacle_id: object | None = None
        qp_status = "optimal"
        qp_success = True
        constraints_summary: list[dict[str, Any]] = []
        restrictive_distance = -float("inf")

        for observation in dynamic_observations:
            num_constraints += 1
            p_obs = np.asarray(observation["position"], dtype=float)
            v_obs = np.asarray(observation["velocity"], dtype=float)
            delta = np.asarray(robot_pose[:2], dtype=float) - p_obs
            d_safe = (
                float(self.config.robot_config.radius)
                + float(observation["radius"])
                + float(self.controller_cfg.cbf_safety_margin)
            )
            h_value = float(np.dot(delta, delta) - d_safe**2)
            A_i = float(2.0 * np.dot(delta, heading))
            b_i = float(
                2.0 * np.dot(delta, v_obs)
                - float(self.controller_cfg.cbf_alpha) * h_value
            )
            hdot_nom = float(A_i * v_nom - 2.0 * np.dot(delta, v_obs))
            margin_nom = float(hdot_nom + float(self.controller_cfg.cbf_alpha) * h_value)

            bound_type = "none"
            bound_value = float("nan")
            if A_i > float(self.controller_cfg.cbf_constraint_eps):
                bound_type = "lower"
                bound_value = float(b_i / A_i)
                old_lower = v_lower
                v_lower = max(v_lower, bound_value)
                if v_lower > old_lower:
                    tightening = float(v_lower - old_lower)
                    if tightening > restrictive_distance:
                        restrictive_distance = tightening
                        most_restrictive_obstacle_id = observation["id"]
            elif A_i < -float(self.controller_cfg.cbf_constraint_eps):
                bound_type = "upper"
                bound_value = float(b_i / A_i)
                old_upper = v_upper
                v_upper = min(v_upper, bound_value)
                if old_upper > v_upper:
                    tightening = float(old_upper - v_upper)
                    if tightening > restrictive_distance:
                        restrictive_distance = tightening
                        most_restrictive_obstacle_id = observation["id"]
            elif b_i > 0.0:
                qp_success = False
                qp_status = "infeasible_zero_coefficient"
                most_restrictive_obstacle_id = observation["id"]

            if h_value < min_h:
                min_h = h_value
                most_critical_obstacle_id = observation["id"]
                most_critical_obstacle_position = [float(value) for value in p_obs]
                most_critical_obstacle_velocity = [float(value) for value in v_obs]

            constraints_summary.append(
                {
                    "obstacle_id": observation["id"],
                    "p_obs": [float(value) for value in p_obs],
                    "v_obs_causal": [float(value) for value in v_obs],
                    "radius": float(observation["radius"]),
                    "d_safe": float(d_safe),
                    "h": float(h_value),
                    "A_i": float(A_i),
                    "b_i": float(b_i),
                    "velocity_source": str(observation.get("velocity_source", "unknown")),
                    "bound_type_from_constraint": bound_type,
                    "bound_value_from_constraint": bound_value,
                    "hdot_at_u_nom": float(hdot_nom),
                    "hdot_plus_alpha_h_at_u_nom": float(margin_nom),
                    "constraint_satisfied_at_u_nom": bool(margin_nom >= 0.0),
                }
            )
            min_margin_nom = min(min_margin_nom, margin_nom)

        if qp_success and v_lower > v_upper:
            qp_success = False
            qp_status = "infeasible_interval"

        v_candidate = (
            float(np.clip(float(v_nom), float(v_lower), float(v_upper)))
            if qp_success
            else 0.0
        )
        for constraint in constraints_summary:
            p_obs = np.asarray(constraint["p_obs"], dtype=float)
            v_obs = np.asarray(constraint["v_obs_causal"], dtype=float)
            delta = np.asarray(robot_pose[:2], dtype=float) - p_obs
            h_value = float(constraint["h"])
            A_i = float(constraint["A_i"])
            hdot_star = float(A_i * v_candidate - 2.0 * np.dot(delta, v_obs))
            margin_star = float(
                hdot_star + float(self.controller_cfg.cbf_alpha) * h_value
            )
            constraint["hdot_at_u_star"] = float(hdot_star)
            constraint["hdot_plus_alpha_h_at_u_star"] = float(margin_star)
            constraint["constraint_satisfied_at_u_star"] = bool(margin_star >= 0.0)
            min_margin_star = min(min_margin_star, margin_star)

        return {
            "qp_success": bool(qp_success),
            "qp_status": str(qp_status),
            "v_feasible_lower": float(v_lower),
            "v_feasible_upper": float(v_upper),
            "num_constraints": int(num_constraints),
            "min_h": float(min_h),
            "min_hdot_plus_alpha_h_at_u_nom": float(min_margin_nom),
            "min_hdot_plus_alpha_h_at_u_star": float(min_margin_star),
            "most_critical_obstacle_id": most_critical_obstacle_id,
            "most_critical_obstacle_position": most_critical_obstacle_position,
            "most_critical_obstacle_velocity": most_critical_obstacle_velocity,
            "most_restrictive_obstacle_id": most_restrictive_obstacle_id,
            "constraints_summary": constraints_summary,
        }

    def _fallback_command(self, omega_nom: float) -> tuple[float, float]:
        if bool(self.controller_cfg.cbf_use_reverse_fallback):
            return (
                float(
                    max(
                        float(self.controller_cfg.cbf_v_min),
                        -float(self.controller_cfg.cbf_reverse_fallback_speed),
                    )
                ),
                float(
                    np.clip(
                        float(omega_nom),
                        float(self.controller_cfg.cbf_omega_min),
                        float(self.controller_cfg.cbf_omega_max),
                    )
                ),
            )
        return (
            0.0,
            float(
                np.clip(
                    float(omega_nom),
                    float(self.controller_cfg.cbf_omega_min),
                    float(self.controller_cfg.cbf_omega_max),
                )
            ),
        )

    def _candidate_score(
        self,
        *,
        rollout_endpoint_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        static_rollout_collision: bool,
        dynamic_cbf_feasible: bool,
        min_static_clearance: float,
        v_star: float,
        omega_candidate: float,
        omega_nom: float,
        min_cbf_margin: float,
    ) -> dict[str, float]:
        score_cfg = self.controller_cfg.score_cfg
        endpoint = np.asarray(rollout_endpoint_pose, dtype=float)
        subgoal = np.asarray(current_subgoal_world, dtype=float)
        goal_progress_cost = float(np.linalg.norm(endpoint[:2] - subgoal[:2]))
        heading_to_goal = math.atan2(
            float(subgoal[1] - endpoint[1]),
            float(subgoal[0] - endpoint[0]),
        )
        heading_cost = abs(wrap_angle(heading_to_goal - float(endpoint[2])))
        path_distance_cost = float(
            distance_point_to_polyline(endpoint[:2], np.asarray(path_world, dtype=float))
        )
        path_cost = float(
            path_distance_cost
            + path_tangent_penalty(endpoint, np.asarray(path_world, dtype=float))
        )
        velocity_penalty = float(velocity_penalty_from_action(float(v_star)))
        score_cfg = self.controller_cfg.score_cfg
        w_goal = (
            float(score_cfg.w_progress)
            if score_cfg is not None
            else float(self.controller_cfg.cbf_w_goal)
        )
        w_heading = (
            float(score_cfg.w_heading)
            if score_cfg is not None
            else float(self.controller_cfg.cbf_w_heading)
        )
        w_path = float(score_cfg.w_path) if score_cfg is not None else 1.1
        w_velocity = (
            float(score_cfg.w_velocity)
            if score_cfg is not None
            else float(self.controller_cfg.cbf_w_velocity)
        )
        static_penalty = (
            float(score_cfg.static_collision_penalty)
            if score_cfg is not None
            else float(self.controller_cfg.cbf_static_collision_penalty)
        )
        dynamic_penalty = (
            float(score_cfg.dynamic_collision_penalty)
            if score_cfg is not None
            else float(self.controller_cfg.cbf_dynamic_infeasible_penalty)
        )
        min_clearance_clip = (
            float(score_cfg.min_clearance_clip) if score_cfg is not None else 0.05
        )
        static_clearance_cost = float(
            1.0 / max(float(min_static_clearance), min_clearance_clip)
        )
        static_collision_penalty_applied = static_penalty * float(bool(static_rollout_collision))
        dynamic_infeasibility_penalty_applied = dynamic_penalty * float(
            not bool(dynamic_cbf_feasible)
        )
        infeasibility_penalty = float(
            static_collision_penalty_applied + dynamic_infeasibility_penalty_applied
        )
        margin_term = (
            float(min_cbf_margin)
            if np.isfinite(float(min_cbf_margin))
            else 0.0
        )
        final_candidate_score = float(
            w_goal * float(goal_progress_cost)
            + w_heading * float(heading_cost)
            + w_path * float(path_cost)
            + w_velocity * float(velocity_penalty)
            + float(static_clearance_cost)
            + float(infeasibility_penalty)
            + float(self.controller_cfg.cbf_w_omega_dev)
            * abs(float(omega_candidate) - float(omega_nom))
            - float(self.controller_cfg.cbf_w_cbf_margin) * margin_term
        )
        return {
            "goal_progress_cost": float(goal_progress_cost),
            "heading_cost": float(heading_cost),
            "path_cost": float(path_cost),
            "velocity_penalty": float(velocity_penalty),
            "static_clearance_cost": float(static_clearance_cost),
            "infeasibility_penalty": float(infeasibility_penalty),
            "static_collision_penalty_applied": float(static_collision_penalty_applied),
            "dynamic_infeasibility_penalty_applied": float(dynamic_infeasibility_penalty_applied),
            "final_candidate_score": float(final_candidate_score),
        }

    def _diagnostic_score(
        self,
        *,
        rollout_endpoint_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        min_combined_clearance: float,
        v: float,
        feasible: bool,
    ) -> dict[str, Any]:
        score_cfg = self.controller_cfg.score_cfg or StructuredControllerConfig(
            horizon=int(self.controller_cfg.horizon)
        )
        return expert_style_rollout_cost(
            rollout_endpoint_pose=np.asarray(rollout_endpoint_pose, dtype=float),
            current_subgoal_world=np.asarray(current_subgoal_world, dtype=float),
            path_world=np.asarray(path_world, dtype=float),
            min_combined_clearance=float(min_combined_clearance),
            velocity_penalty=velocity_penalty_from_action(float(v)),
            feasible=bool(feasible),
            controller_cfg=score_cfg,
        )

    def _object_observations(
        self,
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> list[dict[str, Any]]:
        key_mode = "obstacle_id"
        observations: list[dict[str, Any]] = []
        for slot_index, obstacle in enumerate(dynamic_obstacles):
            obstacle_id = getattr(obstacle, "obstacle_id", None)
            if obstacle_id is None:
                key_mode = "slot_index"
                obstacle_key: object = int(slot_index)
            else:
                obstacle_key = str(obstacle_id)

            position = np.asarray(obstacle.position, dtype=float)
            if obstacle_key in self._previous_obstacle_positions and float(self.config.robot_config.dt) > 0.0:
                velocity = (
                    position
                    - np.asarray(self._previous_obstacle_positions[obstacle_key], dtype=float)
                ) / float(self.config.robot_config.dt)
                velocity_source = "finite_difference"
            else:
                velocity = np.asarray(
                    getattr(obstacle, "velocity", np.zeros(2, dtype=float)),
                    dtype=float,
                ).copy()
                if velocity.shape != (2,):
                    velocity = np.zeros(2, dtype=float)
                velocity_source = (
                    "snapshot_velocity"
                    if np.linalg.norm(velocity) > 0.0
                    else "zero_bootstrap"
                )

            observations.append(
                {
                    "id": obstacle_id if obstacle_id is not None else int(slot_index),
                    "key": obstacle_key,
                    "position": position,
                    "velocity": velocity,
                    "radius": float(obstacle.radius),
                    "slot_index": int(slot_index),
                    "velocity_source": velocity_source,
                }
            )

        if observations:
            self._obstacle_key_mode = key_mode
            return observations
        self._obstacle_key_mode = "no_obstacles"
        return observations

    def _snapshot_observations(
        self,
        dynamic_snapshots: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        key_mode = "obstacle_id"
        observations: list[dict[str, Any]] = []
        for slot_index, snapshot in enumerate(dynamic_snapshots):
            obstacle_id = snapshot.get("obstacle_id")
            if obstacle_id is None:
                key_mode = "slot_index"
                obstacle_key: object = int(slot_index)
            else:
                obstacle_key = str(obstacle_id)
            observations.append(
                {
                    "id": obstacle_id if obstacle_id is not None else int(slot_index),
                    "key": obstacle_key,
                    "position": np.asarray(
                        snapshot.get("position", [0.0, 0.0]),
                        dtype=float,
                    ),
                }
            )
        if observations:
            self._obstacle_key_mode = key_mode
        return observations

    @staticmethod
    def _is_exact_stop(v: float, omega: float, eps: float = 1e-9) -> bool:
        return abs(float(v)) <= float(eps) and abs(float(omega)) <= float(eps)
