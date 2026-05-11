"""Frozen-obstacle repeated-action MPC baseline controller for policy eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import math
import numpy as np

from controllers.baseline_structured_controller import (
    StructuredControllerConfig,
    _is_backward_candidate,
    apply_backward_policy,
    effective_runtime_actions,
    path_tangent_penalty,
    simulate_constant_command_rollout,
    velocity_penalty_from_action,
)
from sim2d.config import ACTIONS, DatasetConfig
from sim2d.obstacles import DynamicObstacle
from sim2d.utils import distance_point_to_polyline, wrap_angle


@dataclass
class MPCStaticControllerConfig:
    horizon: int = 10
    obstacle_model: str = "static"
    debug: bool = False
    score_cfg: StructuredControllerConfig | None = None


class MPCStaticController:
    """Finite-horizon repeated-action MPC with frozen dynamic obstacles."""

    name = "mpc_static"

    def __init__(self, config: DatasetConfig, controller_cfg: MPCStaticControllerConfig):
        if int(controller_cfg.horizon) <= 0:
            raise ValueError("mpc_horizon must be positive")
        if str(controller_cfg.obstacle_model) != "static":
            raise NotImplementedError("mpc_static currently supports only obstacle_model='static'")
        if controller_cfg.score_cfg is None:
            raise ValueError("mpc_static requires StructuredControllerConfig for scoring")
        self.config = config
        self.controller_cfg = controller_cfg
        self.score_cfg = controller_cfg.score_cfg

    def reset(self) -> None:
        """Reset episode state. The MPC-static controller is memoryless."""

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
                action_index=int(action_index),
                robot_pose=np.asarray(robot_pose, dtype=float),
                current_subgoal_world=np.asarray(current_subgoal_world, dtype=float),
                path_world=np.asarray(path_world, dtype=float),
                planning_occupancy=planning_occupancy,
                true_occupancy=true_occupancy,
                resolution=resolution,
                origin=origin,
                occupied_points=occupied_points,
                dynamic_obstacles=dynamic_obstacles,
            )
            for action_index in effective_runtime_actions(self.score_cfg)
        ]
        backward_debug = apply_backward_policy(candidates, self.score_cfg)

        feasible_non_stop = [
            candidate
            for candidate in candidates
            if bool(candidate["feasible"]) and str(candidate["action_name"]) != "stop"
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
        backward_debug["backward_selected"] = bool(_is_backward_candidate(chosen, self.score_cfg))
        chosen["selected"] = True

        action = ACTIONS[int(chosen["action_index"])]
        debug = {
            "mode": "mpc_static",
            "controller_type": "mpc_static",
            "horizon": int(self.controller_cfg.horizon),
            "obstacle_model": str(self.controller_cfg.obstacle_model),
            "weights": {
                "progress": float(self.score_cfg.w_progress),
                "heading": float(self.score_cfg.w_heading),
                "path": float(self.score_cfg.w_path),
                "velocity": float(self.score_cfg.w_velocity),
                "static": float(self.score_cfg.w_static),
                "dynamic": float(self.score_cfg.w_dynamic),
            },
            "min_clearance_clip": float(self.score_cfg.min_clearance_clip),
            "static_collision_penalty": float(self.score_cfg.static_collision_penalty),
            "dynamic_collision_penalty": float(self.score_cfg.dynamic_collision_penalty),
            "selected_action_index": int(chosen["action_index"]),
            "selected_action_name": str(chosen["action_name"]),
            "selected_score": float(chosen["total_cost"]),
            "candidate_count": int(len(candidates)),
            "num_feasible_non_stop": int(len(feasible_non_stop)),
            "num_feasible_all": int(len(feasible_all)),
            "selection_mode": selection_mode,
            **backward_debug,
            "chosen": chosen,
            "candidates": candidates,
        }
        return {
            "action_index": int(chosen["action_index"]),
            "action_name": str(action["name"]),
            "v": float(action["v"]),
            "omega": float(action["omega"]),
            "debug": debug,
            "chosen_action_min_clearance": float(chosen["min_combined_clearance"]),
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
        rollout = simulate_constant_command_rollout(
            robot_pose=robot_pose,
            v=float(action["v"]),
            omega=float(action["omega"]),
            horizon=int(self.controller_cfg.horizon),
            config=self.config,
            planning_occupancy=planning_occupancy,
            true_occupancy=true_occupancy,
            resolution=resolution,
            origin=origin,
            occupied_points=occupied_points,
            dynamic_obstacles=dynamic_obstacles,
        )
        frozen_dynamic = self._frozen_dynamic_rollout(
            robot_rollout_world=rollout["robot_rollout_world"],
            static_clearances=rollout["static_clearances"],
            dynamic_obstacles=dynamic_obstacles,
        )

        pose = np.asarray(rollout["rollout_endpoint_pose"], dtype=float)
        planning_collision = bool(any(bool(value) for value in rollout["planning_collisions"]))
        static_collision = bool(any(bool(value) for value in rollout["static_collisions"]))
        dynamic_collision = bool(frozen_dynamic["dynamic_collision"])
        feasible = not bool(planning_collision or static_collision or dynamic_collision)

        score = self._score_candidate(
            rollout_endpoint_pose=pose,
            current_subgoal_world=current_subgoal_world,
            path_world=path_world,
            min_combined_clearance=float(frozen_dynamic["min_combined_clearance"]),
            velocity_penalty=velocity_penalty_from_action(float(action["v"])),
            static_or_planning_infeasible=bool(planning_collision or static_collision),
            dynamic_collision=dynamic_collision,
        )
        infeasible_reasons: list[str] = []
        if planning_collision:
            infeasible_reasons.append("planning_collision")
        if static_collision:
            infeasible_reasons.append("static_collision")
        if dynamic_collision:
            infeasible_reasons.append("dynamic_collision")
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
            "velocity_penalty": float(score["velocity_cost"]),
            "clearance_cost": float(score["clearance_cost"]),
            "infeasibility_penalty": float(score["infeasibility_penalty"]),
            "planning_collision": bool(planning_collision),
            "static_collision": bool(static_collision),
            "dynamic_collision": bool(dynamic_collision),
            "feasible": bool(feasible),
            "infeasible_reasons": infeasible_reasons,
            "min_static_clearance": float(rollout["min_static_clearance"]),
            "min_dynamic_clearance": float(frozen_dynamic["min_dynamic_clearance"]),
            "min_combined_clearance": float(frozen_dynamic["min_combined_clearance"]),
            "rollout_endpoint_pose": [float(value) for value in pose],
            "robot_rollout_world": rollout["robot_rollout_world"],
            "static_clearances": rollout["static_clearances"],
            "static_collisions": rollout["static_collisions"],
            "planning_collisions": rollout["planning_collisions"],
            "dynamic_clearances": frozen_dynamic["dynamic_clearances"],
            "dynamic_collisions": frozen_dynamic["dynamic_collisions"],
            "combined_clearances": frozen_dynamic["combined_clearances"],
            "selected": False,
        }

    def _frozen_dynamic_rollout(
        self,
        *,
        robot_rollout_world: Sequence[Sequence[float]],
        static_clearances: Sequence[float],
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> dict[str, Any]:
        if not dynamic_obstacles:
            dynamic_clearances = [float("inf")] * len(robot_rollout_world)
            dynamic_collisions = [False] * len(robot_rollout_world)
            combined_clearances = [float(clearance) for clearance in static_clearances]
            return {
                "dynamic_clearances": dynamic_clearances,
                "dynamic_collisions": dynamic_collisions,
                "combined_clearances": combined_clearances,
                "dynamic_collision": False,
                "min_dynamic_clearance": float("inf"),
                "min_combined_clearance": float(np.min(combined_clearances)) if combined_clearances else float("inf"),
            }

        obstacle_positions = [np.asarray(obstacle.position, dtype=float).copy() for obstacle in dynamic_obstacles]
        obstacle_radii = [float(obstacle.radius) for obstacle in dynamic_obstacles]
        robot_radius = float(self.config.robot_config.radius)
        dynamic_clearances: list[float] = []
        dynamic_collisions: list[bool] = []
        combined_clearances: list[float] = []

        for pose, static_clearance in zip(robot_rollout_world, static_clearances):
            robot_xy = np.asarray(pose[:2], dtype=float)
            per_obstacle = [
                float(np.linalg.norm(robot_xy - obstacle_position) - robot_radius - obstacle_radius)
                for obstacle_position, obstacle_radius in zip(obstacle_positions, obstacle_radii)
            ]
            step_dynamic_clearance = float(min(per_obstacle)) if per_obstacle else float("inf")
            step_dynamic_collision = bool(step_dynamic_clearance < 0.0)
            dynamic_clearances.append(step_dynamic_clearance)
            dynamic_collisions.append(step_dynamic_collision)
            combined_clearances.append(float(min(float(static_clearance), step_dynamic_clearance)))

        return {
            "dynamic_clearances": dynamic_clearances,
            "dynamic_collisions": dynamic_collisions,
            "combined_clearances": combined_clearances,
            "dynamic_collision": bool(any(dynamic_collisions)),
            "min_dynamic_clearance": float(np.min(dynamic_clearances)) if dynamic_clearances else float("inf"),
            "min_combined_clearance": float(np.min(combined_clearances)) if combined_clearances else float("inf"),
        }

    def _score_candidate(
        self,
        *,
        rollout_endpoint_pose: np.ndarray,
        current_subgoal_world: np.ndarray,
        path_world: np.ndarray,
        min_combined_clearance: float,
        velocity_penalty: float,
        static_or_planning_infeasible: bool,
        dynamic_collision: bool,
    ) -> dict[str, float]:
        endpoint = np.asarray(rollout_endpoint_pose, dtype=float)
        subgoal = np.asarray(current_subgoal_world, dtype=float)
        goal_progress_cost = float(np.linalg.norm(endpoint[:2] - subgoal))
        heading_to_goal = math.atan2(float(subgoal[1] - endpoint[1]), float(subgoal[0] - endpoint[0]))
        heading_cost = abs(wrap_angle(heading_to_goal - float(endpoint[2])))
        path_distance_cost = float(distance_point_to_polyline(endpoint[:2], path_world))
        tangent_cost = float(path_tangent_penalty(endpoint, path_world))
        path_cost = float(path_distance_cost + tangent_cost)
        clearance_cost = 1.0 / max(float(min_combined_clearance), float(self.score_cfg.min_clearance_clip))
        infeasibility_penalty = (
            (float(self.score_cfg.static_collision_penalty) if static_or_planning_infeasible else 0.0)
            + (float(self.score_cfg.dynamic_collision_penalty) if dynamic_collision else 0.0)
        )
        total_cost = (
            float(self.score_cfg.w_progress) * goal_progress_cost
            + float(self.score_cfg.w_heading) * heading_cost
            + float(self.score_cfg.w_path) * path_cost
            + float(self.score_cfg.w_velocity) * float(velocity_penalty)
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
