"""Learned V4 controller using GRU-predicted robot-relative obstacle rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from controllers.baseline_structured_controller import (
    StructuredControllerConfig,
    dynamic_clearances_from_positions,
    expert_style_rollout_cost,
    planning_point_collision,
    robot_frame_points,
    velocity_penalty_from_action,
)
from models import StructuredDynamicsConfig, StructuredDynamicsModel
from sim2d.config import ACTIONS, DatasetConfig
from sim2d.dynamics import disk_collides_with_occupancy, minimum_static_clearance, unicycle_step
from sim2d.obstacles import DynamicObstacle
from sim2d.waypoint import compute_local_subgoal


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def clone_hidden_state(hidden):
    if hidden is None:
        return None
    if torch.is_tensor(hidden):
        return hidden.clone()
    if isinstance(hidden, (tuple, list)):
        return tuple(x.clone() for x in hidden)
    raise TypeError(f"Unsupported hidden state type: {type(hidden)!r}")


def detach_hidden_state(hidden):
    if hidden is None:
        return None
    if torch.is_tensor(hidden):
        return hidden.detach()
    if isinstance(hidden, (tuple, list)):
        return tuple(x.detach() for x in hidden)
    raise TypeError(f"Unsupported hidden state type: {type(hidden)!r}")


class LearnedStructuredController:
    """Roll out the structured dynamics model in robot frame for dynamic scoring."""

    name = "learned_structured"

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: DatasetConfig,
        controller_cfg: StructuredControllerConfig,
        *,
        device: str = "auto",
    ):
        self.config = config
        self.controller_cfg = controller_cfg
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"{checkpoint_path} does not look like a full training checkpoint")
        model_cfg = dict(checkpoint.get("config", {}).get("model", {}))
        if bool(model_cfg.get("use_lidar", False)):
            raise ValueError("learned structured controller does not support checkpoints with use_lidar=true")
        self.model = StructuredDynamicsModel(StructuredDynamicsConfig(**model_cfg)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.use_goal = bool(model_cfg.get("use_goal", True))
        self.h_real = None
        self.canonical_ids: list[str] | None = None
        self.current_true_state: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self.h_real = None
        self.canonical_ids = None
        self.current_true_state = None

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
        ordered_obstacles = self._order_obstacle_objects(dynamic_obstacles)
        current_state = self._current_state_for_decision(robot_pose=robot_pose, path_world=path_world, dynamic_obstacles=ordered_obstacles)
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
                dynamic_obstacles=ordered_obstacles,
                current_state=current_state,
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
                "mode": "learned_structured_rollout",
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
        ordered_t = self._order_snapshot_dicts(dynamic_obstacles_t)
        ordered_t1 = self._order_snapshot_dicts(dynamic_obstacles_t1)
        state_t = self._state_from_transition(
            robot_pose_prev=np.asarray(robot_pose_t, dtype=float),
            robot_pose_curr=np.asarray(robot_pose_t1, dtype=float),
            current_snapshots=ordered_t,
            future_snapshots=ordered_t1,
            path_world=path_world,
        )
        action_cont_np = np.asarray(action_cont, dtype=np.float32)
        action_tensor = torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        action_cont_tensor = torch.from_numpy(action_cont_np[None]).to(self.device)
        obstacle_pos = torch.from_numpy(state_t["pos_rel"][None]).to(self.device)
        obstacle_vel = torch.from_numpy(state_t["vel_rel"][None]).to(self.device)
        radii = torch.from_numpy(state_t["radii"][None]).to(self.device)
        goal_tensor = torch.from_numpy(state_t["goal"][None]).to(self.device) if self.use_goal else None
        with torch.no_grad():
            _, new_hidden = self.model.forward_step(
                obstacle_pos=obstacle_pos,
                obstacle_vel=obstacle_vel,
                radii=radii,
                action_index=action_tensor,
                action_cont=action_cont_tensor,
                goal=goal_tensor,
                lidar=None,
                hidden=self.h_real,
            )
        self.h_real = detach_hidden_state(new_hidden)
        self.current_true_state = self._next_state_from_snapshots(
            robot_pose_t=np.asarray(robot_pose_t, dtype=float),
            robot_pose_t1=np.asarray(robot_pose_t1, dtype=float),
            next_snapshots=ordered_t1,
            action_cont=action_cont_np,
            path_world=path_world,
        )

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
        current_state: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        action = ACTIONS[int(action_index)]
        pose = np.asarray(robot_pose, dtype=float).copy()
        pos_rel_current = np.asarray(current_state["pos_rel"], dtype=np.float32).copy()
        vel_rel_current = np.asarray(current_state["vel_rel"], dtype=np.float32).copy()
        radii = np.asarray(current_state["radii"], dtype=np.float32).copy()
        h_candidate = clone_hidden_state(self.h_real)
        true_obstacles = [obstacle.clone() for obstacle in dynamic_obstacles]

        robot_rollout_world: list[list[float]] = []
        static_clearances: list[float] = []
        static_collisions: list[bool] = []
        dynamic_clearances: list[float] = []
        dynamic_collisions: list[bool] = []
        planning_collisions: list[bool] = []
        combined_clearances: list[float] = []
        predicted_positions_robot: list[list[list[float]]] = []
        predicted_velocities_robot: list[list[list[float]]] = []
        true_positions_robot: list[list[list[float]]] = []
        true_dynamic_clearances: list[float] = []
        drift_per_obstacle: list[list[float]] = []
        drift_mean: list[float] = []
        drift_max: list[float] = []
        pred_risk: list[float] = []
        collision_next_probs: list[float] = []
        collision_within_3_probs: list[float] = []
        collision_within_5_probs: list[float] = []
        infeasible_reasons: list[str] = []

        action_index_tensor = torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        action_cont_tensor = torch.tensor([[float(action["v"]), float(action["omega"])]], dtype=torch.float32, device=self.device)

        for _ in range(int(self.controller_cfg.horizon)):
            pose = unicycle_step(
                pose,
                float(action["v"]),
                float(action["omega"]),
                float(self.config.robot_config.dt),
            )
            waypoint = compute_local_subgoal(pose, path_world, self.config.lookahead_distance)
            goal_features = np.asarray(waypoint["goal_features"], dtype=np.float32)
            obstacle_pos_tensor = torch.from_numpy(pos_rel_current[None]).to(self.device)
            obstacle_vel_tensor = torch.from_numpy(vel_rel_current[None]).to(self.device)
            radii_tensor = torch.from_numpy(radii[None]).to(self.device)
            goal_tensor = torch.from_numpy(goal_features[None]).to(self.device) if self.use_goal else None

            with torch.no_grad():
                outputs, h_candidate = self.model.forward_step(
                    obstacle_pos=obstacle_pos_tensor,
                    obstacle_vel=obstacle_vel_tensor,
                    radii=radii_tensor,
                    action_index=action_index_tensor,
                    action_cont=action_cont_tensor,
                    goal=goal_tensor,
                    lidar=None,
                    hidden=h_candidate,
                )
            pred_delta_rel = outputs["pred_delta_rel"][0].detach().cpu().numpy().astype(np.float32)
            pos_rel_next = pos_rel_current + pred_delta_rel
            vel_rel_next = (pos_rel_next - pos_rel_current) / float(self.config.robot_config.dt)

            predicted_clearances = dynamic_clearances_from_positions(
                pos_rel_next.astype(float),
                radii.astype(float),
                self.config.robot_config.radius,
            )
            predicted_positions_robot.append(pos_rel_next.astype(float).tolist())
            predicted_velocities_robot.append(vel_rel_next.astype(float).tolist())
            dynamic_clearances.append(float(np.min(predicted_clearances)) if predicted_clearances.size else float("inf"))
            dynamic_collisions.append(bool(np.any(predicted_clearances <= 0.0)))
            planning_collisions.append(bool(planning_point_collision(pose[:2], planning_occupancy, resolution, origin)))

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
            static_clearances.append(static_clearance)
            static_collisions.append(static_collision)
            combined_clearances.append(float(min(static_clearance, dynamic_clearances[-1])))
            robot_rollout_world.append([float(value) for value in pose])

            for obstacle in true_obstacles:
                obstacle.step(float(self.config.robot_config.dt))
            true_world = np.asarray([obstacle.position.copy() for obstacle in true_obstacles], dtype=float)
            true_robot = robot_frame_points(pose, true_world)
            true_clearances = dynamic_clearances_from_positions(true_robot, radii.astype(float), self.config.robot_config.radius)
            true_positions_robot.append(true_robot.astype(float).tolist())
            true_dynamic_clearances.append(float(np.min(true_clearances)) if true_clearances.size else float("inf"))

            drift = np.linalg.norm(pos_rel_next.astype(float) - true_robot.astype(float), axis=1)
            drift_per_obstacle.append(drift.astype(float).tolist())
            drift_mean.append(float(np.mean(drift)) if drift.size else 0.0)
            drift_max.append(float(np.max(drift)) if drift.size else 0.0)

            pred_risk.append(float(outputs["pred_risk"][0].detach().cpu().item()))
            collision_next_probs.append(float(torch.sigmoid(outputs["collision_next_logits"][0]).detach().cpu().item()))
            collision_within_3_probs.append(float(torch.sigmoid(outputs["collision_within_3_logits"][0]).detach().cpu().item()))
            collision_within_5_probs.append(float(torch.sigmoid(outputs["collision_within_5_logits"][0]).detach().cpu().item()))

            pos_rel_current = pos_rel_next
            vel_rel_current = vel_rel_next

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
            "predicted_positions_robot": predicted_positions_robot,
            "predicted_relative_velocities": predicted_velocities_robot,
            "true_positions_robot": true_positions_robot,
            "true_dynamic_clearances": true_dynamic_clearances,
            "drift_per_obstacle": drift_per_obstacle,
            "drift_mean_per_step": drift_mean,
            "drift_max_per_step": drift_max,
            "pred_risk": pred_risk,
            "collision_next_probs": collision_next_probs,
            "collision_within_3_probs": collision_within_3_probs,
            "collision_within_5_probs": collision_within_5_probs,
            "feasible": bool(feasible),
            "infeasible_reasons": infeasible_reasons,
            "min_static_clearance": float(np.min(static_clearances)) if static_clearances else float("inf"),
            "min_dynamic_clearance": float(np.min(dynamic_clearances)) if dynamic_clearances else float("inf"),
            "min_combined_clearance": float(np.min(combined_clearances)) if combined_clearances else float("inf"),
        }

    def _current_state_for_decision(
        self,
        *,
        robot_pose: np.ndarray,
        path_world: np.ndarray,
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> dict[str, np.ndarray]:
        if self.current_true_state is not None:
            return {
                "pos_rel": np.asarray(self.current_true_state["pos_rel"], dtype=np.float32).copy(),
                "vel_rel": np.asarray(self.current_true_state["vel_rel"], dtype=np.float32).copy(),
                "radii": np.asarray(self.current_true_state["radii"], dtype=np.float32).copy(),
                "goal": np.asarray(self.current_true_state["goal"], dtype=np.float32).copy(),
            }

        positions_world = np.asarray([obstacle.position.copy() for obstacle in dynamic_obstacles], dtype=float)
        velocities_world = np.asarray([np.asarray(obstacle.velocity, dtype=float) for obstacle in dynamic_obstacles], dtype=float)
        radii = np.asarray([float(obstacle.radius) for obstacle in dynamic_obstacles], dtype=np.float32)
        pos_rel = robot_frame_points(robot_pose, positions_world).astype(np.float32)
        # Match the training convention: relative velocity is obstacle world velocity minus
        # robot world velocity, then rotated into robot frame. Before the first executed
        # transition, the robot translational velocity is explicitly zero.
        robot_vel_world = np.zeros((2,), dtype=float)
        vel_rel = robot_frame_points(
            np.array([0.0, 0.0, float(robot_pose[2])], dtype=float),
            velocities_world - robot_vel_world[None, :],
        ).astype(np.float32)
        waypoint = compute_local_subgoal(robot_pose, path_world, self.config.lookahead_distance)
        return {
            "pos_rel": pos_rel,
            "vel_rel": vel_rel,
            "radii": radii,
            "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
        }

    def _state_from_transition(
        self,
        *,
        robot_pose_prev: np.ndarray,
        robot_pose_curr: np.ndarray,
        current_snapshots: Sequence[dict[str, object]],
        future_snapshots: Sequence[dict[str, object]],
        path_world: np.ndarray,
    ) -> dict[str, np.ndarray]:
        positions_world = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in current_snapshots], dtype=float)
        positions_world_next = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in future_snapshots], dtype=float)
        radii = np.asarray([float(snapshot["radius"]) for snapshot in current_snapshots], dtype=np.float32)
        obstacle_vel_world = (positions_world_next - positions_world) / float(self.config.robot_config.dt)
        robot_vel_world = (np.asarray(robot_pose_curr[:2], dtype=float) - np.asarray(robot_pose_prev[:2], dtype=float)) / float(
            self.config.robot_config.dt
        )
        waypoint = compute_local_subgoal(robot_pose_prev, path_world, self.config.lookahead_distance)
        return {
            "pos_rel": robot_frame_points(robot_pose_prev, positions_world).astype(np.float32),
            "vel_rel": robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose_prev[2])], dtype=float),
                obstacle_vel_world - robot_vel_world[None, :],
            ).astype(np.float32),
            "radii": radii,
            "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
        }

    def _next_state_from_snapshots(
        self,
        *,
        robot_pose_t: np.ndarray,
        robot_pose_t1: np.ndarray,
        next_snapshots: Sequence[dict[str, object]],
        action_cont: np.ndarray,
        path_world: np.ndarray,
    ) -> dict[str, np.ndarray]:
        del action_cont
        positions_world = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in next_snapshots], dtype=float)
        obstacle_vel_world = np.asarray([np.asarray(snapshot["velocity"], dtype=float) for snapshot in next_snapshots], dtype=float)
        robot_vel_world = (np.asarray(robot_pose_t1[:2], dtype=float) - np.asarray(robot_pose_t[:2], dtype=float)) / float(
            self.config.robot_config.dt
        )
        radii = np.asarray([float(snapshot["radius"]) for snapshot in next_snapshots], dtype=np.float32)
        waypoint = compute_local_subgoal(robot_pose_t1, path_world, self.config.lookahead_distance)
        return {
            "pos_rel": robot_frame_points(robot_pose_t1, positions_world).astype(np.float32),
            "vel_rel": robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose_t1[2])], dtype=float),
                obstacle_vel_world - robot_vel_world[None, :],
            ).astype(np.float32),
            "radii": radii,
            "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
        }

    def _order_obstacle_objects(self, dynamic_obstacles: Sequence[DynamicObstacle]) -> list[DynamicObstacle]:
        ids = [str(obstacle.obstacle_id) for obstacle in dynamic_obstacles]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate dynamic obstacle ids detected: {ids}")
        if self.canonical_ids is None:
            self.canonical_ids = list(ids)
        missing = [obstacle_id for obstacle_id in self.canonical_ids if obstacle_id not in ids]
        extra = [obstacle_id for obstacle_id in ids if obstacle_id not in self.canonical_ids]
        if missing or extra:
            raise ValueError(f"Dynamic obstacle ids changed: missing={missing}, extra={extra}, canonical={self.canonical_ids}, seen={ids}")
        lookup = {str(obstacle.obstacle_id): obstacle for obstacle in dynamic_obstacles}
        return [lookup[obstacle_id] for obstacle_id in self.canonical_ids]

    def _order_snapshot_dicts(self, snapshots: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        ids = [str(snapshot["obstacle_id"]) for snapshot in snapshots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate dynamic obstacle ids detected: {ids}")
        if self.canonical_ids is None:
            self.canonical_ids = list(ids)
        missing = [obstacle_id for obstacle_id in self.canonical_ids if obstacle_id not in ids]
        extra = [obstacle_id for obstacle_id in ids if obstacle_id not in self.canonical_ids]
        if missing or extra:
            raise ValueError(f"Dynamic obstacle ids changed: missing={missing}, extra={extra}, canonical={self.canonical_ids}, seen={ids}")
        lookup = {str(snapshot["obstacle_id"]): snapshot for snapshot in snapshots}
        return [lookup[obstacle_id] for obstacle_id in self.canonical_ids]
