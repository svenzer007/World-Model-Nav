"""Learned V4 controller using recurrent world-frame obstacle rollout for dynamic scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from policy_eval.robustness import RobotPoseEKF, covariance_summary, generate_rollout_noise_sequences
from controllers.baseline_structured_controller import (
    StructuredControllerConfig,
    apply_backward_policy,
    dynamic_clearances_from_positions,
    effective_runtime_actions,
    expert_style_rollout_cost,
    _is_backward_candidate,
    planning_point_collision,
    robot_frame_points,
    velocity_penalty_from_action,
)
from models import FACTOR_MODEL_TYPE, FACTOR_WORLD_MODEL_TYPE, StructuredDynamicsConfig, StructuredDynamicsModel, normalize_model_config
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
    """Roll out the structured dynamics model for dynamic scoring."""

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
        model_cfg = normalize_model_config(checkpoint.get("config", {}).get("model", {}))
        if bool(model_cfg.get("use_lidar", False)):
            raise ValueError("learned structured controller does not support checkpoints with use_lidar=true")
        self.model = StructuredDynamicsModel(StructuredDynamicsConfig(**model_cfg)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.use_goal = bool(model_cfg.get("use_goal", True))
        self.model_type = str(model_cfg.get("model_type"))
        self.h_real = None
        self.canonical_ids: list[str] | None = None
        self.current_state_estimate: dict[str, np.ndarray] | None = None
        self.ekf = RobotPoseEKF.from_pose(np.zeros((3,), dtype=float))
        self.last_ekf_debug: dict[str, object] = {
            "estimated_pose_mean": self.ekf.mean.copy().tolist(),
            "ekf_covariance": covariance_summary(self.ekf.covariance),
        }
        self.rollout_rng = np.random.default_rng()

    def reset(self, *, initial_pose: np.ndarray | None = None, seed: int | None = None) -> None:
        self.h_real = None
        self.canonical_ids = None
        self.current_state_estimate = None
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
                    dynamic_obstacles=ordered_obstacles,
                    current_state=current_state,
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
                    dynamic_obstacles=ordered_obstacles,
                    current_state=current_state,
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
        ordered_t = self._order_snapshot_dicts(dynamic_obstacles_t)
        ordered_t1 = self._order_snapshot_dicts(dynamic_obstacles_t1)
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
        robot_pose_est_t1 = self.current_pose_estimate()
        self.last_ekf_debug = {
            "estimated_pose_mean": robot_pose_est_t1.copy().tolist(),
            "ekf_covariance": covariance_summary(self.ekf.covariance),
            "predicted_pose_mean": np.asarray(prediction_debug["predicted_mean"], dtype=float).tolist(),
            "pose_observation": np.asarray(correction_debug["observation"], dtype=float).tolist(),
            "innovation": np.asarray(correction_debug["innovation"], dtype=float).tolist(),
            "commanded_control": np.asarray(action_cont, dtype=float).tolist(),
        }
        if self.model_type == FACTOR_WORLD_MODEL_TYPE and self.current_state_estimate is not None:
            state_t = {
                "pos_world": np.asarray(self.current_state_estimate["pos_world"], dtype=np.float32).copy(),
                "vel_world": np.asarray(self.current_state_estimate["vel_world"], dtype=np.float32).copy(),
                "radii": np.asarray(self.current_state_estimate["radii"], dtype=np.float32).copy(),
                "goal": np.asarray(self.current_state_estimate["goal"], dtype=np.float32).copy(),
            }
        else:
            state_t = self._state_from_transition(
                robot_pose_prev=np.asarray(robot_pose_est_t, dtype=float),
                robot_pose_curr=np.asarray(robot_pose_est_t1, dtype=float),
                current_snapshots=ordered_t,
                future_snapshots=ordered_t1,
                path_world=path_world,
            )
        action_cont_np = np.asarray(action_cont, dtype=np.float32)
        action_tensor = torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        action_cont_tensor = torch.from_numpy(action_cont_np[None]).to(self.device)
        obstacle_pos = torch.from_numpy(
            state_t["pos_world" if self.model_type == FACTOR_WORLD_MODEL_TYPE else "pos_rel"][None]
        ).to(self.device)
        obstacle_vel = torch.from_numpy(
            state_t["vel_world" if self.model_type == FACTOR_WORLD_MODEL_TYPE else "vel_rel"][None]
        ).to(self.device)
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
        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            self.current_state_estimate = self._next_world_state_from_snapshots(
                current_snapshots=ordered_t,
                next_snapshots=ordered_t1,
                path_world=path_world,
                robot_pose_t1=np.asarray(robot_pose_est_t1, dtype=float),
            )
        else:
            self.current_state_estimate = self._next_state_from_snapshots(
                next_snapshots=ordered_t1,
                current_vel_rel=state_t["vel_rel"],
                path_world=path_world,
                robot_pose_t=np.asarray(robot_pose_est_t, dtype=float),
                robot_pose_t1=np.asarray(robot_pose_est_t1, dtype=float),
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
        control_noise_sequence: np.ndarray | None = None,
    ) -> dict[str, Any]:
        action = ACTIONS[int(action_index)]
        pose = np.asarray(robot_pose, dtype=float).copy()
        use_world_state = self.model_type == FACTOR_WORLD_MODEL_TYPE
        pos_state_current = np.asarray(
            current_state["pos_world" if use_world_state else "pos_rel"],
            dtype=np.float32,
        ).copy()
        vel_state_current = np.asarray(
            current_state["vel_world" if use_world_state else "vel_rel"],
            dtype=np.float32,
        ).copy()
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
        true_relative_velocities: list[list[list[float]]] = []
        true_dynamic_clearances: list[float] = []
        true_dynamic_collisions: list[bool] = []
        drift_per_obstacle: list[list[float]] = []
        drift_mean: list[float] = []
        drift_max: list[float] = []
        infeasible_reasons: list[str] = []
        true_pos_rel_current = robot_frame_points(
            np.asarray(robot_pose, dtype=float),
            np.asarray(current_state["pos_world"], dtype=float) if use_world_state else np.asarray(
                [obstacle.position.copy() for obstacle in dynamic_obstacles],
                dtype=float,
            ),
        ).astype(np.float32)

        action_index_tensor = torch.tensor([int(action_index)], dtype=torch.long, device=self.device)
        action_cont_tensor = torch.tensor([[float(action["v"]), float(action["omega"])]], dtype=torch.float32, device=self.device)

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
            waypoint = compute_local_subgoal(pose, path_world, self.config.lookahead_distance)
            goal_features = np.asarray(waypoint["goal_features"], dtype=np.float32)
            obstacle_pos_tensor = torch.from_numpy(pos_state_current[None]).to(self.device)
            obstacle_vel_tensor = torch.from_numpy(vel_state_current[None]).to(self.device)
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
            pred_delta = outputs["pred_delta_rel"][0].detach().cpu().numpy().astype(np.float32)
            pos_state_next = pos_state_current + pred_delta
            vel_state_next = (pos_state_next - pos_state_current) / float(self.config.robot_config.dt)

            if use_world_state:
                pos_rel_next = robot_frame_points(pose, pos_state_next.astype(float)).astype(np.float32)
                vel_rel_next = robot_frame_points(
                    np.array([0.0, 0.0, float(pose[2])], dtype=float),
                    vel_state_next.astype(float),
                ).astype(np.float32)
            else:
                pos_rel_next = pos_state_next
                if "pred_next_rel_vel" in outputs:
                    vel_rel_next = outputs["pred_next_rel_vel"][0].detach().cpu().numpy().astype(np.float32)
                else:
                    vel_rel_next = vel_state_next

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
            # Log true robot-frame relative velocity using the same finite-difference
            # convention as the predicted rollout velocity so rollout error stays comparable.
            true_vel_rel_next = (true_robot.astype(np.float32) - true_pos_rel_current) / float(self.config.robot_config.dt)
            true_clearances = dynamic_clearances_from_positions(true_robot, radii.astype(float), self.config.robot_config.radius)
            true_positions_robot.append(true_robot.astype(float).tolist())
            true_relative_velocities.append(true_vel_rel_next.astype(float).tolist())
            true_dynamic_clearances.append(float(np.min(true_clearances)) if true_clearances.size else float("inf"))
            # Log the true per-step dynamic collision boundary so we can compare the
            # learned rollout's safety classification against the true rollout outcome.
            true_dynamic_collisions.append(bool(np.any(true_clearances <= 0.0)))

            drift = np.linalg.norm(pos_rel_next.astype(float) - true_robot.astype(float), axis=1)
            drift_per_obstacle.append(drift.astype(float).tolist())
            drift_mean.append(float(np.mean(drift)) if drift.size else 0.0)
            drift_max.append(float(np.max(drift)) if drift.size else 0.0)

            pos_state_current = pos_state_next
            vel_state_current = vel_state_next if use_world_state else vel_rel_next
            true_pos_rel_current = true_robot.astype(np.float32)

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
            "predicted_positions_robot": predicted_positions_robot,
            "predicted_relative_velocities": predicted_velocities_robot,
            "true_positions_robot": true_positions_robot,
            "true_relative_velocities": true_relative_velocities,
            "true_dynamic_clearances": true_dynamic_clearances,
            "true_dynamic_collisions": true_dynamic_collisions,
            "drift_per_obstacle": drift_per_obstacle,
            "drift_mean_per_step": drift_mean,
            "drift_max_per_step": drift_max,
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
        current_state: dict[str, np.ndarray],
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
                current_state=current_state,
                control_noise_sequence=np.asarray(rollout_noise_sequences[sample_index], dtype=float),
            )
            for sample_index in range(int(rollout_noise_sequences.shape[0]))
        ]
        sample_costs = [float(sample["total_cost"]) for sample in sample_payloads]
        collision_flags = [
            bool(any(reason in {"planning_collision", "static_collision", "dynamic_collision"} for reason in sample["infeasible_reasons"]))
            for sample in sample_payloads
        ]
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
            "min_static_clearance": float(min(float(sample["min_static_clearance"]) for sample in sample_payloads)),
            "min_dynamic_clearance": float(min(float(sample["min_dynamic_clearance"]) for sample in sample_payloads)),
            "min_combined_clearance": float(min(float(sample["min_combined_clearance"]) for sample in sample_payloads)),
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

    def _current_state_for_decision(
        self,
        *,
        robot_pose: np.ndarray,
        path_world: np.ndarray,
        dynamic_obstacles: Sequence[DynamicObstacle],
    ) -> dict[str, np.ndarray]:
        if self.current_state_estimate is not None:
            if self.model_type == FACTOR_WORLD_MODEL_TYPE:
                return {
                    "pos_world": np.asarray(self.current_state_estimate["pos_world"], dtype=np.float32).copy(),
                    "vel_world": np.asarray(self.current_state_estimate["vel_world"], dtype=np.float32).copy(),
                    "radii": np.asarray(self.current_state_estimate["radii"], dtype=np.float32).copy(),
                    "goal": np.asarray(self.current_state_estimate["goal"], dtype=np.float32).copy(),
                }
            return {
                "pos_rel": np.asarray(self.current_state_estimate["pos_rel"], dtype=np.float32).copy(),
                "vel_rel": np.asarray(self.current_state_estimate["vel_rel"], dtype=np.float32).copy(),
                "radii": np.asarray(self.current_state_estimate["radii"], dtype=np.float32).copy(),
                "goal": np.asarray(self.current_state_estimate["goal"], dtype=np.float32).copy(),
            }

        positions_world = np.asarray([obstacle.position.copy() for obstacle in dynamic_obstacles], dtype=float)
        velocities_world = np.asarray([np.asarray(obstacle.velocity, dtype=float) for obstacle in dynamic_obstacles], dtype=float)
        radii = np.asarray([float(obstacle.radius) for obstacle in dynamic_obstacles], dtype=np.float32)
        waypoint = compute_local_subgoal(robot_pose, path_world, self.config.lookahead_distance)
        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            return {
                "pos_world": positions_world.astype(np.float32),
                "vel_world": velocities_world.astype(np.float32),
                "radii": radii,
                "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
            }
        pos_rel = robot_frame_points(robot_pose, positions_world).astype(np.float32)
        if self.model_type == FACTOR_MODEL_TYPE:
            # Before the first executed transition there is no previous robot-relative
            # position sample, so bootstrap finite-difference velocity by assuming the
            # robot is instantaneously stationary and rotating obstacle world velocity
            # into the current robot frame.
            vel_rel = robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose[2])], dtype=float),
                velocities_world,
            ).astype(np.float32)
        else:
            vel_rel = robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose[2])], dtype=float),
                velocities_world,
            ).astype(np.float32)
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
        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            positions_world = np.asarray(
                [np.asarray(snapshot["position"], dtype=float) for snapshot in current_snapshots],
                dtype=float,
            )
            velocities_world = np.asarray(
                [np.asarray(snapshot["velocity"], dtype=float) for snapshot in current_snapshots],
                dtype=float,
            )
            radii = np.asarray([float(snapshot["radius"]) for snapshot in current_snapshots], dtype=np.float32)
            waypoint = compute_local_subgoal(robot_pose_prev, path_world, self.config.lookahead_distance)
            return {
                "pos_world": positions_world.astype(np.float32),
                "vel_world": velocities_world.astype(np.float32),
                "radii": radii,
                "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
            }
        positions_world = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in current_snapshots], dtype=float)
        positions_world_next = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in future_snapshots], dtype=float)
        pos_rel = robot_frame_points(robot_pose_prev, positions_world).astype(np.float32)
        radii = np.asarray([float(snapshot["radius"]) for snapshot in current_snapshots], dtype=np.float32)
        if self.model_type == FACTOR_MODEL_TYPE:
            pos_rel_next = robot_frame_points(robot_pose_curr, positions_world_next).astype(np.float32)
            vel_rel = (pos_rel_next - pos_rel) / float(self.config.robot_config.dt)
        else:
            obstacle_vel_world = (positions_world_next - positions_world) / float(self.config.robot_config.dt)
            robot_vel_world = (np.asarray(robot_pose_curr[:2], dtype=float) - np.asarray(robot_pose_prev[:2], dtype=float)) / float(
                self.config.robot_config.dt
            )
            vel_rel = robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose_prev[2])], dtype=float),
                obstacle_vel_world - robot_vel_world[None, :],
            ).astype(np.float32)
        waypoint = compute_local_subgoal(robot_pose_prev, path_world, self.config.lookahead_distance)
        return {
            "pos_rel": pos_rel,
            "vel_rel": vel_rel.astype(np.float32),
            "radii": radii,
            "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
        }

    def _next_state_from_snapshots(
        self,
        *,
        next_snapshots: Sequence[dict[str, object]],
        current_vel_rel: np.ndarray,
        path_world: np.ndarray,
        robot_pose_t: np.ndarray,
        robot_pose_t1: np.ndarray,
    ) -> dict[str, np.ndarray]:
        positions_world = np.asarray([np.asarray(snapshot["position"], dtype=float) for snapshot in next_snapshots], dtype=float)
        radii = np.asarray([float(snapshot["radius"]) for snapshot in next_snapshots], dtype=np.float32)
        waypoint = compute_local_subgoal(robot_pose_t1, path_world, self.config.lookahead_distance)
        if self.model_type == FACTOR_MODEL_TYPE:
            vel_rel = np.asarray(current_vel_rel, dtype=np.float32).copy()
        else:
            obstacle_vel_world = np.asarray([np.asarray(snapshot["velocity"], dtype=float) for snapshot in next_snapshots], dtype=float)
            robot_vel_world = (np.asarray(robot_pose_t1[:2], dtype=float) - np.asarray(robot_pose_t[:2], dtype=float)) / float(
                self.config.robot_config.dt
            )
            vel_rel = robot_frame_points(
                np.array([0.0, 0.0, float(robot_pose_t1[2])], dtype=float),
                obstacle_vel_world - robot_vel_world[None, :],
            ).astype(np.float32)
        return {
            "pos_rel": robot_frame_points(robot_pose_t1, positions_world).astype(np.float32),
            "vel_rel": vel_rel,
            "radii": radii,
            "goal": np.asarray(waypoint["goal_features"], dtype=np.float32),
        }

    def _next_world_state_from_snapshots(
        self,
        *,
        current_snapshots: Sequence[dict[str, object]],
        next_snapshots: Sequence[dict[str, object]],
        path_world: np.ndarray,
        robot_pose_t1: np.ndarray,
    ) -> dict[str, np.ndarray]:
        positions_world_prev = np.asarray(
            [np.asarray(snapshot["position"], dtype=float) for snapshot in current_snapshots],
            dtype=float,
        )
        positions_world_next = np.asarray(
            [np.asarray(snapshot["position"], dtype=float) for snapshot in next_snapshots],
            dtype=float,
        )
        radii = np.asarray([float(snapshot["radius"]) for snapshot in next_snapshots], dtype=np.float32)
        waypoint = compute_local_subgoal(robot_pose_t1, path_world, self.config.lookahead_distance)
        return {
            "pos_world": positions_world_next.astype(np.float32),
            "vel_world": (
                (positions_world_next - positions_world_prev) / float(self.config.robot_config.dt)
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
