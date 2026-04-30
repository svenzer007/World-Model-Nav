"""Runtime robustness helpers for policy-eval EKF and stochastic planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sim2d.dynamics import unicycle_step
from sim2d.utils import wrap_angle


def _wrap_pose_theta(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).copy()
    pose[2] = wrap_angle(float(pose[2]))
    return pose


def covariance_summary(covariance: np.ndarray) -> dict[str, object]:
    cov = np.asarray(covariance, dtype=float)
    diag = np.diag(cov) if cov.ndim == 2 else np.zeros((3,), dtype=float)
    return {
        "diag": [float(value) for value in diag[:3]],
        "trace": float(np.trace(cov)) if cov.ndim == 2 else 0.0,
    }


def sample_executed_control(
    rng: np.random.Generator,
    commanded_control: Sequence[float],
    *,
    execution_noise_enabled: bool,
    sigma_v: float,
    sigma_omega: float,
) -> np.ndarray:
    control = np.asarray(commanded_control, dtype=float)
    if not execution_noise_enabled:
        return control.copy()
    noise = np.array(
        [
            float(rng.normal(0.0, float(sigma_v))),
            float(rng.normal(0.0, float(sigma_omega))),
        ],
        dtype=float,
    )
    return control + noise


def sample_pose_observation(
    rng: np.random.Generator,
    true_pose: Sequence[float],
    *,
    pose_observation_noise_enabled: bool,
    sigma_obs_x: float,
    sigma_obs_y: float,
    sigma_obs_theta: float,
) -> np.ndarray:
    pose = np.asarray(true_pose, dtype=float).copy()
    if pose_observation_noise_enabled:
        pose[0] += float(rng.normal(0.0, float(sigma_obs_x)))
        pose[1] += float(rng.normal(0.0, float(sigma_obs_y)))
        pose[2] = wrap_angle(float(pose[2] + rng.normal(0.0, float(sigma_obs_theta))))
    return pose


def generate_rollout_noise_sequences(
    rng: np.random.Generator,
    *,
    num_stochastic_rollouts: int,
    horizon: int,
    execution_noise_enabled: bool,
    sigma_v: float,
    sigma_omega: float,
) -> np.ndarray:
    num_rollouts = max(1, int(num_stochastic_rollouts))
    horizon_steps = max(1, int(horizon))
    if not execution_noise_enabled:
        return np.zeros((num_rollouts, horizon_steps, 2), dtype=float)
    noise = np.zeros((num_rollouts, horizon_steps, 2), dtype=float)
    noise[:, :, 0] = rng.normal(0.0, float(sigma_v), size=(num_rollouts, horizon_steps))
    noise[:, :, 1] = rng.normal(0.0, float(sigma_omega), size=(num_rollouts, horizon_steps))
    return noise


@dataclass
class RobotPoseEKF:
    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_pose(cls, pose: Sequence[float]) -> "RobotPoseEKF":
        return cls(mean=_wrap_pose_theta(np.asarray(pose, dtype=float)), covariance=np.zeros((3, 3), dtype=float))

    def predict(
        self,
        *,
        commanded_control: Sequence[float],
        dt: float,
        execution_noise_enabled: bool,
        sigma_v: float,
        sigma_omega: float,
    ) -> dict[str, object]:
        control = np.asarray(commanded_control, dtype=float)
        theta = float(self.mean[2])
        v_cmd = float(control[0])
        dt_value = float(dt)
        state_jacobian = np.array(
            [
                [1.0, 0.0, -v_cmd * np.sin(theta) * dt_value],
                [0.0, 1.0, v_cmd * np.cos(theta) * dt_value],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        control_jacobian = np.array(
            [
                [np.cos(theta) * dt_value, 0.0],
                [np.sin(theta) * dt_value, 0.0],
                [0.0, dt_value],
            ],
            dtype=float,
        )
        if execution_noise_enabled:
            control_cov = np.diag([float(sigma_v) ** 2, float(sigma_omega) ** 2])
        else:
            control_cov = np.zeros((2, 2), dtype=float)
        process_cov = control_jacobian @ control_cov @ control_jacobian.T
        predicted_mean = unicycle_step(self.mean, float(control[0]), float(control[1]), dt_value)
        predicted_cov = state_jacobian @ self.covariance @ state_jacobian.T + process_cov
        self.mean = _wrap_pose_theta(predicted_mean)
        self.covariance = predicted_cov
        return {
            "state_jacobian": state_jacobian,
            "process_covariance": process_cov,
            "predicted_mean": self.mean.copy(),
            "predicted_covariance": self.covariance.copy(),
        }

    def correct(
        self,
        *,
        pose_observation: Sequence[float],
        pose_observation_noise_enabled: bool,
        sigma_obs_x: float,
        sigma_obs_y: float,
        sigma_obs_theta: float,
    ) -> dict[str, object]:
        observation = _wrap_pose_theta(np.asarray(pose_observation, dtype=float))
        if pose_observation_noise_enabled:
            observation_cov = np.diag(
                [float(sigma_obs_x) ** 2, float(sigma_obs_y) ** 2, float(sigma_obs_theta) ** 2]
            )
        else:
            observation_cov = np.diag([1e-9, 1e-9, 1e-9])
        innovation = observation - self.mean
        innovation[2] = wrap_angle(float(innovation[2]))
        innovation_cov = self.covariance + observation_cov
        kalman_gain = self.covariance @ np.linalg.inv(innovation_cov)
        updated_mean = self.mean + kalman_gain @ innovation
        identity = np.eye(3, dtype=float)
        updated_cov = (identity - kalman_gain) @ self.covariance
        self.mean = _wrap_pose_theta(updated_mean)
        self.covariance = updated_cov
        return {
            "observation": observation.copy(),
            "observation_covariance": observation_cov,
            "innovation": innovation,
            "innovation_covariance": innovation_cov,
            "kalman_gain": kalman_gain,
            "corrected_mean": self.mean.copy(),
            "corrected_covariance": self.covariance.copy(),
        }
