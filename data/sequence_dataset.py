"""Fixed-length V4 structured obstacle-state sequence dataset."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from models import FACTOR_WORLD_MODEL_TYPE, LEGACY_MODEL_TYPE


@dataclass(frozen=True)
class SequenceWindow:
    episode_id: int
    episode_path: Path
    start_step: int
    sequence_length: int
    stop_fraction: float
    dynamic_event_fraction: float
    max_risk: float
    min_clearance: float
    has_dynamic_collision_next: bool
    has_dynamic_collision_within_3: bool
    has_dynamic_collision_within_5: bool


def list_episode_paths(data_dir: str | Path) -> list[Path]:
    """Return sorted episode pickle paths under a dataset root or episodes directory."""
    root = Path(data_dir)
    episodes_dir = root if root.name == "episodes" else root / "episodes"
    paths = sorted(episodes_dir.glob("episode_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No episode pickles found under {episodes_dir}")
    return paths


def split_episode_paths(paths: Sequence[str | Path], train_fraction: float = 0.9) -> tuple[list[Path], list[Path]]:
    """Split episode paths deterministically by sorted filename order."""
    sorted_paths = sorted(Path(path) for path in paths)
    if len(sorted_paths) < 2:
        return list(sorted_paths), []
    split_index = int(len(sorted_paths) * float(train_fraction))
    split_index = max(1, min(split_index, len(sorted_paths) - 1))
    return list(sorted_paths[:split_index]), list(sorted_paths[split_index:])


@lru_cache(maxsize=64)
def _load_episode_cached(path_str: str) -> dict[str, object]:
    with Path(path_str).open("rb") as handle:
        return pickle.load(handle)


class StructuredSequenceDataset(Dataset[dict[str, object]]):
    """Contiguous temporal windows from V4 episode pickle files."""

    def __init__(
        self,
        episode_paths: Sequence[str | Path],
        sequence_length: int = 20,
        stride: int = 1,
        use_goal: bool = True,
        use_lidar: bool = False,
        rollout_horizon: int = 0,
        model_type: str = LEGACY_MODEL_TYPE,
        cache_episodes: bool = False,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.episode_paths = [Path(path) for path in episode_paths]
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self.use_goal = bool(use_goal)
        self.use_lidar = bool(use_lidar)
        self.rollout_horizon = max(0, int(rollout_horizon))
        self.model_type = str(model_type)
        self.cache_episodes = bool(cache_episodes)
        self.windows: list[SequenceWindow] = []
        self._episode_arrays: dict[str, dict[str, object]] = {}
        self._build_index()

    def _build_index(self) -> None:
        for path in self.episode_paths:
            episode = _load_episode_cached(str(path))
            steps = episode.get("steps", [])
            if not isinstance(steps, list):
                raise TypeError(f"Episode {path} has non-list steps")
            if len(steps) < self.sequence_length:
                continue
            episode_id = int(episode["episode_id"])
            last_start = len(steps) - self.sequence_length
            for start_step in range(0, last_start + 1, self.stride):
                window_steps = steps[start_step : start_step + self.sequence_length]
                stop_fraction = float(np.mean([step.get("action_name") == "stop" for step in window_steps]))
                event_fraction = float(np.mean([bool(step.get("dynamic_event_step_t", False)) for step in window_steps]))
                risks = np.asarray([float(step["dynamic_risk_t1"]) for step in window_steps], dtype=float)
                clearances = np.asarray([float(step["min_dynamic_clearance_t"]) for step in window_steps], dtype=float)
                self.windows.append(
                    SequenceWindow(
                        episode_id=episode_id,
                        episode_path=path,
                        start_step=start_step,
                        sequence_length=self.sequence_length,
                        stop_fraction=stop_fraction,
                        dynamic_event_fraction=event_fraction,
                        max_risk=float(np.max(risks)) if risks.size else 0.0,
                        min_clearance=float(np.min(clearances)) if clearances.size else float("inf"),
                        has_dynamic_collision_next=any(bool(step["dynamic_collision_t1"]) for step in window_steps),
                        has_dynamic_collision_within_3=any(
                            bool(step["dynamic_collision_within_3"]) for step in window_steps
                        ),
                        has_dynamic_collision_within_5=any(
                            bool(step["dynamic_collision_within_5"]) for step in window_steps
                        ),
                    )
                )
            if self.cache_episodes:
                self._episode_arrays[str(path)] = self._build_episode_arrays(episode)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        if self.cache_episodes:
            cached = self._episode_arrays.get(str(window.episode_path))
            if cached is not None:
                return self._cached_item(window, cached)

        episode = _load_episode_cached(str(window.episode_path))
        steps = episode["steps"][window.start_step : window.start_step + window.sequence_length]
        if len(steps) != window.sequence_length:
            raise IndexError("Incomplete sequence window")

        dt_value = float(steps[0].get("dt", episode.get("robot_config", {}).get("dt", 0.1)))
        dt_safe = max(dt_value, 1e-9)
        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            obstacle_pos = np.stack(
                [np.asarray(step["dynamic_obstacles_world_t"], dtype=np.float32) for step in steps],
                axis=0,
            )
            delta_world = np.stack(
                [self._delta_world_from_step(step) for step in steps],
                axis=0,
            )
            obstacle_vel = np.stack(
                [
                    self._causal_world_velocity_from_step(
                        episode=episode,
                        absolute_step_index=window.start_step + offset,
                        step=step,
                        dt=dt_safe,
                    )
                    for offset, step in enumerate(steps)
                ],
                axis=0,
            )
            target_next_obstacle_vel = None
            delta_rel = None
        else:
            obstacle_pos = np.stack(
                [np.asarray(step["dynamic_obstacles_robot_t"], dtype=np.float32) for step in steps], axis=0
            )
            delta_rel = np.stack(
                [self._delta_rel_from_step(step) for step in steps],
                axis=0,
            )
            obstacle_vel = np.stack(
                [self._fd_velocity_t_from_step(step, dt_safe) for step in steps],
                axis=0,
            )
            target_next_obstacle_vel = np.stack(
                [
                    self._fd_velocity_t1_from_step(
                        episode=episode,
                        absolute_step_index=window.start_step + offset,
                        step=step,
                        dt=dt_safe,
                    )
                    for offset, step in enumerate(steps)
                ],
                axis=0,
            )
            delta_world = None
        radii = np.asarray(steps[0]["dynamic_obstacle_radii"], dtype=np.float32)

        action_index = np.asarray([int(step["action_index"]) for step in steps], dtype=np.int64)
        action_cont = np.asarray([[float(step["v_t"]), float(step["omega_t"])] for step in steps], dtype=np.float32)
        risk = np.asarray([float(step["dynamic_risk_t1"]) for step in steps], dtype=np.float32)
        collision_next = np.asarray([float(step["dynamic_collision_t1"]) for step in steps], dtype=np.float32)
        collision_within_3 = np.asarray([float(step["dynamic_collision_within_3"]) for step in steps], dtype=np.float32)
        collision_within_5 = np.asarray([float(step["dynamic_collision_within_5"]) for step in steps], dtype=np.float32)
        valid_mask = np.ones((window.sequence_length,), dtype=np.float32)

        sample: dict[str, object] = {
            "obstacle_pos": torch.from_numpy(obstacle_pos),
            "obstacle_vel": torch.from_numpy(obstacle_vel),
            "radii": torch.from_numpy(radii),
            "action_index": torch.from_numpy(action_index),
            "action_cont": torch.from_numpy(action_cont),
            "risk": torch.from_numpy(risk),
            "collision_next": torch.from_numpy(collision_next),
            "collision_within_3": torch.from_numpy(collision_within_3),
            "collision_within_5": torch.from_numpy(collision_within_5),
            "valid_mask": torch.from_numpy(valid_mask),
            "dt": torch.tensor(dt_value, dtype=torch.float32),
            "episode_id": torch.tensor(window.episode_id, dtype=torch.long),
            "start_step": torch.tensor(window.start_step, dtype=torch.long),
            "source_file": str(window.episode_path),
        }
        if delta_rel is not None:
            sample["delta_rel"] = torch.from_numpy(delta_rel)
        if target_next_obstacle_vel is not None:
            sample["target_next_obstacle_vel"] = torch.from_numpy(target_next_obstacle_vel)
        if delta_world is not None:
            sample["delta_world"] = torch.from_numpy(delta_world)
        if self.use_goal:
            goal = np.stack([np.asarray(step["goal_t"], dtype=np.float32) for step in steps], axis=0)
            sample["goal"] = torch.from_numpy(goal)
        if self.use_lidar:
            lidar = np.stack([np.asarray(step["lidar_t"], dtype=np.float32) for step in steps], axis=0)
            sample["lidar"] = torch.from_numpy(lidar)
        return sample

    def _build_episode_arrays(self, episode: dict[str, object]) -> dict[str, object]:
        steps = episode.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Episode {episode.get('episode_id')} has no steps to cache")

        dt_values = np.asarray(
            [float(step.get("dt", episode.get("robot_config", {}).get("dt", 0.1))) for step in steps],
            dtype=np.float32,
        )
        action_index = np.asarray([int(step["action_index"]) for step in steps], dtype=np.int64)
        action_cont = np.asarray([[float(step["v_t"]), float(step["omega_t"])] for step in steps], dtype=np.float32)
        arrays: dict[str, object] = {
            "dt": dt_values,
            "radii": np.asarray(steps[0]["dynamic_obstacle_radii"], dtype=np.float32),
            "action_index": action_index,
            "action_cont": action_cont,
            "risk": np.asarray([float(step["dynamic_risk_t1"]) for step in steps], dtype=np.float32),
            "collision_next": np.asarray([float(step["dynamic_collision_t1"]) for step in steps], dtype=np.float32),
            "collision_within_3": np.asarray(
                [float(step["dynamic_collision_within_3"]) for step in steps], dtype=np.float32
            ),
            "collision_within_5": np.asarray(
                [float(step["dynamic_collision_within_5"]) for step in steps], dtype=np.float32
            ),
        }

        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            obstacle_pos = np.stack(
                [np.asarray(step["dynamic_obstacles_world_t"], dtype=np.float32) for step in steps],
                axis=0,
            )
            arrays["obstacle_pos"] = obstacle_pos
            arrays["delta_world"] = np.stack([self._delta_world_from_step(step) for step in steps], axis=0)
            arrays["obstacle_vel"] = self._cached_causal_world_velocity(episode, steps, obstacle_pos, dt_values)
        else:
            obstacle_pos = np.stack(
                [np.asarray(step["dynamic_obstacles_robot_t"], dtype=np.float32) for step in steps],
                axis=0,
            )
            arrays["obstacle_pos"] = obstacle_pos
            arrays["delta_rel"] = np.stack([self._delta_rel_from_step(step) for step in steps], axis=0)
            obstacle_vel = np.stack(
                [
                    self._fd_velocity_t_from_step(step, max(float(dt_values[step_index]), 1e-9))
                    for step_index, step in enumerate(steps)
                ],
                axis=0,
            )
            arrays["obstacle_vel"] = obstacle_vel
            if all("dynamic_obstacle_rel_vel_robot_fd_t1" in step for step in steps):
                arrays["target_next_obstacle_vel"] = np.stack(
                    [np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t1"], dtype=np.float32) for step in steps],
                    axis=0,
                )
            else:
                target_next = np.empty_like(obstacle_vel)
                target_next[:-1] = obstacle_vel[1:]
                target_next[-1] = obstacle_vel[-1]
                arrays["target_next_obstacle_vel"] = target_next

        if self.use_goal:
            arrays["goal"] = np.stack([np.asarray(step["goal_t"], dtype=np.float32) for step in steps], axis=0)
        if self.use_lidar:
            arrays["lidar"] = np.stack([np.asarray(step["lidar_t"], dtype=np.float32) for step in steps], axis=0)
        return arrays

    def _cached_item(self, window: SequenceWindow, arrays: dict[str, object]) -> dict[str, object]:
        start = window.start_step
        stop = start + window.sequence_length
        valid_mask = np.ones((window.sequence_length,), dtype=np.float32)
        sample: dict[str, object] = {
            "obstacle_pos": torch.from_numpy(arrays["obstacle_pos"][start:stop]),
            "obstacle_vel": torch.from_numpy(arrays["obstacle_vel"][start:stop]),
            "radii": torch.from_numpy(arrays["radii"]),
            "action_index": torch.from_numpy(arrays["action_index"][start:stop]),
            "action_cont": torch.from_numpy(arrays["action_cont"][start:stop]),
            "risk": torch.from_numpy(arrays["risk"][start:stop]),
            "collision_next": torch.from_numpy(arrays["collision_next"][start:stop]),
            "collision_within_3": torch.from_numpy(arrays["collision_within_3"][start:stop]),
            "collision_within_5": torch.from_numpy(arrays["collision_within_5"][start:stop]),
            "valid_mask": torch.from_numpy(valid_mask),
            "dt": torch.tensor(float(arrays["dt"][start]), dtype=torch.float32),
            "episode_id": torch.tensor(window.episode_id, dtype=torch.long),
            "start_step": torch.tensor(window.start_step, dtype=torch.long),
            "source_file": str(window.episode_path),
        }
        if self.model_type == FACTOR_WORLD_MODEL_TYPE:
            sample["delta_world"] = torch.from_numpy(arrays["delta_world"][start:stop])
        else:
            sample["delta_rel"] = torch.from_numpy(arrays["delta_rel"][start:stop])
            sample["target_next_obstacle_vel"] = torch.from_numpy(arrays["target_next_obstacle_vel"][start:stop])
        if self.use_goal:
            sample["goal"] = torch.from_numpy(arrays["goal"][start:stop])
        if self.use_lidar:
            sample["lidar"] = torch.from_numpy(arrays["lidar"][start:stop])
        return sample

    @staticmethod
    def _cached_causal_world_velocity(
        episode: dict[str, object],
        steps: list[dict[str, object]],
        obstacle_pos: np.ndarray,
        dt_values: np.ndarray,
    ) -> np.ndarray:
        if all("dynamic_obstacle_vel_world_causal_t" in step for step in steps):
            return np.stack(
                [np.asarray(step["dynamic_obstacle_vel_world_causal_t"], dtype=np.float32) for step in steps],
                axis=0,
            )
        obstacle_vel = np.empty_like(obstacle_pos)
        snapshots = list(steps[0].get("dynamic_obstacles_t", []))
        if not snapshots:
            raise ValueError(f"Cannot synthesize causal world velocity for episode {episode.get('episode_id')} at t=0")
        obstacle_vel[0] = np.asarray(
            [np.asarray(snapshot["velocity"], dtype=np.float32) for snapshot in snapshots],
            dtype=np.float32,
        )
        for step_index in range(1, len(steps)):
            dt_safe = max(float(dt_values[step_index]), 1e-9)
            obstacle_vel[step_index] = (obstacle_pos[step_index] - obstacle_pos[step_index - 1]) / dt_safe
        return obstacle_vel

    @staticmethod
    def _delta_rel_from_step(step: dict[str, object]) -> np.ndarray:
        if "dynamic_obstacle_delta_robot_t" in step:
            return np.asarray(step["dynamic_obstacle_delta_robot_t"], dtype=np.float32)
        pos_t = np.asarray(step["dynamic_obstacles_robot_t"], dtype=np.float32)
        pos_t1 = np.asarray(step["dynamic_obstacles_robot_t1"], dtype=np.float32)
        return pos_t1 - pos_t

    @classmethod
    def _fd_velocity_t_from_step(cls, step: dict[str, object], dt: float) -> np.ndarray:
        if "dynamic_obstacle_rel_vel_robot_fd_t" in step:
            return np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=np.float32)
        return cls._delta_rel_from_step(step) / dt

    @classmethod
    def _fd_velocity_t1_from_step(
        cls,
        *,
        episode: dict[str, object],
        absolute_step_index: int,
        step: dict[str, object],
        dt: float,
    ) -> np.ndarray:
        if "dynamic_obstacle_rel_vel_robot_fd_t1" in step:
            return np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t1"], dtype=np.float32)
        next_index = absolute_step_index + 1
        episode_steps = episode.get("steps", [])
        if next_index < len(episode_steps):
            return cls._fd_velocity_t_from_step(episode_steps[next_index], dt)
        return cls._fd_velocity_t_from_step(step, dt)

    @staticmethod
    def _delta_world_from_step(step: dict[str, object]) -> np.ndarray:
        if "dynamic_obstacle_delta_world_t" in step:
            return np.asarray(step["dynamic_obstacle_delta_world_t"], dtype=np.float32)
        pos_t = np.asarray(step["dynamic_obstacles_world_t"], dtype=np.float32)
        pos_t1 = np.asarray(step["dynamic_obstacles_world_t1"], dtype=np.float32)
        return pos_t1 - pos_t

    @classmethod
    def _causal_world_velocity_from_step(
        cls,
        *,
        episode: dict[str, object],
        absolute_step_index: int,
        step: dict[str, object],
        dt: float,
    ) -> np.ndarray:
        if "dynamic_obstacle_vel_world_causal_t" in step:
            return np.asarray(step["dynamic_obstacle_vel_world_causal_t"], dtype=np.float32)
        if absolute_step_index == 0:
            snapshots = list(step.get("dynamic_obstacles_t", []))
            if not snapshots:
                raise ValueError("Cannot synthesize causal world velocity for t=0 without dynamic_obstacles_t")
            return np.asarray([np.asarray(snapshot["velocity"], dtype=np.float32) for snapshot in snapshots], dtype=np.float32)
        episode_steps = episode.get("steps", [])
        previous_step = episode_steps[absolute_step_index - 1]
        current_world = np.asarray(step["dynamic_obstacles_world_t"], dtype=np.float32)
        previous_world = np.asarray(previous_step["dynamic_obstacles_world_t"], dtype=np.float32)
        return (current_world - previous_world) / dt
