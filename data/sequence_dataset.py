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
        self.windows: list[SequenceWindow] = []
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

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        episode = _load_episode_cached(str(window.episode_path))
        steps = episode["steps"][window.start_step : window.start_step + window.sequence_length]
        if len(steps) != window.sequence_length:
            raise IndexError("Incomplete sequence window")

        obstacle_pos = np.stack(
            [np.asarray(step["dynamic_obstacles_robot_t"], dtype=np.float32) for step in steps], axis=0
        )
        obstacle_pos_t1 = np.stack(
            [np.asarray(step["dynamic_obstacles_robot_t1"], dtype=np.float32) for step in steps], axis=0
        )
        dt_value = float(steps[0].get("dt", episode.get("robot_config", {}).get("dt", 0.1)))
        dt_safe = max(dt_value, 1e-6)
        if all("dynamic_obstacle_delta_robot_t" in step for step in steps):
            delta_rel = np.stack(
                [np.asarray(step["dynamic_obstacle_delta_robot_t"], dtype=np.float32) for step in steps], axis=0
            )
        else:
            delta_rel = obstacle_pos_t1 - obstacle_pos
        if all("dynamic_obstacle_rel_vel_robot_fd_t" in step for step in steps):
            obstacle_vel = np.stack(
                [np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=np.float32) for step in steps], axis=0
            )
        else:
            obstacle_vel = delta_rel / dt_safe
        if all("dynamic_obstacle_rel_vel_robot_fd_t1" in step for step in steps):
            target_next_obstacle_vel = np.stack(
                [np.asarray(step["dynamic_obstacle_rel_vel_robot_fd_t1"], dtype=np.float32) for step in steps], axis=0
            )
        else:
            episode_steps = episode["steps"]
            target_next_velocity_rows: list[np.ndarray] = []
            for local_index, step in enumerate(steps):
                episode_step_index = window.start_step + local_index
                if episode_step_index + 1 < len(episode_steps):
                    next_step = episode_steps[episode_step_index + 1]
                    if "dynamic_obstacle_rel_vel_robot_fd_t" in next_step:
                        next_vel = np.asarray(next_step["dynamic_obstacle_rel_vel_robot_fd_t"], dtype=np.float32)
                    else:
                        next_pos = np.asarray(next_step["dynamic_obstacles_robot_t"], dtype=np.float32)
                        next_pos_t1 = np.asarray(next_step["dynamic_obstacles_robot_t1"], dtype=np.float32)
                        next_vel = (next_pos_t1 - next_pos) / dt_safe
                else:
                    next_vel = delta_rel[local_index] / dt_safe
                target_next_velocity_rows.append(next_vel)
            target_next_obstacle_vel = np.stack(target_next_velocity_rows, axis=0)
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
            "delta_rel": torch.from_numpy(delta_rel),
            "target_next_obstacle_vel": torch.from_numpy(target_next_obstacle_vel),
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
        if self.use_goal:
            goal = np.stack([np.asarray(step["goal_t"], dtype=np.float32) for step in steps], axis=0)
            sample["goal"] = torch.from_numpy(goal)
        if self.use_lidar:
            lidar = np.stack([np.asarray(step["lidar_t"], dtype=np.float32) for step in steps], axis=0)
            sample["lidar"] = torch.from_numpy(lidar)
        return sample
