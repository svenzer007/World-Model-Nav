# Deploying the Learned Structured Policy

This repository trains a structured world model, but deployment uses that model through
`LearnedStructuredController`. The controller is the policy interface: it evaluates the
fixed action set, uses the learned model to predict dynamic-obstacle rollout and risk,
scores each candidate action, and returns the robot command to apply.

For closed-loop simulation evaluation and animations, see
[`README_v4_policy_comparison.md`](./README_v4_policy_comparison.md). For offline
checkpoint metrics on saved data, see [`README_v4_evaluation.md`](./README_v4_evaluation.md).

## What You Need Once

Use a full training checkpoint:

- Good: `best.pt`, `latest.pt`, `epoch_017.pt`
- Not enough: `*_weights.pt`

Instantiate the policy once per run:

```python
from pathlib import Path

from controllers.baseline_structured_controller import StructuredControllerConfig
from controllers.learned_structured_controller import LearnedStructuredController
from sim2d.config import DatasetConfig

checkpoint_path = Path("artifacts/structured_world_model_rollout_from_epoch_10/best.pt")

config = DatasetConfig(max_steps=200)
controller_cfg = StructuredControllerConfig(horizon=10)

policy = LearnedStructuredController(
    checkpoint_path=checkpoint_path,
    config=config,
    controller_cfg=controller_cfg,
    device="auto",
)
```

Call `policy.reset()` at the start of each episode, rollout, or deployment run. This
clears the controller's recurrent hidden state and obstacle ID tracking.

## Per-Timestep Inputs

At each timestep, call `policy.select_action(...)` with the current world state adapted
to the controller interface.

Required inputs:

- `robot_pose`: `np.ndarray` with shape `(3,)`, containing `[x, y, theta]` in world frame.
- `current_subgoal_world`: `np.ndarray` with shape `(2,)`, usually from `compute_local_subgoal(...)`.
- `path_world`: `np.ndarray` with shape `(N, 2)`, the reference or A* path in world coordinates.
- `planning_occupancy`: 2D occupancy grid used for planning feasibility checks, usually inflated.
- `true_occupancy`: 2D occupancy grid used for actual static collision checks.
- `resolution`: map resolution in meters per grid cell.
- `origin`: map origin as `(x, y)`.
- `occupied_points`: `np.ndarray` with shape `(M, 2)`, world-frame static occupied points for clearance checks.
- `dynamic_obstacles`: dynamic obstacle objects with stable IDs.

Each dynamic obstacle must provide the same behavior as `sim2d.obstacles.DynamicObstacle`:

- `obstacle_id`: stable string ID. IDs must not change within an episode.
- `position`: current world-frame `np.ndarray` with shape `(2,)`.
- `radius`: obstacle radius in meters.
- `velocity`: current world-frame velocity vector.
- `clone()`: returns an independent copy.
- `step(dt)`: advances the obstacle copy during candidate rollouts.
- `snapshot()`: returns a dict with at least `obstacle_id`, `position`, `velocity`, and `radius`.

The number of dynamic obstacles must match the checkpoint's model configuration. The
current default model expects 4 dynamic obstacles.

The learned deployment controller does not support checkpoints trained with
`use_lidar=true`; it passes `lidar=None`.

## Policy Outputs

`policy.select_action(...)` returns a decision dictionary:

```python
{
    "action_index": int,
    "action_name": str,
    "v": float,
    "omega": float,
    "debug": dict,
    "chosen_action_min_clearance": float,
    "emergency_override_used": bool,
}
```

For deployment, apply:

- `v`: linear velocity command.
- `omega`: angular velocity command.

The fixed action set is:

| Index | Name | `v` | `omega` |
| ---: | --- | ---: | ---: |
| 0 | `stop` | `0.0` | `0.0` |
| 1 | `forward` | `0.4` | `0.0` |
| 2 | `forward_left` | `0.4` | `1.0` |
| 3 | `forward_right` | `0.4` | `-1.0` |
| 4 | `slow_left` | `0.15` | `1.0` |
| 5 | `slow_right` | `0.15` | `-1.0` |

The `debug` payload is useful for logging. It includes the chosen candidate, all
candidate scores, predicted obstacle rollout, clearance estimates, collision flags, and
model risk outputs. Your runtime should not need `debug` to apply the command.

After applying the command and observing the next state, call `policy.commit_step(...)`.
This updates the GRU hidden state and stores the latest true obstacle state for the next
decision. Skipping this call makes the controller behave as if it has no recurrent
history.

## Python Call Template

This template shows the control loop shape. Replace the `get_*`, `apply_control`, and
termination functions with your simulator or robot integration.

```python
from pathlib import Path

import numpy as np

from controllers.baseline_structured_controller import StructuredControllerConfig
from controllers.learned_structured_controller import LearnedStructuredController
from sim2d.config import DatasetConfig
from sim2d.waypoint import compute_local_subgoal


def snapshot_obstacles(dynamic_obstacles):
    return [obstacle.snapshot() for obstacle in dynamic_obstacles]


checkpoint_path = Path("artifacts/structured_world_model_rollout_from_epoch_10/best.pt")

config = DatasetConfig(max_steps=200)
controller_cfg = StructuredControllerConfig(horizon=10)

policy = LearnedStructuredController(
    checkpoint_path=checkpoint_path,
    config=config,
    controller_cfg=controller_cfg,
    device="auto",
)

policy.reset()

for timestep in range(config.max_steps):
    # Fill these from your simulator, middleware, or runtime state estimator.
    robot_pose_t = get_robot_pose()                  # np.ndarray shape (3,)
    path_world = get_reference_path()                # np.ndarray shape (N, 2)
    planning_occupancy = get_planning_occupancy()    # np.ndarray shape (H, W)
    true_occupancy = get_true_occupancy()            # np.ndarray shape (H, W)
    occupied_points = get_occupied_points_world()    # np.ndarray shape (M, 2)
    dynamic_obstacles = get_dynamic_obstacles()      # stable obstacle objects
    resolution = get_map_resolution()                # meters per cell
    origin = get_map_origin()                        # tuple/list/array (x, y)

    waypoint = compute_local_subgoal(
        robot_pose_t,
        path_world,
        config.lookahead_distance,
    )

    decision = policy.select_action(
        robot_pose=robot_pose_t,
        current_subgoal_world=waypoint["selected_subgoal_world"],
        path_world=path_world,
        planning_occupancy=planning_occupancy,
        true_occupancy=true_occupancy,
        resolution=resolution,
        origin=origin,
        occupied_points=occupied_points,
        dynamic_obstacles=dynamic_obstacles,
    )

    dynamic_t = snapshot_obstacles(dynamic_obstacles)

    v = float(decision["v"])
    omega = float(decision["omega"])
    apply_control(v, omega)

    # After the simulator or robot advances, collect the next state.
    robot_pose_t1 = get_robot_pose()
    dynamic_obstacles_t1 = get_dynamic_obstacles()
    dynamic_t1 = snapshot_obstacles(dynamic_obstacles_t1)

    policy.commit_step(
        robot_pose_t=robot_pose_t,
        robot_pose_t1=robot_pose_t1,
        dynamic_obstacles_t=dynamic_t,
        dynamic_obstacles_t1=dynamic_t1,
        action_index=int(decision["action_index"]),
        action_cont=np.array([v, omega], dtype=np.float32),
        path_world=path_world,
    )

    log_policy_decision(timestep, decision)

    if reached_goal_or_collision():
        break
```

## Minimal Deployment Checklist

Before using the policy in another simulator or runtime, make sure you can provide:

- A full checkpoint whose model config matches this code.
- Robot pose in the same world coordinate frame as the map and path.
- A valid path with enough points for local subgoal selection.
- Static map grids plus `occupied_points` in world coordinates.
- Dynamic obstacles with stable IDs, current position, radius, velocity, `clone()`, and `step(dt)`.
- A loop that calls `select_action(...)`, applies `v` and `omega`, observes the next state, then calls `commit_step(...)`.

## Existing Evaluation Runner

To see the expected interface in action, inspect `run_policy_comparison_v4.py`. It
samples simulator scenarios, calls `select_action(...)`, applies the returned `v` and
`omega`, calls `commit_step(...)`, and can save GIF/MP4 animations:

```bash
python3 run_policy_comparison_v4.py \
  --checkpoint artifacts/structured_world_model_rollout_from_epoch_10/best.pt \
  --output_dir artifacts/policy_comparison_v4_eval \
  --num_episodes 5 \
  --max_steps 200 \
  --horizon 10 \
  --save_previews \
  --save_gifs
```

Animations are written under:

```text
artifacts/policy_comparison_v4_eval/learned_structured/gif/
```
