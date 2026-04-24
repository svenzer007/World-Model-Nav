# V4 Structured Dynamic-Obstacle Dataset

V4 generates data for structured dynamic-obstacle prediction and dynamic-risk supervision. The primary learning inputs are fixed-ID obstacle states, especially robot-relative obstacle positions and robot-relative velocities. LiDAR is saved only as auxiliary front-sector context and should not be treated as the main dynamic-obstacle representation.

## Generate

```bash
cd /home/stavan/s4/rnl/4proj
python -m sim2d.generate_data \
  --num_episodes 20 \
  --max_steps 250 \
  --output_dir ./output_v4_dynamic_state \
  --seed 0
```

Smoke test:

```bash
cd /home/stavan/s4/rnl/4proj
python -m sim2d.generate_data \
  --num_episodes 2 \
  --max_steps 40 \
  --output_dir /tmp/rnl_v4_smoke \
  --seed 0
```

Inspect:

```bash
cd /home/stavan/s4/rnl/4proj
python -m sim2d.inspect_episodes \
  /tmp/rnl_v4_smoke \
  --report \
  --report_path /tmp/rnl_v4_smoke/dataset_quality_report.json
```

## Defaults

- Robot: radius `0.19m`, `dt=0.1s`, goal tolerance `0.4m`.
- LiDAR: auxiliary front sector, `241` beams, `[-pi/2, pi/2]`, max range `6.0m`, step size `0.05m`.
- Dynamic obstacles: fixed `N_dyn=4`, radii sampled from `(0.17, 0.26)m`, speeds from `(0.18, 0.30)m/s`, route length from `2` to `4` waypoints.
- Dynamic labels: `dynamic_risk_tau=0.5`, `dynamic_risk_positive_threshold=0.35`, `dynamic_visibility_radius=1.2`.
- Acceptance: `min_dynamic_visible_fraction=0.06`, `min_dynamic_risk_fraction=0.04`, `min_dynamic_event_steps=3`, `dynamic_interaction_resample_limit=30`.

## Per-Step Schema

Robot/action/task:

```text
robot_pose_t, robot_pose_t1
goal_t, goal_t1
action_index, action_name, v_t, omega_t
```

Primary structured obstacle fields:

```text
dynamic_obstacle_ids
dynamic_obstacles_world_t       shape (N_dyn, 2)
dynamic_obstacles_world_t1      shape (N_dyn, 2)
dynamic_obstacle_delta_world_t  shape (N_dyn, 2)
dynamic_obstacles_robot_t       shape (N_dyn, 2)
dynamic_obstacles_robot_t1      shape (N_dyn, 2)
dynamic_obstacle_radii          shape (N_dyn,)
dynamic_obstacle_vel_world_t    shape (N_dyn, 2)
dynamic_obstacle_vel_robot_t    shape (N_dyn, 2)
dynamic_obstacle_rel_vel_robot_t shape (N_dyn, 2)
```

Auxiliary context:

```text
lidar_t, lidar_t1
lidar_static_t, lidar_static_t1
lidar_dynamic_residual_t, lidar_dynamic_residual_t1
```

Dynamic hazard targets:

```text
dynamic_visible_t
dynamic_collision_t1
dynamic_collision_within_3
dynamic_collision_within_5
min_dynamic_clearance_t
min_dynamic_clearance_t1
dynamic_risk_t1
dynamic_event_step_t
```

## Identity and Motion

Obstacle identity is fixed within each episode. Obstacles are sampled once, assigned stable IDs `dyn_00`, `dyn_01`, ..., and every `(N_dyn, 2)` array is assembled in that same list order. The inspector verifies that IDs remain unchanged and that `dynamic_obstacles_world_t1` at step `i` matches `dynamic_obstacles_world_t` at step `i+1`.

Dynamic motion is smooth waypoint patrol motion: each obstacle moves at a sampled constant speed toward its current route target, then advances to the next waypoint. There is no frame-to-frame random walk, so deterministic short-horizon prediction over about seven steps remains meaningful.

## Frames, Deltas, and Velocities

Robot-frame obstacle position is:

```text
p_robot = R(-theta_robot) @ (p_world - robot_xy)
```

The rollout target delta is:

```text
dynamic_obstacle_delta_world_t = dynamic_obstacles_world_t1 - dynamic_obstacles_world_t
```

Velocity fields are finite-difference fields:

```text
dynamic_obstacle_vel_world_t = dynamic_obstacle_delta_world_t / dt
dynamic_obstacle_vel_robot_t = R(-theta_t) @ dynamic_obstacle_vel_world_t
robot_vel_world_t = (robot_pose_t1[:2] - robot_pose_t[:2]) / dt
dynamic_obstacle_rel_vel_robot_t = R(-theta_t) @ (dynamic_obstacle_vel_world_t - robot_vel_world_t)
```

`dynamic_obstacle_vel_robot_t` is world velocity expressed in robot axes. It is not ego-relative. `dynamic_obstacle_rel_vel_robot_t` subtracts robot translation and is the preferred relative-velocity input for later control learning.

## Dynamic Labels

Clearance is dynamic-only:

```text
min_dynamic_clearance = min_i(||robot_xy - obs_i_xy|| - robot_radius - obs_i_radius)
```

Visibility is structured-state based:

```text
dynamic_visible_t = min_dynamic_clearance_t < dynamic_visibility_radius
```

Dynamic collision and risk:

```text
dynamic_collision_t1 = min_dynamic_clearance_t1 <= 0
dynamic_risk_t1 = 1.0 if min_dynamic_clearance_t1 <= 0
dynamic_risk_t1 = exp(-min_dynamic_clearance_t1 / dynamic_risk_tau) otherwise
```

Horizon labels:

```text
dynamic_collision_within_H at step i =
  any dynamic_collision_t1 in step indices [i, i + H - 1]
```

Because `dynamic_collision_t1` is already the collision event after taking the action at step `i`, `dynamic_collision_within_H` corresponds to collision events from times `t+1` through `t+H`.

Event steps:

```text
dynamic_event_step_t =
  dynamic_visible_t
  OR dynamic_risk_t1 >= dynamic_risk_positive_threshold
  OR dynamic_collision_within_3
  OR dynamic_collision_within_5
```

The simulator, acceptance filter, inspector, and report use this same definition.

## Optional Action-Conditional Labels

V4 v1 does not implement privileged per-action labels such as `dynamic_collision_next_if_action` or `dynamic_risk_if_action`. The schema is intentionally focused on the main fixed-ID state pipeline; per-action rollout labels can be added later without changing the primary obstacle-state fields.
