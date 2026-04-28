# World-Model Dynamic State Data

This is the authoritative design note for the `world_model` branch under `/home/stavan/s4/rnl/world_model`.

For the stage-1 / stage-2 training workflow, see [README_v4_training.md](/home/stavan/s4/rnl/world_model/README_v4_training.md).

## Active path summary

The active practical world-model path is:

- `model_type: structured_obstacle_rssm_vc`
- `model_version: version_c_world_delta_v3`

Its intended semantics are:

- keep the RSSM / prior-posterior / latent-rollout structure
- use obstacle world position, causal world velocity, and radius as the practical obstacle state
- predict world-frame obstacle delta as the grounded target
- keep rollout bookkeeping in world coordinates
- derive robot-relative obstacle geometry analytically inside the controller for scoring

So this cleanup changes practical state semantics, not the underlying world-model family.

## Current branch audit

Before this cleanup, the Version-C / RSSM path stored both world-frame and robot-frame obstacle fields:

- `dynamic_obstacles_world_t`
- `dynamic_obstacles_world_t1`
- `dynamic_obstacle_delta_world_t`
- `dynamic_obstacles_robot_t`
- `dynamic_obstacles_robot_t1`
- `dynamic_obstacle_delta_robot_t`
- `dynamic_obstacle_radii`
- `dynamic_obstacle_vel_world_t`
- `dynamic_obstacle_vel_robot_t`
- `dynamic_obstacle_rel_vel_robot_t`
- `dynamic_obstacle_rel_vel_robot_fd_t`
- `dynamic_obstacle_rel_vel_robot_fd_t1`

The active Version-C learned path used:

- posterior input per obstacle: robot-relative position, robot-frame finite-difference relative velocity, radius
- target per obstacle: robot-frame delta and next-step robot-frame finite-difference velocity
- prior rollout supervision on those same robot-frame targets
- controller rollout directly in robot-relative obstacle state

So even though world-frame obstacle information was stored, robot-frame semantics were central all the way through the practical Version-C branch.

## Velocity audit

The branch stored multiple velocity notions:

- `dynamic_obstacle_vel_world_t = (p_world_t1 - p_world_t) / dt`
- `dynamic_obstacle_vel_robot_t = R_t dynamic_obstacle_vel_world_t`
- `dynamic_obstacle_rel_vel_robot_t = R_t (v_obs_world - v_robot_world)`
- `dynamic_obstacle_rel_vel_robot_fd_t = (dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t) / dt`

The active learned Version-C input was `dynamic_obstacle_rel_vel_robot_fd_t`.

That quantity had two practical problems:

1. It was robot-frame, so it mixed obstacle dynamics with ego-motion-induced frame rotation and translation.
2. It was future-derived, because it used `t -> t+1` positions to construct the input state at time `t`.

So the old practical Version-C path did have the causal/semantic issue:

- the learned input velocity used future information
- the obstacle state was expressed in the robot frame rather than a smoother world frame

`dynamic_obstacle_rel_vel_robot_t` was also not the right practical learned state, because it is a translational-relative robot-frame velocity and is not equal to the finite-difference derivative of robot-relative position during turning.

## Why move to world-frame state semantics

Obstacle world dynamics are smoother than robot-frame obstacle motion.

In the robot frame, the model has to absorb:

- obstacle world motion
- robot translation
- robot rotation
- rotating-frame geometry effects

Because the action set contains significant angular motion, robot-frame obstacle velocity is highly nonstationary even when obstacle world motion is simple.

The cleaner decomposition is:

- learned RSSM predicts obstacle world motion
- controller analytically rolls out robot pose
- future robot-relative obstacle geometry is derived from predicted obstacle world positions plus rolled-out robot pose

That keeps the latent world model focused on obstacle dynamics and moves frame geometry back to the controller where it is analytic and exact.

## Causal velocity-state requirement

The learned obstacle input velocity at time `t` must be causal.

The practical Version-C state is now defined as:

```text
v_world_t =
- simulator-provided current obstacle world velocity, for episode step t = 0
- (p_world_t - p_world_t-1) / dt, for episode step t >= 1
```

This is serialized as:

- `dynamic_obstacle_vel_world_causal_t`

Why this is the correct learned state:

- it matches eval-time availability
- it avoids future leakage
- it gives a grounded first-order world-motion state

Not acceptable as learned input state:

```text
(p_world_t+1 - p_world_t) / dt
```

That remains a valid realized-step diagnostic quantity as `dynamic_obstacle_vel_world_t`, but it is not the practical learned input state.

## New Version-C world-model semantics

The RSSM / latent rollout structure is preserved.

The practical obstacle observation for the cleaned-up Version-C path is now:

- `x_world`
- `y_world`
- `vx_world_causal`
- `vy_world_causal`
- `radius`

The practical target is intentionally simple:

- `dynamic_obstacle_delta_world_t = dynamic_obstacles_world_t1 - dynamic_obstacles_world_t`

So the cleaned-up Version-C path is:

- world-frame causal obstacle state in
- world-frame delta out
- no explicit next-velocity head
- prior/posterior latent world-model structure retained

If rollout-side velocity bookkeeping is needed outside the latent chain, it is derived causally from predicted positions:

```text
p_world_next = p_world_prev + pred_delta_world
v_world_next = (p_world_next - p_world_prev) / dt
```

## Controller decomposition

Learned world model:

- predicts obstacle world motion

Controller:

- stores current obstacle world position, causal world velocity, and radius
- seeds posterior updates from real observed world-frame obstacle state
- rolls out robot pose analytically under candidate actions
- propagates predicted obstacle world positions from decoded deltas
- transforms predicted obstacle world positions into the rolled-out robot frame for scoring

Future robot-relative obstacle position is therefore not the main learned model target anymore. It is a controller-side derived quantity.

This is the key decomposition to keep in mind:

- learned model predicts obstacle world motion
- controller rolls out robot motion analytically
- scoring happens after transforming predicted obstacle world positions into the rolled-out robot frame

## Action and goal plumbing

The public Version-C interface still accepts action and goal tensors to keep churn low.

But for the cleaned-up world-frame semantics, the effective obstacle-dynamics path does not meaningfully rely on robot action:

- action/goal plumbing is preserved at the interface
- the world-frame practical Version-C path treats obstacle dynamics as world-dynamics-first
- retained context plumbing should be understood as architectural compatibility, not as a statement that robot action should drive obstacle world motion targets

## Risk and collision auxiliaries

Risk and collision auxiliaries are not central to the cleaned-up Version-C path.

The practical learned path stays motion-only:

- delta loss
- latent consistency loss
- rollout delta supervision
- rollout latent consistency

No risk/collision auxiliary heads are required for the practical Version-C learned model path.

## Edge cases and assumptions

- `t = 0` causal bootstrap:
  - use stored snapshot velocity from `dynamic_obstacles_t[*]["velocity"]`
- `t >= 1` causal velocity:
  - use previous/current world positions from the same episode
- fixed obstacle count:
  - still assumed
- stable slot order:
  - preserved by simulator construction and controller canonical ordering
- episode/window boundaries:
  - training windows remain contiguous and do not cross episodes
  - loader synthesis may look one step backward in the episode even if that row is outside the sampled window
- `dt` usage:
  - generation uses `robot_config.dt`
  - loader/training use stored row `dt` with fallback
- compatibility:
  - older datasets remain usable through loader-side synthesis of `dynamic_obstacle_vel_world_causal_t`
  - newly generated datasets are marked `v4_dynamic_state_data_3`
  - old `version_c_motion_v2` checkpoints are intentionally incompatible with the new world-frame semantic path
