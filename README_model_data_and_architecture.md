# Longer-Horizons Model, Data, and Architecture Notes

This document describes the current learned-model/data semantics for the `longer_horizons` branch.

Related docs:

- training workflow: [README_v4_training.md](./README_v4_training.md)
- controller overview: [README_controllers.md](./README_controllers.md)
- evaluation workflow: [README_policy_evaluation_and_comparison.md](./README_policy_evaluation_and_comparison.md)

## Current practical branch

The active practical learned-model branch uses:

- `model_type: factorized_world_v3`
- world-frame obstacle prediction
- causal world-velocity input
- delta-only motion targets

The intended decomposition is:

- the learned model predicts obstacle motion in world frame
- the controller rolls out robot motion analytically
- future robot-relative geometry is derived inside the controller for scoring

## Dynamic-state representation

The primary learned per-obstacle input state is:

- `x_world`
- `y_world`
- `vx_world_causal`
- `vy_world_causal`
- `radius`

The primary learned target is:

- world-frame obstacle displacement for `t -> t+1`

```text
dynamic_obstacle_delta_world_t = dynamic_obstacles_world_t1 - dynamic_obstacles_world_t
```

The practical branch intentionally keeps the target simple:

- predict world-frame delta only
- do not require a separate learned next-velocity head
- derive next-step velocity causally from consecutive positions when rollout needs it

## Causal world velocity

The learned model still uses obstacle velocity as input, but it must be causal.

At time `t`, the input velocity is:

```text
v_world_t =
- simulator-provided current obstacle world velocity, for t = 0
- (p_world_t - p_world_t-1) / dt, for t >= 1
```

This is important because it:

- matches what is available at evaluation time
- avoids future leakage
- provides a grounded first-order motion state

The non-causal quantity

```text
(p_world_t+1 - p_world_t) / dt
```

is not a valid model input state, although it can still exist as an offline diagnostic or realized-step label.

## Why world-frame prediction is used

Obstacle motion happens in world coordinates, while robot-relative geometry changes with both:

- obstacle motion
- robot translation
- robot rotation

Predicting directly in the robot frame forces the model to absorb ego-motion-induced coordinate changes. The world-frame formulation avoids that by:

- learning obstacle motion in a smoother frame
- keeping the model focused on obstacle dynamics
- leaving frame transformation to the analytic controller

## Controller-facing semantics

The learned controller uses the current branch decomposition:

1. predict obstacle motion in world frame
2. roll out robot motion under candidate actions
3. transform predicted obstacle world positions into the rolled-out robot frame
4. score dynamic clearance and collision in that derived geometry

This keeps the controller scoring philosophy close to the analytic baseline while changing only the obstacle-motion source.

## How the controllers differ from the learned model

The current branch intentionally separates model/data assumptions from controller runtime assumptions.

`learned_structured`

- requires the learned world-frame obstacle model
- consumes causal world velocity as model input
- predicts obstacle motion in world frame
- keeps robot rollout, static collision checks, and candidate scoring analytic

`baseline_structured`

- does not use the learned model
- does not need any learned-data assumptions at runtime
- uses true simulator obstacle motion during candidate rollout
- is best interpreted as an analytic true-dynamics reference baseline

`mpc_static`

- does not use the learned model
- does not need obstacle velocity as a motion model
- treats dynamic obstacles as frozen at their current positions across the whole horizon
- recomputes dynamic clearance against that frozen snapshot during candidate evaluation

`cbf_qp`

- does not use the learned model
- uses current/causal obstacle state directly inside the CBF constraint computation
- is a continuous local safety filter, not a discrete horizon action evaluator
- can use heading lookahead and margin-priority risk mode, but does not rely on learned obstacle evolution

## Architecture context

The practical branch keeps the same broad recurrent structure:

- shared per-obstacle encoder
- recurrent temporal state
- shared per-obstacle prediction head

For `factorized_world_v3`, the effective obstacle-dynamics path is not meaningfully driven by robot action:

- action tensors are still accepted by the public interface for plumbing stability
- obstacle prediction is driven by the obstacle state representation rather than action-conditioned obstacle dynamics

## Dataset expectations

The current practical branch expects loader-side access to:

- obstacle position in world frame
- causal obstacle velocity in world frame
- radius
- world-frame delta target
- action tensors for interface compatibility
- optional goal features when enabled

Older datasets can still be usable if the loader can synthesize causal world velocity from:

- stored snapshot velocity at the first step
- previous world positions afterward

These dataset/model assumptions are required for `learned_structured`, but not for:

- `baseline_structured`
- `mpc_static`
- `cbf_qp`

Those controllers still depend on the simulator/state interfaces exposed at policy-eval time, but they do not require learned-model retraining or dataset-format changes.

## Stable assumptions in the current branch

- dynamic obstacle count is fixed by generation
- obstacle IDs and slot ordering are expected to remain stable within an episode
- training windows remain contiguous and do not cross episode boundaries
- `dt` is effectively constant within an episode, with row-level handling retained for robustness

## Practical takeaway

The current branch is intentionally built around this separation:

- learned model: obstacle dynamics in world frame
- controller: ego-motion rollout and robot-relative scoring

That separation is the key assumption behind the present training, policy-eval, and controller stack.

A future MPC variant could use a causal constant-velocity obstacle model, but that is not implemented in the current branch.
