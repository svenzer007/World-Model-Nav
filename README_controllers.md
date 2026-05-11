# Longer-Horizons Controllers

This document describes the controllers currently implemented in the `longer_horizons` branch.

The branch currently supports four policy-eval controllers:

- `baseline_structured`
- `learned_structured`
- `cbf_qp`
- `mpc_static`

Other controller families may be added later, but this README only documents code that exists in the current branch.

## Comparison Table

| Controller | Action type | Horizon use | Dynamic obstacle model | Static map handling | Backward handling | Requires checkpoint? | Main limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_structured` | Discrete `ACTIONS` | Repeated-action analytic rollout over `horizon` | True simulator obstacle motion | Planning collision + true static collision in rollout | Shared backward gate + backward penalty unless backward is globally removed | No | Uses true future obstacle motion, so it is not deployable as-is |
| `learned_structured` | Discrete `ACTIONS` | Repeated-action analytic robot rollout over `horizon` | Learned world-frame obstacle prediction | Planning collision + true static collision in rollout | Shared backward gate + backward penalty unless backward is globally removed | Yes | Quality depends on checkpoint prediction accuracy |
| `cbf_qp` | Continuous `v, omega` | Local nominal command plus omega scan and short static rollout | Current/causal obstacle state inside dynamic CBF constraints | Short static/planning rollout over `cbf_static_rollout_steps` | Not part of the discrete backward framework | No | Local/reactive, not a long-horizon planner |
| `mpc_static` | Discrete `ACTIONS` | Repeated constant-action rollout over `mpc_horizon` | Frozen obstacles over the horizon | Planning collision + true static collision in rollout | Shared backward gate + backward penalty unless backward is globally removed | No | No dynamic-obstacle motion model |

## Shared runtime interface

All four controllers are evaluated through the same closed-loop policy-eval interface in `policy_eval/common.py`.

Each controller is expected to provide:

- `reset()`
- `select_action(...)`
- `commit_step(...)`

Each controller returns a decision payload containing:

- `action_index`
- `action_name`
- `v`
- `omega`
- `debug`
- `chosen_action_min_clearance`
- `emergency_override_used`

The policy-eval loop then applies the returned continuous `(v, omega)` command in the simulator.

## Shared discrete-controller semantics

The three discrete controllers:

- `baseline_structured`
- `learned_structured`
- `mpc_static`

share the same runtime action set source:

- `sim2d/config.py::ACTIONS`

Shared discrete-controller score structure:

```text
total_cost =
    w_progress * goal_progress_cost
  + w_heading * heading_cost
  + w_path * path_cost
  + w_velocity * velocity_cost
  + clearance_cost
  + infeasibility_penalty
```

Shared discrete-controller backward-policy controls:

- `--disable_backward`
- `--backward_penalty_enabled`
- `--backward_penalty`
- `--backward_gate_enabled`
- `--dynamic_stop_clearance_threshold`
- `--backward_dynamic_margin`
- `--backward_action_name`

Behavior:

- if `--disable_backward` is passed, the backward action is removed before candidate construction
- otherwise the backward candidate is present and the shared backward-policy helper may:
  - gate backward as infeasible
  - append `backward_gated` to `infeasible_reasons`
  - add `backward_penalty` to the backward candidate `total_cost`

## `baseline_structured`

Implemented in:

- `controllers/baseline_structured_controller.py`

Role:

- analytic expert-style baseline using true dynamic-obstacle rollout
- no learned model
- discrete action selection over `sim2d/config.py::ACTIONS`
- runtime horizon controlled by `--horizon`

Behavior:

- rolls the robot forward analytically for each discrete action
- uses true dynamic-obstacle motion from the simulator
- computes planning-map point collision and true static collision during rollout
- scores each action with the expert-style cost:
  - progress
  - heading
  - path adherence
  - velocity penalty
  - clearance cost
  - infeasibility penalty
- includes the backward-action penalty and backward-action gate used in this branch

Key hyperparameters:

- `horizon`
- `w_progress`, `w_heading`, `w_path`, `w_velocity`
- `static_collision_penalty`, `dynamic_collision_penalty`
- `min_clearance_clip`
- `backward_penalty_enabled`, `backward_penalty`
- `backward_gate_enabled`
- `dynamic_stop_clearance_threshold`
- `backward_dynamic_margin`
- `disable_backward`

Assumptions:

- dynamic obstacle rollout is available from the simulator state
- runtime action set comes from `ACTIONS`
- this is the main non-learned structured baseline for comparison

## `learned_structured`

Implemented in:

- `controllers/learned_structured_controller.py`

Role:

- longer-horizon / world-model controller
- learned obstacle prediction + analytic robot rollout
- discrete action selection over `sim2d/config.py::ACTIONS`
- runtime horizon controlled by `--horizon`

Behavior:

- loads a structured world-model checkpoint
- predicts obstacle motion in world frame
- rolls out robot motion analytically under each candidate action
- transforms predicted obstacle world positions into the rolled-out robot-relative geometry for scoring
- reuses the same expert-style action-scoring structure as the baseline where practical
- uses the same shared backward gate and backward penalty behavior as the baseline

What is learned vs analytic:

- learned: obstacle motion prediction in world frame
- analytic: robot rollout, static collision checking, path/heading/clearance scoring

Key hyperparameters:

- `horizon`
- `w_progress`, `w_heading`, `w_path`, `w_velocity`
- `static_collision_penalty`, `dynamic_collision_penalty`
- `min_clearance_clip`
- `backward_penalty_enabled`, `backward_penalty`
- `backward_gate_enabled`
- `dynamic_stop_clearance_threshold`
- `backward_dynamic_margin`
- `disable_backward`
- `checkpoint`

Assumptions:

- checkpoint is required
- learned obstacle prediction and controller rollout are intentionally separated
- current practical branch uses the world-frame `factorized_world_v3` family described in `README_model_data_and_architecture.md`

## `cbf_qp`

Implemented in:

- `controllers/cbf_qp_controller.py`

Role:

- local continuous-command safety-filter baseline
- nominal navigation command + dynamic-obstacle CBF filtering
- comparison baseline against the longer-horizon learned controller

Behavior:

- outputs continuous `v, omega`
- scans omega candidates
- can enable heading lookahead with `cbf_heading_lookahead_enabled` / `cbf_heading_lookahead_time`
- evaluates dynamic CBF feasibility by projecting the nominal linear speed into a feasible interval
- checks static feasibility through short-horizon rollout against the static/planning map
- uses score-based feasible selection in normal mode
- can enable margin-trigger priority mode with `cbf_margin_priority_use_margin_trigger`
- in risk mode, uses continuity-preserving tie-breaking when best CBF margins collapse near zero

Important semantics:

- this controller is intentionally local and reactive
- it does not use the learned world model
- it may slow, stop, or become conservative near moving obstacles
- general continuous commands use `action_index = -1` and `action_name = "cbf_qp"`
- exact stop commands still use the normal stop identity so stop statistics remain meaningful
- it does not use the discrete backward framework

Key hyperparameters:

- `cbf_alpha`, `cbf_safety_margin`
- `cbf_k_v`, `cbf_k_omega`
- `cbf_omega_scan_enabled`, `cbf_omega_candidates`
- `cbf_static_rollout_steps`
- `cbf_heading_lookahead_enabled`, `cbf_heading_lookahead_time`
- `cbf_margin_priority_use_margin_trigger`
- `cbf_margin_priority_threshold`
- `cbf_margin_tie_positive_eps`
- `cbf_w_omega_dev`, `cbf_w_cbf_margin`
- `cbf_use_reverse_fallback`, `cbf_reverse_fallback_speed`
- `cbf_v_min`, `cbf_v_max`
- `cbf_omega_min`, `cbf_omega_max`

## `mpc_static`

Implemented in:

- `controllers/mpc_static_controller.py`

Role:

- finite-horizon repeated-action MPC baseline
- no learned model
- no dynamic-obstacle motion model
- discrete action selection over `sim2d/config.py::ACTIONS`
- runtime horizon controlled by `--mpc_horizon`

Behavior:

- evaluates one constant-action robot rollout per runtime action
- uses the same unicycle rollout style as the other analytic controllers
- treats dynamic obstacles as frozen at their current positions over the horizon
- recomputes dynamic clearances locally against the frozen obstacle snapshot over the whole rollout
- scores candidates with the same expert-style cost structure used by the structured controllers
- applies the same shared backward gate and backward penalty helper as the other discrete controllers when backward is present

Important semantics:

- this controller plans the robot trajectory over a horizon
- it does not extrapolate obstacle velocity
- it does not step obstacle routes over the horizon
- it does not use future labels or the learned world model
- its action outputs are discrete, so its action histograms are directly comparable to `baseline_structured` and `learned_structured`

Key hyperparameters:

- `mpc_horizon`
- `mpc_obstacle_model`
- `mpc_debug`
- shared score weights through `score_cfg`:
  - `w_progress`, `w_heading`, `w_path`, `w_velocity`
  - `static_collision_penalty`, `dynamic_collision_penalty`
  - `min_clearance_clip`
- shared backward controls:
  - `disable_backward`
  - `backward_penalty_enabled`, `backward_penalty`
  - `backward_gate_enabled`
  - `dynamic_stop_clearance_threshold`
  - `backward_dynamic_margin`

## How they relate

- `baseline_structured` answers: how well does expert-style analytic rollout perform with true obstacle motion?
- `mpc_static` answers: how well does finite-horizon robot planning perform when the obstacle field is frozen at the current timestep?
- `learned_structured` answers: how well does the learned world-model controller perform under the same policy-eval setup?
- `cbf_qp` answers: how well does a local CBF safety filter perform when compared against the longer-horizon controller and the analytic baseline?

The intended interpretation is:

- `cbf_qp`: local safety filter using current or causal obstacle state
- `mpc_static`: horizon robot planning with a frozen obstacle field
- `learned_structured`: horizon action evaluation with predicted obstacle evolution

A future MPC variant could use a constant-velocity obstacle model derived from causal finite-difference velocity, but that is not implemented in this branch today.

For how to run these controllers and compare their outputs, see [README_policy_evaluation_and_comparison.md](./README_policy_evaluation_and_comparison.md).
