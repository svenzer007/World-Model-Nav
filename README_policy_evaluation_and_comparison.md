# Longer-Horizons Policy Evaluation and Comparison

This document describes the current evaluation workflow for the `longer_horizons` branch.

Related docs:

- controller overview: [README_controllers.md](./README_controllers.md)
- model/data/architecture semantics: [README_model_data_and_architecture.md](./README_model_data_and_architecture.md)
- training workflow: [README_v4_training.md](./README_v4_training.md)

## Entry points

The current workflow uses three scripts:

1. `eval_structured_world_model.py`
2. `run_policy_eval.py`
3. `compare_policy_eval_runs.py`

Use the same learned checkpoint for offline model evaluation and `learned_structured` policy evaluation.

## Offline model evaluation

Use `eval_structured_world_model.py` to evaluate a trained structured world model on saved episode windows.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model_stage2.yaml \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt
```

Accepted dataset layout:

- `<data_dir>/episodes/episode_*.pkl`
- or `data_dir` itself as the `episodes/` directory

## Closed-loop policy evaluation

Use `run_policy_eval.py` to evaluate one controller at a time:

- `learned_structured`
- `baseline_structured`
- `cbf_qp`
- `mpc_static`

Examples:

Learned:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py   --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/epoch_009.pt   --num_episodes 50   --horizon 10 --output_dir artifacts/policy_eval_wm_100_100 --save_previews --save_videos
```

Baseline:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --controller baseline_structured \
  --num_episodes 10 \
  --output_dir artifacts/policy_eval_baseline
```

CBF:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --controller cbf_qp \
  --num_episodes 10 \
  --cbf_debug \
  --save_videos \
  --output_dir artifacts/policy_eval_cbf
```

MPC-static:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --controller mpc_static \
  --num_episodes 10 \
  --max_steps 250 \
  --mpc_horizon 10 \
  --output_dir artifacts/policy_eval_mpc_static
```

### Scenario sources

`run_policy_eval.py` supports two source modes.

Simulator-sampled mode:

- omit `--episode_source`
- the simulator samples fresh scenarios from the branch’s current distribution

Saved-episode replay mode:

- pass `--episode_source PATH`
- policy eval reconstructs scenario starts from saved episodes
- this reuses:
  - map
  - path
  - start pose
  - goal
  - dynamic obstacle routes
- it does not replay recorded expert actions

## Output layout

Each `run_policy_eval.py` output directory contains:

- `summary.json`
- `decision_logs/episode_*.json`
- `episode_results/episode_*.json`
- learned-only:
  - `rollout_error_stats/episode_*_rollout_error_stats.json`
  - `rollout_error_stats_summary.json`
- optional visual artifacts:
  - `preview/`
  - `video/`
  - `episodes/`

## Aggregate metrics and controller-specific summaries

The shared aggregate summary includes metrics such as:

- success rate
- dynamic collision rate
- static collision rate
- timeout rate
- average path length
- average progress
- average minimum dynamic clearance
- average minimum static clearance
- stop fraction
- action distribution
- mean score components

### Discrete controllers

For `baseline_structured`, `learned_structured`, and `mpc_static`:

- action distributions are discrete distributions over `ACTIONS`
- score components correspond to the chosen discrete action at each step
- `--disable_backward` removes the backward action before candidates are constructed
- if backward is present, the shared backward-policy helper may:
  - gate backward as infeasible
  - append `backward_gated` to `infeasible_reasons`
  - add `backward_penalty` to the backward candidate score

Shared discrete-controller weights and penalties:

- `w_progress`
- `w_heading`
- `w_path`
- `w_velocity`
- `static_collision_penalty`
- `dynamic_collision_penalty`
- `min_clearance_clip`
- `backward_penalty_enabled`
- `backward_penalty`
- `backward_gate_enabled`
- `dynamic_stop_clearance_threshold`
- `backward_dynamic_margin`

For `mpc_static` specifically:

- robot rollouts are finite-horizon repeated-action rollouts
- dynamic obstacles are frozen at their current positions over the horizon
- no checkpoint is required
- this is a horizon planner without a dynamic-obstacle prediction model
- backward handling now uses the same shared backward gate and penalty helper as `baseline_structured` and `learned_structured`

### Continuous CBF controller

For `cbf_qp`:

- non-stop commands are continuous and reported under `action_name = "cbf_qp"`
- exact stop still uses the normal stop identity
- the aggregate summary includes a `continuous_command_summary` block with:
  - `mean_v`
  - `mean_omega`
  - `mean_abs_v`
  - `mean_abs_omega`
  - `stop_fraction`
  - `fallback_fraction`
  - `average_control_deviation_norm`
  - `average_cbf_margin`
  - `average_v_feasible_interval_width`

This keeps discrete action histograms meaningful while still reporting continuous-command behavior for CBF.

CBF does not use the discrete backward-action framework. It is a continuous-command controller with its own CBF-specific priority and fallback logic.

## Visual artifacts

### Standard previews

All controllers can save static previews via `--save_previews`.

### MP4 video export

- MP4 is the supported animated export format for all controllers
- use `--save_videos` to write per-episode MP4 files under `video/`
- `--save_gifs` is a deprecated compatibility alias
- if `--save_gifs` is used, policy eval still saves MP4 videos and prints a deprecation warning
- policy eval no longer writes animated GIF artifacts

When candidate-debug data is available, MP4 videos use a landscape layout:

- map, robot, path, goal, and obstacles on the left
- weighted score-contribution table on the right

The weighted contribution tables are presentation-only. They do not change controller scoring or selection.

For the discrete controllers, the table shows weighted score contributions rather than raw unweighted terms:

- `progress = w_progress * goal_progress_cost`
- `heading = w_heading * heading_cost`
- `path = w_path * path_cost`
- `velocity = w_velocity * velocity_cost`
- `clearance = clearance_cost`
- `infeasibility = infeasibility_penalty`
- `total = total_cost`

### CBF candidate-debug videos

The `cbf_qp` video overlay uses controller debug fields to explain the current omega-candidate choice.

The CBF MP4 overlay shows:

- the map, robot, goal, and obstacles
- the selected command and CBF solve status
- the evaluated omega candidates
- weighted score contributions for each candidate
- which candidate was selected

For the discrete controllers:

- `baseline_structured`
- `learned_structured`
- `mpc_static`

the MP4 overlay uses the same landscape map-left/table-right layout when candidate debug is available and shows weighted score contributions for each discrete action candidate.

## Comparing two completed runs

Use `compare_policy_eval_runs.py` after two `run_policy_eval.py` runs have completed.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python compare_policy_eval_runs.py \
  --run_a artifacts/policy_eval_run_a \
  --run_b artifacts/policy_eval_run_b \
  --label_a run_a \
  --label_b run_b \
  --output_dir artifacts/policy_eval_comparison
```

The comparison output writes:

- `comparison_summary.json`

It compares:

- aggregate metrics
- action distributions
- mean score components
- learned rollout-error metrics when both runs provide them
- per-episode deltas when both runs provide episode results

## Practical guidance

- use `baseline_structured` as the analytic true-dynamics baseline
- use `mpc_static` to measure what finite-horizon robot planning can do with a frozen obstacle field
- use `learned_structured` for the longer-horizon learned/world-model controller
- use `cbf_qp` for the continuous local safety-filter baseline
- compare completed run directories rather than mixing controller-specific reports by hand

The intended comparison is:

- `cbf_qp`: local safety filtering using current or causal obstacle state
- `mpc_static`: finite-horizon robot planning with frozen dynamic obstacles
- `learned_structured`: finite-horizon action evaluation with predicted obstacle evolution

## Useful command patterns

Disable backward globally for the discrete controllers:

```bash
python run_policy_eval.py \
  --controller baseline_structured \
  --num_episodes 10 \
  --disable_backward
```

Run MPC-static with backward policy enabled:

```bash
python run_policy_eval.py \
  --controller mpc_static \
  --num_episodes 10 \
  --mpc_horizon 10 \
  --mpc_debug
```

Run CBF with margin-priority risk mode:

```bash
python run_policy_eval.py \
  --controller cbf_qp \
  --num_episodes 10 \
  --cbf_debug \
  --cbf_margin_priority_use_margin_trigger \
  --cbf_margin_priority_threshold 0.17
```
