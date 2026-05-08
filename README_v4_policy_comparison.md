# World-Model Policy Comparison

This README covers the comparison step for the current `world_model` workflow.

Policy comparison is a compare-two-output-dirs workflow:

1. run `run_policy_eval.py` once for the first checkpoint or setting
2. run `run_policy_eval.py` again for the second checkpoint or setting
3. compare the two completed output directories with `compare_policy_eval_runs.py`

For the full workflow, including simulator-default policy eval and saved-episode replay, see [README_v4_evaluation.md](./README_v4_evaluation.md).

## Inputs

`compare_policy_eval_runs.py` takes:

- `--run_a`
- `--run_b`
- optional `--label_a`
- optional `--label_b`
- `--output_dir`

Example:

```bash
cd /home/stavan/s4/rnl/world_model
python compare_policy_eval_runs.py \
  --run_a artifacts/policy_eval_h10_a \
  --run_b artifacts/policy_eval_h10_b \
  --label_a checkpoint_a \
  --label_b checkpoint_b \
  --output_dir artifacts/policy_eval_comparison_h10
```

## Expected Policy-Eval Run Layout

Each input run directory should come from `run_policy_eval.py` and contain:

- `summary.json`
- `episode_results/episode_*.json`
- `decision_logs/episode_*.json`
- learned-only: `rollout_error_stats_summary.json`
- optional `rollout_error_stats/episode_*_rollout_error_stats.json`
- optional visual artifacts

## Outputs

The comparison utility writes:

- `comparison_summary.json`

It also prints a short terminal summary for the main aggregate metrics.

The comparison payload includes:

- run labels and paths
- checkpoint references when present
- controller names
- aggregate metric comparisons
- action-distribution comparisons
- mean score-component comparisons
- rollout-error metric comparisons when both runs provide rollout summaries
- per-episode deltas when both runs provide episode results

## Compared Metrics

The current comparison step includes:

- success rate
- dynamic collision rate
- static collision rate
- timeout rate
- average path length
- average progress
- average min dynamic clearance
- average min static clearance
- stop fraction
- action distribution
- mean score components
- learned rollout-error summary metrics when present

## Relationship To Policy Eval

`run_policy_eval.py` is the script that runs either:

- `learned_structured`
- `baseline_structured`

`compare_policy_eval_runs.py` does not rerun the simulator or the controller. It compares two completed policy-eval outputs after the fact.

## Action-Space Notes

For the `world_model` branch, the canonical runtime action definition lives in:

- `sim2d/config.py::ACTIONS`

That is the normal single place to edit when you want to change the branch’s runtime action space again.

For this branch:

- policy eval candidate enumeration comes from the controllers, and both controllers derive candidates from `ACTIONS`
- action names, action indices, and runtime `(v, omega)` values come from `ACTIONS`
- `run_policy_eval.py --disable_backward` is a runtime-only filter on top of `ACTIONS`
- the active `version_c_world_delta_v3` policy/runtime shell now follows the same canonical runtime action setup as `longer_horizons`
- older or non-active offline-eval fallback paths still keep their own `action_vocab_size` / YAML fallback behavior where needed
