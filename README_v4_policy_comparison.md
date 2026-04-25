# V4 Policy Comparison

This README covers the closed-loop controller comparison runner:

- `run_policy_comparison_v4.py`

It compares:

- `baseline_structured`
- `learned_structured`

The learned controller loads a trained structured world-model checkpoint. The documented example checkpoint for this branch is:

- `/home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt`

## What The Script Does

For each sampled simulator episode, the script runs both controllers on the same scenario:

- `baseline_structured`: analytic robot rollout plus true cloned dynamic-obstacle rollout
- `learned_structured`: analytic robot rollout plus learned robot-relative obstacle rollout from the checkpoint

Both controllers evaluate the same repeated-action candidate set:

- `stop`
- `forward`
- `forward_left`
- `forward_right`
- `slow_left`
- `slow_right`

The script samples scenarios internally through the simulator. It does not read the training dataset directory directly.

## Required Inputs

The main required input is the learned checkpoint:

```bash
--checkpoint /path/to/checkpoint.pt
```

Useful runtime arguments from the current script:

- `--output_dir`
- `--num_episodes`
- `--seed`
- `--max_steps`
- `--num_dynamic_obstacles`
- `--device`
- `--horizon`
- `--save_episodes`
- `--save_previews`
- `--save_gifs`
- `--save_videos`

This branch’s default policy-comparison horizon is `10`, and the default output directory is `artifacts/policy_comparison_v4_h10`.

## Smoke Test

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_comparison_v4.py \
  --num_episodes 2 \
  --max_steps 80 \
  --horizon 10 \
  --checkpoint /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt \
  --output_dir /home/stavan/s4/rnl/longer_horizons/artifacts/policy_comparison_v4_smoke \
  --save_previews
```

## Outputs

The runner always writes:

- `<output_dir>/summary.json`
- `<output_dir>/baseline_structured/decision_logs/episode_*.json`
- `<output_dir>/baseline_structured/episode_results/episode_*.json`
- `<output_dir>/learned_structured/decision_logs/episode_*.json`
- `<output_dir>/learned_structured/episode_results/episode_*.json`
- `<output_dir>/learned_structured/rollout_error_stats/episode_*_rollout_error_stats.json`
- `<output_dir>/learned_structured/rollout_error_stats_summary.json`

Optional outputs are controlled by flags:

- `--save_episodes` writes `episodes/episode_*.pkl`
- `--save_previews` writes `preview/episode_*.png`
- `--save_gifs` writes `gif/episode_*.gif`
- `--save_videos` writes `video/episode_*.mp4`

## Summary Output

`summary.json` includes:

- checkpoint path
- run config
- aggregated controller metrics
- per-episode compact results for each controller

The per-controller aggregates include:

- success rate
- dynamic collision rate
- static collision rate
- timeout rate
- average path length
- average progress
- average minimum dynamic/static clearance
- stop fraction
- action distribution
- mean score components

## Learned Rollout Error Stats

The learned controller additionally writes rollout diagnostics:

- per-episode `rollout_error_stats/episode_*_rollout_error_stats.json`
- cross-episode `rollout_error_stats_summary.json`

These files summarize:

- position/velocity rollout error versus true robot-relative obstacle rollout
- clearance error
- dynamic collision confusion statistics
- horizon-growth summaries
- action-conditioned rollout error summaries

## Troubleshooting

If the run fails immediately, check:

- the checkpoint path exists and is a full training checkpoint
- the Python environment has the dependencies needed by the learned controller
- the output directory is writable
- the checkpoint matches the structured world-model codepath in this repo

If results look inconsistent:

- remember the script defaults to `--horizon 10` unless you override it
- remember policy comparison uses internally sampled simulator scenarios, not a dataset root
- remember the learned-only rollout diagnostics are written under `learned_structured/rollout_error_stats/`

## Note On Defaults

The script still ships with generic defaults such as:

- `--checkpoint artifacts/structured_world_model_rollout_from_epoch_10/best.pt`
- `--output_dir artifacts/policy_comparison_v4_h10`

For this branch’s documented workflow, prefer explicitly passing the checkpoint path and output directory you want, as shown above.
