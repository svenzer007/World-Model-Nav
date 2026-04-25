# V4 Checkpoint Evaluation

This README covers offline evaluation of a trained structured world-model checkpoint on a dataset:

- `eval_structured_world_model.py`

The documented example checkpoint for this branch is:

- `/home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt`

For the dataset format expected by evaluation, see [README_v4_dynamic_state_data.md](./README_v4_dynamic_state_data.md).

## What The Evaluation Script Does

The evaluation script:

- loads a full training checkpoint
- reconstructs the model from checkpoint/config model settings
- loads the validation split from the dataset configured in `data.data_dir`
- computes offline metrics over the validation windows

Current metrics include:

- `delta_mae`
- `delta_rmse`
- `risk_mae`
- `risk_mse`
- collision precision/recall/F1 for:
  - `collision_next`
  - `collision_within_3`
  - `collision_within_5`
- `normalized_action_sensitivity`
- optional rollout metrics when `rollout.eval_enabled: true`:
  - `rollout_delta_mae`
  - `rollout_delta_rmse`

This is checkpoint evaluation on saved data. It is not closed-loop policy evaluation.

## Required Inputs

The script requires:

- `--config`
- `--checkpoint`

Optional arguments:

- `--output`
- `--threshold`

The dataset path comes from the config:

```yaml
data:
  data_dir: ./dataset
```

The evaluation loader expects:

- `<data_dir>/episodes/episode_*.pkl`
- or `data_dir` to point directly at the `episodes/` directory

## Example Command

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --checkpoint /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt
```

Example with an explicit output path:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --checkpoint /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt \
  --output /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/eval_metrics_epoch_015.json
```

Example with a custom classification threshold:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --checkpoint /home/stavan/s4/rnl/longer_horizons/artifacts/structured_world_model_rollout_from_epoch_10/epoch_015.pt \
  --threshold 0.5
```

## Checkpoint And Config Requirements

The script loads:

- the full checkpoint from `--checkpoint`
- the config from `--config`

Model construction uses:

- `checkpoint["config"]["model"]` if present
- otherwise `config["model"]`

The data split and evaluation settings come from the config file, including:

- `data.data_dir`
- `data.sequence_length`
- `data.stride`
- `data.train_fraction`
- `evaluation.threshold`
- `evaluation.output_path`

Because the script uses the config for dataset loading, make sure `data.data_dir` points at the intended dataset before running evaluation.

## Outputs

If `--output` is omitted, the script writes to:

```text
evaluation.output_path
```

For this branch, that default should now resolve into:

```text
./artifacts/structured_world_model_rollout_from_epoch_10/eval_metrics.json
```

The script also prints the metrics JSON to stdout.

## Troubleshooting

Common issues:

- Wrong dataset path:
  - `data.data_dir` must lead to `episodes/episode_*.pkl`
- Empty validation split:
  - very small datasets can leave no validation windows after the deterministic split
- Checkpoint/config mismatch:
  - model shape settings in the checkpoint and config must match the intended codepath
- Missing expected fields:
  - the evaluation loader requires the fields documented in `README_v4_dynamic_state_data.md`

If `use_goal: true`, the dataset must provide `goal_t`.

If `use_lidar: true`, the dataset must provide `lidar_t`.
