# Longer-Horizons Evaluation

This is the main evaluation guide for the current `longer_horizons` branch.

The active branch uses:

- `model_type: factorized_world_v3`
- world-frame obstacle prediction
- causal world velocity input
- delta-only target
- analytic robot rollout in the controller, with predicted obstacle world positions transformed into robot-relative geometry for scoring

For exact state semantics and required fields, see [dynamic_state_data.md](./dynamic_state_data.md).
For training, see [README_v4_training.md](./README_v4_training.md).
For compare-two-runs details, see [README_v4_policy_comparison.md](./README_v4_policy_comparison.md).

## Workflow

The current workflow has three entrypoints:

1. `eval_structured_world_model.py`
2. `run_policy_eval.py`
3. `compare_policy_eval_runs.py`

Use the same checkpoint for offline model eval and policy eval.

## Offline Model Eval

Use `eval_structured_world_model.py` to evaluate the world model on saved dataset windows from `data.data_dir`.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model_stage2.yaml \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt
```

To evaluate on a new dataset, point the config at the dataset root:

```yaml
data:
  data_dir: /path/to/your/dataset
```

Accepted dataset layout:

- `<data_dir>/episodes/episode_*.pkl`
- or `data_dir` itself as the `episodes/` directory

## Policy Eval

Use `run_policy_eval.py` to evaluate one learned controller:

- `learned_structured`

In this branch, “model + policy in tandem” means:

1. the learned model predicts obstacle world motion
2. the controller rolls out robot pose analytically under candidate actions
3. predicted obstacle world positions are transformed into the rolled-out robot frame
4. dynamic clearance and collision scoring happen in that derived robot-relative geometry

### Mode 1: Simulator Default

If `--episode_source` is omitted, policy eval uses the branch’s current simulator-sampled episode distribution.

The main controls are:

- `--num_episodes`
- `--seed`
- `--max_steps`
- `--num_dynamic_obstacles`
- `--map_width`
- `--map_height`
- `--disable_backward`

These are runtime policy-eval overrides only. They do not change the base
`sim2d` defaults used as the training/data-generation baseline.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt \
  --num_episodes 10 \
  --seed 1000 \
  --horizon 10 \
  --output_dir artifacts/policy_eval_h10
```

Larger-map example using the same default map-family mix:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt \
  --num_episodes 10 \
  --map_width 200 \
  --map_height 200 \
  --horizon 10 \
  --output_dir artifacts/policy_eval_map20_h10
```

With the current `resolution=0.1`, `--map_width 200 --map_height 200` means a
20m x 20m runtime map. The generator family mix stays the same; corridors and
gaps become wider because the current generators scale those quantities from map size.

No-backward example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt \
  --num_episodes 10 \
  --disable_backward \
  --horizon 10 \
  --output_dir artifacts/policy_eval_no_backward_h10
```

Combined larger-map + no-backward example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt \
  --num_episodes 10 \
  --map_width 200 \
  --map_height 200 \
  --disable_backward \
  --horizon 10 \
  --output_dir artifacts/policy_eval_map20_no_backward_h10
```

### Mode 2: Saved-Episode Replay Override

If `--episode_source PATH` is provided, policy eval replays saved episode **starts** from an existing dataset or episodes directory.

Accepted source layout:

- `<episode_source>/episodes/episode_*.pkl`
- or `episode_source` itself as the `episodes/` directory

Replay mode uses each saved episode only as a scenario definition:

- stored map
- stored path
- stored start pose
- stored goal
- stored dynamic obstacle routes

It does **not** replay the recorded expert actions. The learned controller runs from scratch on the reconstructed scenario.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt \
  --episode_source ./dataset \
  --num_episodes 10 \
  --horizon 10 \
  --output_dir artifacts/policy_eval_dataset_h10
```

Replay-mode notes:

- `--num_episodes` is an optional cap on the number of resolved `episode_*.pkl` files
- if `--num_episodes <= 0`, all resolved saved episodes are used
- `--seed` remains meaningful for simulator mode, but it is not used to choose replay episodes
- `--map_width` and `--map_height` are rejected in replay mode, because replay uses
  saved map geometry and planning occupancy from the episode pickle

### Policy-Eval Outputs

`run_policy_eval.py` writes a self-contained output directory containing:

- `summary.json`
- `episode_results/episode_*.json`
- `decision_logs/episode_*.json`
- `rollout_error_stats/episode_*_rollout_error_stats.json`
- `rollout_error_stats_summary.json`
- optional `episodes/`, `preview/`, `gif/`, `video/`

`summary.json` records the source mode in an `episode_source` block:

- simulator mode:
  - `mode: simulator_sampled`
- replay mode:
  - `mode: saved_episode_replay`
  - `path`
  - `resolved_episode_count`
  - `used_episode_files`

## Policy Comparison

Use `compare_policy_eval_runs.py` to compare two completed policy-eval output directories.

Example:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python compare_policy_eval_runs.py \
  --run_a artifacts/policy_eval_h10_a \
  --run_b artifacts/policy_eval_h10_b \
  --label_a checkpoint_a \
  --label_b checkpoint_b \
  --output_dir artifacts/policy_eval_comparison_h10
```

It reads the two completed policy-eval directories and writes:

- `comparison_summary.json`

## Recommended Order

If you want to evaluate a checkpoint thoroughly:

1. run offline model eval on dataset windows with `eval_structured_world_model.py`
2. run policy eval with `run_policy_eval.py`
3. compare two completed policy-eval runs with `compare_policy_eval_runs.py` if you want checkpoint-to-checkpoint or setting-to-setting comparison

## Action-Space Source Of Truth

For the active `factorized_world_v3` branch, the runtime action space is defined in one place:

- `sim2d/config.py::ACTIONS`

For this active branch:

- policy eval candidate enumeration starts from runtime `ACTIONS`
- offline eval action-sensitivity enumeration uses runtime `ACTIONS`
- runtime `(v, omega)` values come from `ACTIONS`
- `--disable_backward` applies a runtime-only filter on top of `ACTIONS` for policy eval;
  it does not change the base action table or training/data-generation defaults

`evaluation.action_cont_values` remains only as backward-compatible fallback for older or non-active paths.
