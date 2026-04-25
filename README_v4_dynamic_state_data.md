# V4 Dataset Requirements

This README is the shared dataset-format reference for:

- training
- checkpoint evaluation
- policy-comparison-adjacent debugging

The training and evaluation scripts in this repo consume episode pickle files and turn them into contiguous fixed-length sequence windows. The policy comparison runner does not read this dataset directly, but the learned checkpoint it uses was trained against this schema.

## Expected Directory Layout

The loader accepts either:

- a dataset root containing `episodes/`
- or the `episodes/` directory itself

Expected file layout:

```text
<data_dir>/
  episodes/
    episode_000000.pkl
    episode_000001.pkl
    ...
```

The filename pattern must match:

```text
episode_*.pkl
```

This is enforced by `list_episode_paths(...)` in `data/sequence_dataset.py`.

## Split Behavior

Train/validation splitting is deterministic and based on sorted filename order:

- `split_episode_paths(paths, train_fraction=0.9)`

That means collaborators should not expect a random split unless they rename or reshuffle the episode files externally.

## Sequence Window Construction

`StructuredSequenceDataset` builds contiguous windows from each episode:

- default `sequence_length: 20`
- default `stride: 1`

Episodes shorter than `sequence_length` are skipped.

Each dataset sample corresponds to one contiguous time window and returns tensors such as:

- `obstacle_pos`
- `obstacle_vel`
- `radii`
- `action_index`
- `action_cont`
- `delta_rel`
- `risk`
- `collision_next`
- `collision_within_3`
- `collision_within_5`
- `valid_mask`
- `dt`
- `goal` when enabled
- `lidar` when enabled

## Required Per-Step Fields

The current training/evaluation pipeline reads these fields from each episode step:

- `dynamic_obstacles_robot_t`
- `dynamic_obstacles_robot_t1`
- `dynamic_obstacle_rel_vel_robot_t`
- `dynamic_obstacle_radii`
- `action_index`
- `v_t`
- `omega_t`
- `dynamic_risk_t1`
- `dynamic_collision_t1`
- `dynamic_collision_within_3`
- `dynamic_collision_within_5`

Additional fields used conditionally:

- `goal_t` when `model.use_goal: true`
- `lidar_t` when `model.use_lidar: true`

Time step handling:

- `dt` is taken from the step if present
- otherwise it falls back to `episode["robot_config"]["dt"]`
- otherwise it falls back again to `0.1`

## Shapes And Meaning

For the current model path, the important structured dynamic-obstacle fields are:

- `dynamic_obstacles_robot_t`: `(N_dyn, 2)` robot-relative obstacle positions at time `t`
- `dynamic_obstacles_robot_t1`: `(N_dyn, 2)` robot-relative obstacle positions at time `t+1`
- `dynamic_obstacle_rel_vel_robot_t`: `(N_dyn, 2)` robot-relative obstacle velocities used as the model velocity input
- `dynamic_obstacle_radii`: `(N_dyn,)` obstacle radii, constant across an episode

The dataset code derives:

- `delta_rel = dynamic_obstacles_robot_t1 - dynamic_obstacles_robot_t`
- `action_cont = [v_t, omega_t]`

Risk/collision targets are scalar per step:

- `dynamic_risk_t1`
- `dynamic_collision_t1`
- `dynamic_collision_within_3`
- `dynamic_collision_within_5`

## Fixed Obstacle Count Assumption

The current config uses:

```yaml
model:
  num_dynamic_obstacles: 4
```

So the documented working path assumes a fixed four-obstacle representation. Collaborator-generated datasets should match that expectation unless they also change the model/config accordingly.

## Goal And LiDAR Inputs

The checked-in model config currently uses:

```yaml
model:
  use_goal: true
  use_lidar: false
```

So for the default workflow:

- `goal_t` is required
- `lidar_t` is not required

If `use_lidar` is turned on in the config, each step must also contain `lidar_t`.

## Generation

Generate a dataset with the current simulator pipeline:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python -m sim2d.generate_data \
  --num_episodes 20 \
  --max_steps 250 \
  --output_dir ./output_v4_dynamic_state \
  --seed 0
```

Smoke test:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python -m sim2d.generate_data \
  --num_episodes 2 \
  --max_steps 40 \
  --output_dir /tmp/rnl_v4_smoke \
  --seed 0
```

Inspect the generated episodes:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python -m sim2d.inspect_episodes \
  /tmp/rnl_v4_smoke \
  --report \
  --report_path /tmp/rnl_v4_smoke/dataset_quality_report.json
```

## Notes For Training And Evaluation

- Training and evaluation both consume episode pickles, not pre-baked sequence files.
- Evaluation runs on the validation split derived from the same deterministic filename-based split logic as training.
- If a collaborator points `data.data_dir` at the wrong directory level and there is no `episodes/episode_*.pkl`, loading will fail immediately.

For commands, outputs, and workflow details:

- training: [README_v4_training.md](./README_v4_training.md)
- evaluation: [README_v4_evaluation.md](./README_v4_evaluation.md)
- policy comparison: [README_v4_policy_comparison.md](./README_v4_policy_comparison.md)
