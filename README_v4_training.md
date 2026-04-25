# V4 Structured World-Model Training

This README covers the current training workflow under `/home/stavan/s4/rnl/4proj_3horizon`.

The recommended baseline path in this repo is the rollout-horizon-3 setup driven by:

- `train_structured_world_model.py`
- `configs/structured_world_model.yaml`

The currently documented working checkpoint example is:

- `/home/stavan/s4/rnl/4proj_3horizon/artifacts/structured_world_model_rollout_from_epoch_10/epoch_040.pt`

For the dataset format expected by training, see [README_v4_dynamic_state_data.md](./README_v4_dynamic_state_data.md).

## Prerequisites

The training script imports:

- `torch`
- `yaml` via `PyYAML`
- `numpy`
- `tqdm`

## Config

The default training config is:

```bash
/home/stavan/s4/rnl/4proj_3horizon/configs/structured_world_model.yaml
```

Important defaults in that file:

- `data.data_dir: ./dataset`
- `data.sequence_length: 20`
- `model.num_dynamic_obstacles: 4`
- `model.use_goal: true`
- `model.use_lidar: false`
- `rollout.enabled: true`
- `rollout.horizon: 3`

Training expects a dataset root that contains `episodes/episode_*.pkl`, or an `episodes/` directory directly. The loader splits train/validation deterministically by sorted filename order.

## Train

Compile first:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python -m compileall .
```

Run training with the default config:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python train_structured_world_model.py --config configs/structured_world_model.yaml
```

If you want the output directory to be the simpler horizon-3 baseline path, set this in the config before launching:

```yaml
training:
  output_dir: ./artifacts/structured_world_model
```

The current checked-in config uses:

```yaml
training:
  output_dir: ./artifacts/structured_world_model_rollout_from_epoch_10
```

## Resume

Resume from the default latest checkpoint in the configured output directory:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume
```

Resume from a specific full checkpoint:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume \
  --checkpoint /home/stavan/s4/rnl/4proj_3horizon/artifacts/structured_world_model_rollout_from_epoch_10/epoch_040.pt
```

Resume with a fresh optimizer state:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python train_structured_world_model.py \
  --config configs/structured_world_model.yaml \
  --resume \
  --checkpoint /home/stavan/s4/rnl/4proj_3horizon/artifacts/structured_world_model_rollout_from_epoch_10/epoch_040.pt \
  --fresh_optimizer
```

## Initialize A New Run From A Checkpoint

The script also supports starting a new run from pretrained weights through the config:

```yaml
training:
  output_dir: ./artifacts/structured_world_model_new_run
  init_checkpoint: /home/stavan/s4/rnl/4proj_3horizon/artifacts/structured_world_model_rollout_from_epoch_10/epoch_040.pt
  init_load_optimizer_state: false
```

Then launch normally:

```bash
cd /home/stavan/s4/rnl/4proj_3horizon
python train_structured_world_model.py --config configs/structured_world_model.yaml
```

Important behavior:

- `--resume` continues the same run in the current output directory.
- `training.init_checkpoint` starts a brand-new run in a new output directory.
- `--resume` and `training.init_checkpoint` cannot be used together.

## Outputs

Training writes artifacts under `training.output_dir`.

Current output files include:

- `epoch_001.pt`, `epoch_002.pt`, ...
- `epoch_001_weights.pt`, `epoch_002_weights.pt`, ...
- `latest.pt`
- `latest_weights.pt`
- `best.pt`
- `best_weights.pt`
- `history.json`
- `config_snapshot.json`
- `init_source.json` when `training.init_checkpoint` is used

Meaning of the main files:

- `epoch_###.pt`: full checkpoint for that epoch
- `latest.pt`: latest full checkpoint
- `best.pt`: full checkpoint for the best validation loss seen so far
- `*_weights.pt`: weights-only `state_dict` companion file
- `history.json`: per-epoch train/validation metrics
- `config_snapshot.json`: saved config plus train/val dataset counts and collision class weights
- `init_source.json`: provenance for a new run initialized from another checkpoint

Each full checkpoint includes:

- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `config`
- `metrics`
- `best_val_loss`

## Notes

- This README documents the current horizon-3 training path only.
- It does not treat longer-horizon curriculum variants or other branches as the main workflow.
- The example checkpoint path above is used because it exists in this repo today, even though the rollout horizon in the config is still `3`.
