# World-Model Version-C Training

This branch keeps the `structured_obstacle_rssm_vc` world-model family and its stage-1 / stage-2 RSSM training structure.

The current practical Version-C path now uses causal world-frame obstacle semantics:

- obstacle inputs: world position, causal world velocity, radius
- obstacle outputs: world-frame delta only
- latent prior/posterior rollout remains intact

For the detailed field and frame semantics, see [dynamic_state_data.md](/home/stavan/s4/rnl/world_model/dynamic_state_data.md).

## Active Version-C model

The active cleaned-up config is:

```text
model_type: structured_obstacle_rssm_vc
model_version: version_c_world_delta_v3
```

The RSSM structure is preserved:

- shared per-obstacle encoder
- deterministic history update
- posterior/prior latent state
- shared decoder for practical motion prediction

The practical decoder target is now world-frame delta only.

## Action and goal interface

Action and goal tensors remain in the public interface to keep churn low.

But the cleaned-up world-frame Version-C path should be interpreted as obstacle-dynamics-first rather than action-driven obstacle prediction:

- action/goal plumbing remains available
- the semantic target is obstacle world motion
- future robot-relative geometry is handled analytically in the controller

## Losses

For `version_c_world_delta_v3`, Stage 1 and Stage 2 both keep:

- `loss_delta`
- `loss_latent`

Stage 2 additionally adds rollout supervision on the prior chain:

- rollout world-delta supervision
- rollout latent consistency

There is no explicit next-velocity head and no velocity consistency loss in the world-frame Version-C path.

## Training stages

Stage 1:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage1.yaml
```

Stage 2:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml
```

### Stage 2 initialization

`configs/structured_world_model_version_c_stage2.yaml` is the intended warm-start
Stage 2 config. It uses:

```yaml
training:
  init_checkpoint: ./artifacts/structured_world_model_version_c_stage1/best.pt
```

That means a fresh Stage 2 launch initializes model weights from the Stage 1
checkpoint before starting rollout training.

`training.init_checkpoint` and `--resume` are different:

- `training.init_checkpoint` starts a new Stage 2 run from an earlier checkpoint
- `--resume` continues an existing Stage 2 run from checkpoints already written
  into that run's own `training.output_dir`

Resume the current run in place:

```bash
cd /home/stavan/s4/rnl/world_model
python train_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml \
  --resume
```

## Evaluation

Evaluation focuses on world-frame delta metrics and rollout delta metrics.

Example:

```bash
cd /home/stavan/s4/rnl/world_model
python eval_structured_world_model.py \
  --config configs/structured_world_model_version_c_stage2.yaml \
  --checkpoint artifacts/structured_world_model_version_c_stage2/best.pt
```

## Compatibility

The world-frame cleanup uses a new explicit model version:

- `version_c_world_delta_v3`

That keeps old `version_c_motion_v2` checkpoints recoverable while preventing them from silently loading into the new semantics.

Older datasets remain usable because the loader can synthesize causal world velocity from:

- snapshot velocity at the first episode step
- previous/current world positions afterward
