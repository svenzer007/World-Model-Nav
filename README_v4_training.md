# Longer-Horizons Practical Training

This README covers the practical recurrent learned-model branch under `/home/stavan/s4/rnl/longer_horizons`.

The current recommended practical branch is the world-frame redesign:

```yaml
model:
  model_type: factorized_world_v3
```

For the detailed state semantics and causal-velocity definition, see [dynamic_state_data.md](/home/stavan/s4/rnl/longer_horizons/dynamic_state_data.md).

## Current practical model

The practical branch keeps the same general recurrent family:

- shared per-obstacle encoder
- recurrent GRU temporal state
- shared per-obstacle prediction head

The new world-frame practical path uses:

- input per obstacle:
  - `x_world`
  - `y_world`
  - `vx_world_causal`
  - `vy_world_causal`
  - `radius`
- output per obstacle:
  - `pred_delta_world`

The world-frame branch intentionally does not keep:

- explicit next-velocity head
- velocity consistency loss
- risk/collision auxiliary heads

## Action conditioning

The new world-frame practical branch does not meaningfully rely on robot action for obstacle prediction.

Implementation detail:

- the public model interface still accepts action tensors so the training/controller plumbing stays stable
- but `factorized_world_v3` uses pooled obstacle state as its recurrent input context rather than action-conditioned obstacle dynamics

That keeps churn low while making the effective obstacle-dynamics path action-independent.

## Losses

For `factorized_world_v3`, training is motion-only:

- one-step world-delta loss
- rollout world-delta loss when rollout training is enabled

Rollout uses:

```text
p_world_next = p_world_prev + pred_delta_world
v_world_next = (p_world_next - p_world_prev) / dt
```

So a causal world velocity state is still carried through rollout, but it is derived from consecutive predicted world positions rather than predicted directly by a second head.

## Stage 1 / Stage 2

Stage 1 one-step-only training:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py --config configs/structured_world_model_stage1.yaml
```

Stage 2 rollout training:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py --config configs/structured_world_model_stage2.yaml
```

The shared rollout-enabled config remains:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python train_structured_world_model.py --config configs/structured_world_model.yaml
```

## Evaluation

Example evaluation command:

```bash
cd /home/stavan/s4/rnl/longer_horizons
python eval_structured_world_model.py \
  --config configs/structured_world_model_stage2.yaml \
  --checkpoint artifacts/structured_world_model_factorized_world_v3_stage2/best.pt
```

## Compatibility

The world-frame redesign uses a new explicit model type:

- `factorized_world_v3`

That keeps old robot-frame factorized checkpoints recoverable and prevents them from silently loading into the new semantics.

## Dataset expectations

The world-frame practical branch expects the loader to provide:

- `obstacle_pos` in world frame
- `obstacle_vel` as causal world velocity
- `radii`
- `delta_world`
- `action_index`, `action_cont`
- `goal` if enabled

Older datasets remain usable because the loader can synthesize causal world velocity from:

- stored snapshot velocity at the first step
- prior world positions afterward
