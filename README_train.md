# Dreamer Map Training Commands

```bash
cd /home/john/workspaces/navigation_ws/src/World-Model-Nav
conda activate isaac6
```

```bash
python3 -m sim2d.generate_data \
  --map_yaml ./maps/dreamer/map.yaml \
  --output_dir ./dataset_dreamer_map_spread \
  --num_episodes 20 \
  --max_steps 450 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --dynamic_obstacle_speed 0.5 \
  --no_lidar \
  --num_workers 4
```

```bash
python3 -m sim2d.generate_data \
  --map_yaml ./maps/dreamer/map.yaml \
  --output_dir ./dataset_dreamer_map_large_spread \
  --num_episodes 2000 \
  --max_steps 450 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --dynamic_obstacle_speed 0.5 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 8
```

```bash
python3 train_structured_world_model.py \
  --config configs/structured_world_model_dreamer_map_finetune.yaml
```

```bash
python3 run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_dreamer_map_finetune/best.pt \
  --episode_source ./dataset_dreamer_map_large_spread \
  --output_dir artifacts/policy_eval_dreamer_map \
  --num_episodes 100 \
  --max_steps 450 \
  --robot_radius 0.4 \
  --horizon 10 \
  --backward_penalty 10.0 \
  --save_previews
```

```bash
mkdir -p ./export/world_model_nav_ros2/model
cp artifacts/structured_world_model_dreamer_map_finetune/best.pt \
  ./export/world_model_nav_ros2/model/best.pt
```
