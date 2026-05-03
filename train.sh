python3 -m sim2d.generate_data \
  --map_yaml ./maps/dreamer/map.yaml \
  --output_dir ./dataset_dreamer_map_large_spread \
  --num_episodes 20000 \
  --max_steps 450 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report

python3 train_structured_world_model.py \
  --config configs/structured_world_model_dreamer_map_finetune.yaml

python3 run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_dreamer_map_finetune/best.pt \
  --episode_source ./dataset_dreamer_map_large_spread \
  --output_dir artifacts/policy_eval_dreamer_map \
  --num_episodes 100 \
  --max_steps 450 \
  --robot_radius 0.4 \
  --horizon 10 \
  --backward_penalty 10.0 \
  --save_previews \
  --save_videos