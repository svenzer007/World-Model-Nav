#!/usr/bin/env bash
set -euo pipefail

python3 -m sim2d.generate_data \
  --map_yaml ./maps/dreamer/map.yaml \
  --output_dir ./dataset_dreamer_map_large_spread_robust \
  --num_episodes 20000 \
  --max_steps 450 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_v_gain_min 0.85 \
  --robot_command_v_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_v_noise_std 0.08 \
  --robot_command_omega_noise_std 0.15 \
  --robot_command_v_clip_min -0.8 \
  --robot_command_v_clip_max 1.0 \
  --robot_command_omega_clip_min -1.5 \
  --robot_command_omega_clip_max 1.5

python3 train_structured_world_model.py \
  --config configs/structured_world_model_dreamer_map_robust.yaml

python3 run_policy_eval.py \
  --checkpoint artifacts/structured_world_model_dreamer_map_robust/best.pt \
  --episode_source ./dataset_dreamer_map_large_spread_robust \
  --output_dir artifacts/policy_eval_dreamer_map_robust \
  --num_episodes 100 \
  --max_steps 450 \
  --robot_radius 0.4 \
  --horizon 10 \
  --backward_penalty 10.0 \
  --save_previews \
  --save_videos \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_v_gain_min 0.85 \
  --robot_command_v_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_v_noise_std 0.08 \
  --robot_command_omega_noise_std 0.15 \
  --robot_command_v_clip_min -0.8 \
  --robot_command_v_clip_max 1.0 \
  --robot_command_omega_clip_min -1.5 \
  --robot_command_omega_clip_max 1.5
