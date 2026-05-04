#!/usr/bin/env bash
set -euo pipefail

INITIAL_DATASET="./dataset_quadruped_initial"
ROS_DATASET="./dataset_quadruped_ros_map"
INITIAL_ARTIFACTS="./artifacts/structured_world_model_quadruped_initial"
FINETUNE_ARTIFACTS="./artifacts/structured_world_model_quadruped_ros_finetune"
INITIAL_EVAL="./artifacts/policy_eval_quadruped_initial"
FINETUNE_EVAL="./artifacts/policy_eval_quadruped_ros_finetune"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/world_model_nav_mplconfig_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

rm -rf \
  "${INITIAL_DATASET}" \
  "${ROS_DATASET}" \
  "${INITIAL_ARTIFACTS}" \
  "${FINETUNE_ARTIFACTS}" \
  "${INITIAL_EVAL}" \
  "${FINETUNE_EVAL}"

python3 -m sim2d.generate_data \
  --output_dir "${INITIAL_DATASET}" \
  --num_episodes 3000 \
  --max_steps 250 \
  --map_width 200 \
  --map_height 200 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --dynamic_obstacle_speed_min 0.4 \
  --dynamic_obstacle_speed_max 0.6 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_vx_gain_min 0.85 \
  --robot_command_vx_gain_max 1.10 \
  --robot_command_vy_gain_min 0.85 \
  --robot_command_vy_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_vx_noise_std 0.06 \
  --robot_command_vy_noise_std 0.04 \
  --robot_command_omega_noise_std 0.08 \
  --robot_command_vx_clip_min -0.4 \
  --robot_command_vx_clip_max 0.8 \
  --robot_command_vy_clip_min -0.4 \
  --robot_command_vy_clip_max 0.4 \
  --robot_command_omega_clip_min -0.6 \
  --robot_command_omega_clip_max 0.6

python3 train_structured_world_model.py \
  --config configs/structured_world_model_quadruped_initial.yaml

python3 -m sim2d.generate_data \
  --map_yaml ./maps/dreamer/map.yaml \
  --output_dir "${ROS_DATASET}" \
  --num_episodes 3000 \
  --max_steps 450 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --dynamic_obstacle_radius 0.5 \
  --dynamic_obstacle_speed_min 0.4 \
  --dynamic_obstacle_speed_max 0.6 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_vx_gain_min 0.85 \
  --robot_command_vx_gain_max 1.10 \
  --robot_command_vy_gain_min 0.85 \
  --robot_command_vy_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_vx_noise_std 0.06 \
  --robot_command_vy_noise_std 0.04 \
  --robot_command_omega_noise_std 0.08 \
  --robot_command_vx_clip_min -0.4 \
  --robot_command_vx_clip_max 0.8 \
  --robot_command_vy_clip_min -0.4 \
  --robot_command_vy_clip_max 0.4 \
  --robot_command_omega_clip_min -0.6 \
  --robot_command_omega_clip_max 0.6

python3 train_structured_world_model.py \
  --config configs/structured_world_model_quadruped_ros_finetune.yaml

python3 run_policy_eval.py \
  --checkpoint "${INITIAL_ARTIFACTS}/best.pt" \
  --episode_source "${INITIAL_DATASET}" \
  --output_dir "${INITIAL_EVAL}" \
  --num_episodes 20 \
  --max_steps 250 \
  --robot_radius 0.4 \
  --horizon 10 \
  --backward_penalty 10.0 \
  --save_previews \
  --save_videos \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_vx_gain_min 0.85 \
  --robot_command_vx_gain_max 1.10 \
  --robot_command_vy_gain_min 0.85 \
  --robot_command_vy_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_vx_noise_std 0.06 \
  --robot_command_vy_noise_std 0.04 \
  --robot_command_omega_noise_std 0.08 \
  --robot_command_vx_clip_min -0.4 \
  --robot_command_vx_clip_max 0.8 \
  --robot_command_vy_clip_min -0.4 \
  --robot_command_vy_clip_max 0.4 \
  --robot_command_omega_clip_min -0.6 \
  --robot_command_omega_clip_max 0.6

python3 run_policy_eval.py \
  --checkpoint "${FINETUNE_ARTIFACTS}/best.pt" \
  --episode_source "${ROS_DATASET}" \
  --output_dir "${FINETUNE_EVAL}" \
  --num_episodes 20 \
  --max_steps 450 \
  --robot_radius 0.4 \
  --horizon 10 \
  --backward_penalty 10.0 \
  --save_previews \
  --save_videos \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 2 \
  --robot_command_vx_gain_min 0.85 \
  --robot_command_vx_gain_max 1.10 \
  --robot_command_vy_gain_min 0.85 \
  --robot_command_vy_gain_max 1.10 \
  --robot_command_omega_gain_min 0.80 \
  --robot_command_omega_gain_max 1.20 \
  --robot_command_vx_noise_std 0.06 \
  --robot_command_vy_noise_std 0.04 \
  --robot_command_omega_noise_std 0.08 \
  --robot_command_vx_clip_min -0.4 \
  --robot_command_vx_clip_max 0.8 \
  --robot_command_vy_clip_min -0.4 \
  --robot_command_vy_clip_max 0.4 \
  --robot_command_omega_clip_min -0.6 \
  --robot_command_omega_clip_max 0.6
