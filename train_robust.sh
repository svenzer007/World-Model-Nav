#!/usr/bin/env bash
set -euo pipefail

INITIAL_DATASET="./dataset_quadruped_initial"
ROS_DATASET="./dataset_quadruped_ros_map"
INITIAL_ARTIFACTS="./artifacts/structured_world_model_quadruped_initial"
FINETUNE_ARTIFACTS="./artifacts/structured_world_model_quadruped_ros_finetune"
INITIAL_EVAL="./artifacts/policy_eval_quadruped_initial"
FINETUNE_EVAL="./artifacts/policy_eval_quadruped_ros_finetune"

ROBOT_DT="0.05"
EXPERT_HORIZON_SECONDS="3.0"
EXPERT_GOAL_PROGRESS_WEIGHT="1.8"
EXPERT_HEADING_WEIGHT="0.6"
EXPERT_CLEARANCE_WEIGHT="3.0"
EXPERT_PATH_ADHERENCE_WEIGHT="1.0"
EXPERT_VELOCITY_WEIGHT="0.3"
EXPERT_MIN_CLEARANCE_CLIP="0.05"

ROS_MAX_ROBOT_PATH_DEVIATION="4.0"
ROS_MAX_ROBOT_TRAJECTORY_PATH_RATIO="2.5"

POLICY_EVAL_PROGRESS_WEIGHT="1.5"
POLICY_EVAL_HEADING_WEIGHT="0.5"
POLICY_EVAL_PATH_WEIGHT="0.5"
POLICY_EVAL_MIN_CLEARANCE_CLIP="0.02"
POLICY_EVAL_HORIZON_STEPS="60"

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
  --max_steps 500 \
  --map_width 200 \
  --map_height 200 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --robot_dt "${ROBOT_DT}" \
  --dynamic_obstacle_radius 1.0 \
  --dynamic_obstacle_speed_min 0.4 \
  --dynamic_obstacle_speed_max 0.6 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report \
  --expert_horizon_seconds "${EXPERT_HORIZON_SECONDS}" \
  --expert_goal_progress_weight "${EXPERT_GOAL_PROGRESS_WEIGHT}" \
  --expert_heading_weight "${EXPERT_HEADING_WEIGHT}" \
  --expert_clearance_weight "${EXPERT_CLEARANCE_WEIGHT}" \
  --expert_path_adherence_weight "${EXPERT_PATH_ADHERENCE_WEIGHT}" \
  --expert_velocity_weight "${EXPERT_VELOCITY_WEIGHT}" \
  --expert_min_clearance_clip "${EXPERT_MIN_CLEARANCE_CLIP}" \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 4 \
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
  --max_steps 900 \
  --num_dynamic_obstacles 4 \
  --robot_radius 0.4 \
  --robot_dt "${ROBOT_DT}" \
  --dynamic_obstacle_radius 1.0 \
  --dynamic_obstacle_speed_min 0.4 \
  --dynamic_obstacle_speed_max 0.6 \
  --no_lidar \
  --preview_every 50 \
  --num_workers 256 \
  --skip_quality_report \
  --max_robot_path_deviation "${ROS_MAX_ROBOT_PATH_DEVIATION}" \
  --max_robot_trajectory_path_ratio "${ROS_MAX_ROBOT_TRAJECTORY_PATH_RATIO}" \
  --expert_horizon_seconds "${EXPERT_HORIZON_SECONDS}" \
  --expert_goal_progress_weight "${EXPERT_GOAL_PROGRESS_WEIGHT}" \
  --expert_heading_weight "${EXPERT_HEADING_WEIGHT}" \
  --expert_clearance_weight "${EXPERT_CLEARANCE_WEIGHT}" \
  --expert_path_adherence_weight "${EXPERT_PATH_ADHERENCE_WEIGHT}" \
  --expert_velocity_weight "${EXPERT_VELOCITY_WEIGHT}" \
  --expert_min_clearance_clip "${EXPERT_MIN_CLEARANCE_CLIP}" \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 4 \
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
  --max_steps 500 \
  --robot_radius 0.4 \
  --robot_dt "${ROBOT_DT}" \
  --horizon "${POLICY_EVAL_HORIZON_STEPS}" \
  --backward_penalty 10.0 \
  --w_progress "${POLICY_EVAL_PROGRESS_WEIGHT}" \
  --w_heading "${POLICY_EVAL_HEADING_WEIGHT}" \
  --w_path "${POLICY_EVAL_PATH_WEIGHT}" \
  --min_clearance_clip "${POLICY_EVAL_MIN_CLEARANCE_CLIP}" \
  --save_previews \
  --save_videos \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 4 \
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
  --max_steps 900 \
  --robot_radius 0.4 \
  --robot_dt "${ROBOT_DT}" \
  --horizon "${POLICY_EVAL_HORIZON_STEPS}" \
  --backward_penalty 10.0 \
  --w_progress "${POLICY_EVAL_PROGRESS_WEIGHT}" \
  --w_heading "${POLICY_EVAL_HEADING_WEIGHT}" \
  --w_path "${POLICY_EVAL_PATH_WEIGHT}" \
  --min_clearance_clip "${POLICY_EVAL_MIN_CLEARANCE_CLIP}" \
  --save_previews \
  --save_videos \
  --robot_command_randomization \
  --robot_command_delay_min_steps 0 \
  --robot_command_delay_max_steps 4 \
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
