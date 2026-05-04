#!/usr/bin/env bash
set -euo pipefail

BASE_CHECKPOINT="./artifacts/structured_world_model_quadruped_initial/best.pt"
DATASET="./dataset_quadruped_ros_map_dynobs_r1_continue"
ARTIFACTS="./artifacts/structured_world_model_quadruped_ros_dynobs_r1_finetune_from_initial"
EVAL="./artifacts/policy_eval_quadruped_ros_dynobs_r1_finetune_from_initial"
CONFIG="./configs/structured_world_model_quadruped_ros_dynobs_r1_finetune_from_initial.yaml"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/world_model_nav_mplconfig_${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${DATASET}/episodes" ]]; then
  echo "Missing dataset episodes directory: ${DATASET}/episodes" >&2
  exit 1
fi

rm -rf \
  "${ARTIFACTS}" \
  "${EVAL}"

python3 train_structured_world_model.py \
  --config "${CONFIG}"

python3 run_policy_eval.py \
  --checkpoint "${ARTIFACTS}/best.pt" \
  --episode_source "${DATASET}" \
  --output_dir "${EVAL}" \
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
