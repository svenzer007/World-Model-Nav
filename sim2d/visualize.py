"""Offline visualization helpers using matplotlib Agg."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:  # pragma: no cover - depends on the local Python environment.
    imageio = None

from .config import VisualizationConfig
from .lidar import beam_angles
from .utils import json_safe, save_json


def save_episode_preview_png(episode: dict[str, object], output_path: Path, vis_config: VisualizationConfig) -> None:
    """Save a static overview plot for one episode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    occupancy = np.asarray(episode["occupancy_grid"])
    path_world = np.asarray(episode["astar_path_world"])
    trajectory = np.asarray(episode["episode_stats"]["robot_trajectory"])
    resolution = float(episode["resolution"])
    origin = np.asarray(episode["origin"], dtype=float)

    fig, ax = plt.subplots(figsize=vis_config.figure_size, dpi=vis_config.dpi)
    _draw_map(ax, occupancy, resolution, origin)
    if len(path_world):
        ax.plot(path_world[:, 0], path_world[:, 1], color="tab:cyan", linewidth=1.5, label="A* path")
    if len(trajectory):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", linewidth=1.8, label="robot")
    for obstacle_id, traj in episode["episode_stats"]["dynamic_trajectories"].items():
        traj_arr = np.asarray(traj)
        if len(traj_arr):
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], linestyle="--", linewidth=1.0, label=obstacle_id)
    start_pose = np.asarray(episode["start_pose"])
    goal = np.asarray(episode["final_goal"])
    ax.scatter(start_pose[0], start_pose[1], c="lime", s=70, marker="o", label="start")
    ax.scatter(goal[0], goal[1], c="gold", s=90, marker="*", label="goal")
    ax.set_title(f"Episode {episode['episode_id']:06d} | {episode['episode_stats']['done_reason']}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_episode_gif(episode: dict[str, object], output_path: Path, vis_config: VisualizationConfig) -> None:
    """Save an animated preview GIF for one episode."""
    if imageio is None:
        raise ModuleNotFoundError("Install imageio to save GIF previews")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = episode["steps"]
    if not steps:
        return

    with imageio.get_writer(output_path, mode="I", fps=vis_config.gif_fps) as writer:
        for step in steps:
            writer.append_data(render_episode_step_frame(episode, step, vis_config))


def save_episode_video(episode: dict[str, object], output_path: Path, vis_config: VisualizationConfig) -> bool:
    """Save an MP4 preview video for one episode if a video backend is available."""
    if imageio is None:
        print(f"Warning: could not save MP4 video to {output_path}: imageio is not installed", flush=True)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = episode["steps"]
    if not steps:
        return False

    writer = None
    try:
        writer = imageio.get_writer(output_path, fps=vis_config.gif_fps)
        for step in steps:
            writer.append_data(render_episode_step_frame(episode, step, vis_config))
        return True
    except Exception as exc:
        print(f"Warning: could not save MP4 video to {output_path}: {exc}", flush=True)
        return False
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                print(f"Warning: could not close MP4 writer for {output_path}: {exc}", flush=True)


def save_episode_html_viewer(episode: dict[str, object], output_dir: Path, vis_config: VisualizationConfig) -> dict[str, str]:
    """Save per-step PNG frames and a local HTML slider viewer."""
    if imageio is None:
        raise ModuleNotFoundError("Install imageio to save HTML viewer frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    step_metadata: list[dict[str, object]] = []
    for frame_index, step in enumerate(episode["steps"]):
        frame_path = frames_dir / f"frame_{frame_index:04d}.png"
        imageio.imwrite(frame_path, render_episode_step_frame(episode, step, vis_config))
        step_metadata.append(_build_viewer_step_metadata(step, frame_path.relative_to(output_dir)))

    html_path = output_dir / "viewer.html"
    _write_episode_viewer_html(episode, html_path, step_metadata)
    return {
        "viewer_html": str(html_path),
        "frames_dir": str(frames_dir),
    }


def render_episode_step_frame(
    episode: dict[str, object],
    step: dict[str, object],
    vis_config: VisualizationConfig,
) -> np.ndarray:
    """Render one episode step to an RGB image array."""
    occupancy = np.asarray(episode["occupancy_grid"])
    path_world = np.asarray(episode["astar_path_world"])
    resolution = float(episode["resolution"])
    origin = np.asarray(episode["origin"], dtype=float)
    lidar_angles = _lidar_angles_from_episode(episode)

    fig, ax = plt.subplots(figsize=vis_config.figure_size, dpi=vis_config.dpi)
    _draw_map(ax, occupancy, resolution, origin)
    if len(path_world):
        ax.plot(path_world[:, 0], path_world[:, 1], color="tab:cyan", linewidth=1.2)

    robot_pose = np.asarray(step["robot_pose"])
    robot_radius = float(episode.get("robot_config", {}).get("radius", 0.25))
    _draw_robot(ax, robot_pose, robot_radius)
    goal = np.asarray(step["selected_subgoal_world"])
    ax.scatter(goal[0], goal[1], c="orange", s=45, marker="x")

    for obstacle_state in step["dynamic_obstacles_t"]:
        _draw_dynamic_obstacle(ax, obstacle_state)

    mode = step.get("controller_debug", {}).get("mode", "")
    lidar_source = "none"
    if vis_config.draw_lidar_in_gif:
        lidar_ranges, lidar_source = _select_lidar_for_rendering(step, vis_config)
        _draw_lidar(ax, robot_pose, lidar_ranges, lidar_angles)

    mode_suffix = f" | {mode}" if mode else ""
    lidar_suffix = f" | {lidar_source}" if lidar_source != "none" else ""
    ax.set_title(
        f"Ep {int(episode['episode_id']):06d} | t={int(step['t'])} | action={step['action_name']} | "
        f"{step['done_reason_t1']}{mode_suffix}{lidar_suffix}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb = frame[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _select_lidar_for_rendering(step: dict[str, object], vis_config: VisualizationConfig) -> tuple[np.ndarray, str]:
    debug = step.get("controller_debug", {})
    if (
        bool(getattr(vis_config, "draw_predicted_lidar", False))
        and debug.get("mode") == "world_model_rollout"
    ):
        predicted_rollout = debug.get("chosen", {}).get("predicted_lidar_rollout", [])
        if predicted_rollout:
            return np.asarray(predicted_rollout[0], dtype=float), "predicted lidar"
    return np.asarray(step["lidar_t"], dtype=float), "true lidar"


def _build_viewer_step_metadata(step: dict[str, object], frame_path: Path) -> dict[str, object]:
    debug = step.get("controller_debug", {})
    return {
        "frame": str(frame_path),
        "step_index": int(step["t"]),
        "action_name": str(step["action_name"]),
        "done": bool(step["done_t1"]),
        "done_reason": str(step["done_reason_t1"]),
        "collision": bool(step["collision_t1"]),
        "controller_mode": str(debug.get("mode", "")),
    }


def _write_episode_viewer_html(
    episode: dict[str, object],
    output_path: Path,
    step_metadata: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not step_metadata:
        output_path.write_text("<!doctype html><html><body><p>No frames available.</p></body></html>", encoding="utf-8")
        return

    frames_js = json.dumps(json_safe(step_metadata), indent=2)
    title = f"Episode {int(episode['episode_id']):06d} | {episode.get('controller', '')}"
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f3; color: #222; }}
    .viewer {{ max-width: 1100px; margin: 0 auto; }}
    img {{ width: 100%; border: 1px solid #bbb; background: white; }}
    input[type="range"] {{ width: 100%; margin: 18px 0; }}
    .metadata {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 18px; margin: 12px 0 18px; }}
    .metadata div {{ background: white; border: 1px solid #ddd; padding: 8px 10px; }}
    .label {{ font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-weight: 700; margin-top: 3px; }}
  </style>
</head>
<body>
  <main class="viewer">
    <h1>{html.escape(title)}</h1>
    <input id="slider" type="range" min="0" max="{len(step_metadata) - 1}" value="0" step="1">
    <div class="metadata">
      <div><div class="label">Step</div><div id="step" class="value"></div></div>
      <div><div class="label">Action</div><div id="action" class="value"></div></div>
      <div><div class="label">Controller Mode</div><div id="mode" class="value"></div></div>
      <div><div class="label">Done</div><div id="done" class="value"></div></div>
      <div><div class="label">Done Reason</div><div id="reason" class="value"></div></div>
      <div><div class="label">Collision</div><div id="collision" class="value"></div></div>
    </div>
    <img id="frame" alt="episode frame">
  </main>
  <script>
    const frames = {frames_js};
    const slider = document.getElementById('slider');
    const frame = document.getElementById('frame');
    const fields = {{
      step: document.getElementById('step'),
      action: document.getElementById('action'),
      mode: document.getElementById('mode'),
      done: document.getElementById('done'),
      reason: document.getElementById('reason'),
      collision: document.getElementById('collision'),
    }};
    function showFrame(index) {{
      const item = frames[index];
      frame.src = item.frame;
      fields.step.textContent = item.step_index;
      fields.action.textContent = item.action_name;
      fields.mode.textContent = item.controller_mode || 'n/a';
      fields.done.textContent = item.done ? 'true' : 'false';
      fields.reason.textContent = item.done_reason;
      fields.collision.textContent = item.collision ? 'true' : 'false';
    }}
    slider.addEventListener('input', () => showFrame(Number(slider.value)));
    document.addEventListener('keydown', (event) => {{
      let value = Number(slider.value);
      if (event.key === 'ArrowRight') value = Math.min(Number(slider.max), value + 1);
      if (event.key === 'ArrowLeft') value = Math.max(0, value - 1);
      slider.value = value;
      showFrame(value);
    }});
    showFrame(0);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def save_timestep_debug_artifacts(
    episode: dict[str, object],
    timestep: int,
    output_dir: Path,
    vis_config: VisualizationConfig | None = None,
) -> dict[str, str]:
    """Generate focused debug artifacts for a single timestep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if vis_config is None:
        vis_config = VisualizationConfig()
    if timestep < 0 or timestep >= len(episode["steps"]):
        raise IndexError(f"Timestep {timestep} out of range for episode {episode['episode_id']}")

    step = episode["steps"][timestep]
    stem = f"episode_{int(episode['episode_id']):06d}_t{int(step['t']):04d}"
    compare_path = output_dir / f"{stem}_compare_maps.png"
    candidate_path = output_dir / f"{stem}_candidate_rollouts.png"
    lidar_path = output_dir / f"{stem}_lidar.png"
    json_path = output_dir / f"{stem}_debug.json"

    _save_compare_maps_debug(episode, step, compare_path, vis_config)
    _save_candidate_rollouts_debug(episode, step, candidate_path, vis_config)
    _save_lidar_debug(episode, step, lidar_path, vis_config)
    save_json(json_path, build_timestep_debug_summary(episode, timestep))

    return {
        "compare_maps_png": str(compare_path),
        "candidate_rollouts_png": str(candidate_path),
        "lidar_png": str(lidar_path),
        "debug_json": str(json_path),
    }


def build_timestep_debug_summary(episode: dict[str, object], timestep: int) -> dict[str, object]:
    """Build a compact JSON-serializable summary for one timestep."""
    step = episode["steps"][timestep]
    chosen_candidate = None
    for candidate in step.get("expert_debug", {}).get("candidates", []):
        if int(candidate.get("action_index", -1)) == int(step["action_index"]):
            chosen_candidate = candidate
            break
    return {
        "episode_id": int(episode["episode_id"]),
        "timestep": int(step["t"]),
        "robot_pose": np.asarray(step["robot_pose"], dtype=float),
        "selected_subgoal_world": np.asarray(step["selected_subgoal_world"], dtype=float),
        "nearest_path_index": int(step["nearest_path_index"]),
        "chosen_action": {
            "action_index": int(step["action_index"]),
            "action_name": str(step["action_name"]),
            "vx": float(step.get("vx", step.get("v", 0.0))),
            "vy": float(step.get("vy", 0.0)),
            "v": float(step.get("vx", step.get("v", 0.0))),
            "omega": float(step["omega"]),
            "wz": float(step.get("wz", step["omega"])),
            "feasible": None if chosen_candidate is None else bool(chosen_candidate.get("feasible", False)),
        },
        "inflation_metadata": {
            "inflation_radius_m": float(episode.get("inflation_radius_m", 0.0)),
            "inflation_radius_cells": int(episode.get("inflation_radius_cells", 0)),
        },
        "candidates": step.get("expert_debug", {}).get("candidates", []),
        "chosen_candidate": chosen_candidate,
    }


def _draw_map(ax: plt.Axes, occupancy: np.ndarray, resolution: float, origin: Sequence[float]) -> None:
    extent = [
        origin[0],
        origin[0] + occupancy.shape[1] * resolution,
        origin[1],
        origin[1] + occupancy.shape[0] * resolution,
    ]
    ax.imshow(occupancy, cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")


def _draw_robot(ax: plt.Axes, robot_pose: np.ndarray, robot_radius: float) -> None:
    circle = plt.Circle((robot_pose[0], robot_pose[1]), radius=robot_radius, color="tab:red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    heading_length = max(0.5, float(robot_radius) * 1.4)
    heading_end = robot_pose[:2] + heading_length * np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])], dtype=float)
    ax.plot([robot_pose[0], heading_end[0]], [robot_pose[1], heading_end[1]], color="tab:red", linewidth=1.5)


def _draw_dynamic_obstacle(ax: plt.Axes, obstacle_state: dict[str, object]) -> None:
    pos = np.asarray(obstacle_state["position"], dtype=float)
    radius = float(obstacle_state["radius"])
    circle = plt.Circle((pos[0], pos[1]), radius=radius, color="tab:blue", alpha=0.45)
    ax.add_patch(circle)


def _draw_lidar(ax: plt.Axes, robot_pose: np.ndarray, lidar_ranges: np.ndarray, lidar_angles: np.ndarray) -> None:
    for ray_range, rel_angle in zip(lidar_ranges[::6], lidar_angles[::6]):
        angle = robot_pose[2] + rel_angle
        end = robot_pose[:2] + float(ray_range) * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        ax.plot([robot_pose[0], end[0]], [robot_pose[1], end[1]], color="tab:green", linewidth=0.35, alpha=0.6)


def _draw_lidar_full(ax: plt.Axes, robot_pose: np.ndarray, lidar_ranges: np.ndarray, lidar_angles: np.ndarray) -> None:
    for ray_range, rel_angle in zip(lidar_ranges, lidar_angles):
        angle = robot_pose[2] + rel_angle
        end = robot_pose[:2] + float(ray_range) * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        ax.plot([robot_pose[0], end[0]], [robot_pose[1], end[1]], color="tab:green", linewidth=0.25, alpha=0.45)


def _lidar_angles_from_episode(episode: dict[str, object]) -> np.ndarray:
    lidar_cfg = episode["lidar_config"]
    return beam_angles(
        type(
            "TmpCfg",
            (),
            {
                "num_beams": int(lidar_cfg["num_beams"]),
                "angle_min": float(lidar_cfg["angle_min"]),
                "angle_max": float(lidar_cfg["angle_max"]),
            },
        )()
    )


def _save_compare_maps_debug(
    episode: dict[str, object],
    step: dict[str, object],
    output_path: Path,
    vis_config: VisualizationConfig,
) -> None:
    """Compare original and planning occupancy maps for a given timestep."""
    occupancy = np.asarray(episode["occupancy_grid"])
    planning_occupancy = np.asarray(episode.get("planning_occupancy", episode["occupancy_grid"]))
    path_world = np.asarray(episode["astar_path_world"])
    resolution = float(episode["resolution"])
    origin = np.asarray(episode["origin"], dtype=float)
    robot_pose = np.asarray(step["robot_pose"], dtype=float)
    subgoal = np.asarray(step["selected_subgoal_world"], dtype=float)
    robot_radius = float(episode.get("robot_config", {}).get("radius", 0.25))
    inflation_m = float(episode.get("inflation_radius_m", 0.0))
    inflation_cells = int(episode.get("inflation_radius_cells", 0))

    fig, axes = plt.subplots(1, 2, figsize=(vis_config.figure_size[0] * 1.8, vis_config.figure_size[1]), dpi=vis_config.dpi)
    for axis, grid, panel_title in zip(axes, [occupancy, planning_occupancy], ["Original occupancy", "Planning occupancy"]):
        _draw_map(axis, grid, resolution, origin)
        _overlay_path_robot_goal(axis, path_world, robot_pose, subgoal, robot_radius)
        axis.set_title(panel_title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")

    fig.suptitle(
        f"Episode {int(episode['episode_id']):06d} | t={int(step['t']):04d} | inflation={inflation_m:.2f} m ({inflation_cells} cells)"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_candidate_rollouts_debug(
    episode: dict[str, object],
    step: dict[str, object],
    output_path: Path,
    vis_config: VisualizationConfig,
) -> None:
    """Overlay expert rollout candidates on the occupancy map."""
    occupancy = np.asarray(episode["occupancy_grid"])
    planning_occupancy = np.asarray(episode.get("planning_occupancy", episode["occupancy_grid"]))
    path_world = np.asarray(episode["astar_path_world"])
    resolution = float(episode["resolution"])
    origin = np.asarray(episode["origin"], dtype=float)
    robot_pose = np.asarray(step["robot_pose"], dtype=float)
    subgoal = np.asarray(step["selected_subgoal_world"], dtype=float)
    robot_radius = float(episode.get("robot_config", {}).get("radius", 0.25))
    candidates = step.get("expert_debug", {}).get("candidates", [])

    fig, ax = plt.subplots(figsize=(vis_config.figure_size[0] * 1.7, vis_config.figure_size[1]), dpi=vis_config.dpi)
    _draw_map(ax, occupancy, resolution, origin)
    _overlay_path_robot_goal(ax, path_world, robot_pose, subgoal, robot_radius)
    _draw_planning_contour(ax, planning_occupancy, resolution, origin)

    table_lines = ["action        feas  coll              step  score      min_clr  true1"]
    for candidate in candidates:
        feasible = bool(candidate.get("feasible", False))
        color = "tab:green" if feasible else "tab:red"
        rollout_positions = np.asarray(candidate.get("rollout_positions", []), dtype=float)
        if rollout_positions.size:
            ax.plot(rollout_positions[:, 0], rollout_positions[:, 1], color=color, linewidth=1.4, alpha=0.9)
            endpoint = rollout_positions[-1]
            ax.scatter(endpoint[0], endpoint[1], color=color, s=22)
            ax.text(endpoint[0], endpoint[1], str(candidate.get("action_name", "")), color=color, fontsize=7)
        collision_position = candidate.get("collision_position", None)
        if collision_position is not None:
            collision_xy = np.asarray(collision_position, dtype=float)
            ax.scatter(collision_xy[0], collision_xy[1], marker="x", c=color, s=48, linewidths=1.3)
        table_lines.append(
            f"{str(candidate.get('action_name', '')):<13}"
            f"{'Y' if feasible else 'N':<6}"
            f"{str(candidate.get('collision_reason', '')):<18}"
            f"{str(candidate.get('collision_step_index', '')):<6}"
            f"{float(candidate.get('total_score', 0.0)):<10.2f}"
            f"{float(candidate.get('min_clearance', 0.0)):<9.2f}"
            f"{'N' if candidate.get('true_map_one_step_collision', False) else 'Y'}"
        )

    ax.text(
        1.02,
        0.98,
        "\n".join(table_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    ax.set_title(f"Episode {int(episode['episode_id']):06d} | t={int(step['t']):04d} candidate rollouts")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_lidar_debug(
    episode: dict[str, object],
    step: dict[str, object],
    output_path: Path,
    vis_config: VisualizationConfig,
) -> None:
    """Save a LiDAR ray debug plot for one timestep."""
    occupancy = np.asarray(episode["occupancy_grid"])
    path_world = np.asarray(episode["astar_path_world"])
    resolution = float(episode["resolution"])
    origin = np.asarray(episode["origin"], dtype=float)
    robot_pose = np.asarray(step["robot_pose"], dtype=float)
    subgoal = np.asarray(step["selected_subgoal_world"], dtype=float)
    robot_radius = float(episode.get("robot_config", {}).get("radius", 0.25))
    lidar_angles = _lidar_angles_from_episode(episode)

    fig, ax = plt.subplots(figsize=vis_config.figure_size, dpi=vis_config.dpi)
    _draw_map(ax, occupancy, resolution, origin)
    _overlay_path_robot_goal(ax, path_world, robot_pose, subgoal, robot_radius)
    _draw_lidar_full(ax, robot_pose, np.asarray(step["lidar_t"], dtype=float), lidar_angles)
    ax.set_title(f"Episode {int(episode['episode_id']):06d} | t={int(step['t']):04d} lidar")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _overlay_path_robot_goal(
    ax: plt.Axes,
    path_world: np.ndarray,
    robot_pose: np.ndarray,
    subgoal: np.ndarray,
    robot_radius: float,
) -> None:
    if len(path_world):
        ax.plot(path_world[:, 0], path_world[:, 1], color="tab:cyan", linewidth=1.2)
    _draw_robot_with_radius(ax, robot_pose, robot_radius)
    ax.scatter(subgoal[0], subgoal[1], c="orange", marker="x", s=42)


def _draw_robot_with_radius(ax: plt.Axes, robot_pose: np.ndarray, robot_radius: float) -> None:
    circle = plt.Circle((robot_pose[0], robot_pose[1]), radius=robot_radius, color="tab:red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    heading_end = robot_pose[:2] + 0.55 * np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])], dtype=float)
    ax.plot([robot_pose[0], heading_end[0]], [robot_pose[1], heading_end[1]], color="tab:red", linewidth=1.4)


def _draw_planning_contour(ax: plt.Axes, planning_occupancy: np.ndarray, resolution: float, origin: Sequence[float]) -> None:
    extent = [
        origin[0],
        origin[0] + planning_occupancy.shape[1] * resolution,
        origin[1],
        origin[1] + planning_occupancy.shape[0] * resolution,
    ]
    ax.contour(planning_occupancy, levels=[0.5], origin="lower", extent=extent, colors=["tab:orange"], linewidths=0.6)
