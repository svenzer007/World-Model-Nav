"""Deterministic factorized RSSM-style obstacle dynamics model for Version C."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


VERSION_C_MOTION_MODEL_VERSION = "version_c_motion_v2"
VERSION_C_WORLD_DELTA_MODEL_VERSION = "version_c_world_delta_v3"
SUPPORTED_VERSION_C_MODEL_VERSIONS = {
    VERSION_C_MOTION_MODEL_VERSION,
    VERSION_C_WORLD_DELTA_MODEL_VERSION,
}


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.extend(
        [
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        ]
    )
    return nn.Sequential(*layers)


@dataclass
class StructuredObstacleRSSMConfig:
    model_type: str = "structured_obstacle_rssm_vc"
    model_version: str = VERSION_C_WORLD_DELTA_MODEL_VERSION
    num_dynamic_obstacles: int = 4
    obstacle_feature_dim: int = 5
    obstacle_embed_dim: int = 32
    action_vocab_size: int = 6
    action_embed_dim: int = 16
    action_cont_dim: int = 2
    goal_dim: int = 5
    use_goal: bool = True
    use_lidar: bool = False
    goal_embed_dim: int = 32
    hidden_dim: int = 128
    latent_dim: int = 64
    head_hidden_dim: int = 128
    dropout: float = 0.0
    delta_scale: float = 0.05
    velocity_scale: float = 1.0


class StructuredObstacleRSSMModel(nn.Module):
    """Shared per-obstacle deterministic RSSM-inspired latent dynamics model."""

    def __init__(self, config: StructuredObstacleRSSMConfig):
        super().__init__()
        if bool(config.use_lidar):
            raise ValueError("StructuredObstacleRSSMModel does not support use_lidar=true")
        if str(config.model_version) not in SUPPORTED_VERSION_C_MODEL_VERSIONS:
            raise ValueError(
                f"Unsupported model_version={config.model_version!r}; "
                f"expected one of {sorted(SUPPORTED_VERSION_C_MODEL_VERSIONS)}"
            )
        self.config = config
        self.delta_scale = float(config.delta_scale)
        self.velocity_scale = float(config.velocity_scale)
        self.num_dynamic_obstacles = int(config.num_dynamic_obstacles)
        self.hidden_dim = int(config.hidden_dim)
        self.latent_dim = int(config.latent_dim)
        self.world_delta_only = str(config.model_version) == VERSION_C_WORLD_DELTA_MODEL_VERSION

        self.obstacle_encoder = _mlp(
            input_dim=int(config.obstacle_feature_dim),
            hidden_dim=int(config.obstacle_embed_dim),
            output_dim=int(config.obstacle_embed_dim),
            dropout=float(config.dropout),
        )
        self.action_embedding = nn.Embedding(int(config.action_vocab_size), int(config.action_embed_dim))

        context_dim = int(config.action_embed_dim) + int(config.action_cont_dim)
        self.goal_encoder: nn.Module | None = None
        if bool(config.use_goal):
            self.goal_encoder = _mlp(
                input_dim=int(config.goal_dim),
                hidden_dim=int(config.goal_embed_dim),
                output_dim=int(config.goal_embed_dim),
                dropout=float(config.dropout),
            )
            context_dim += int(config.goal_embed_dim)
        self.context_dim = context_dim

        self.history_cell = nn.GRUCell(
            input_size=int(config.latent_dim) + context_dim,
            hidden_size=int(config.hidden_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(int(config.hidden_dim) + int(config.obstacle_embed_dim), int(config.head_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(config.head_hidden_dim), int(config.latent_dim)),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(int(config.hidden_dim), int(config.head_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(config.head_hidden_dim), int(config.latent_dim)),
        )
        self.decoder_body = _mlp(
            input_dim=int(config.hidden_dim) + int(config.latent_dim) + context_dim,
            hidden_dim=int(config.head_hidden_dim),
            output_dim=int(config.head_hidden_dim),
            dropout=float(config.dropout),
        )
        self.delta_head = nn.Linear(int(config.head_hidden_dim), 2)
        self.next_velocity_head: nn.Module | None = None
        if not self.world_delta_only:
            self.next_velocity_head = nn.Linear(int(config.head_hidden_dim), 2)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "history": torch.zeros(
                batch_size,
                self.num_dynamic_obstacles,
                self.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            "latent": torch.zeros(
                batch_size,
                self.num_dynamic_obstacles,
                self.latent_dim,
                device=device,
                dtype=dtype,
            ),
        }

    def forward_sequence(
        self,
        obstacle_pos: torch.Tensor,
        obstacle_vel: torch.Tensor,
        radii: torch.Tensor,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None = None,
        lidar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lidar is not None:
            raise ValueError("StructuredObstacleRSSMModel does not support lidar inputs")
        batch_size, seq_len, n_dyn, _ = obstacle_pos.shape
        if n_dyn != self.num_dynamic_obstacles:
            raise ValueError(f"Expected {self.num_dynamic_obstacles} obstacles, got {n_dyn}")
        state = self.initial_state(batch_size, device=obstacle_pos.device, dtype=obstacle_pos.dtype)
        prev_context = self.zero_context(batch_size, device=obstacle_pos.device, dtype=obstacle_pos.dtype)
        outputs: dict[str, list[torch.Tensor]] = {}
        for step_index in range(seq_len):
            # RSSM order: update history from previous latent/action, then form
            # posterior/prior for the current step, then decode next-step targets.
            step_outputs, state = self.forward_step(
                obstacle_pos=obstacle_pos[:, step_index],
                obstacle_vel=obstacle_vel[:, step_index],
                radii=radii,
                action_index=action_index[:, step_index],
                action_cont=action_cont[:, step_index],
                goal=goal[:, step_index] if goal is not None else None,
                lidar=None,
                hidden=state,
                prev_context=prev_context,
                use_posterior=True,
            )
            for key, value in step_outputs.items():
                outputs.setdefault(key, []).append(value)
            prev_context = self.encode_context(
                action_index=action_index[:, step_index],
                action_cont=action_cont[:, step_index],
                goal=goal[:, step_index] if goal is not None else None,
            )
        return {key: torch.stack(values, dim=1) for key, values in outputs.items()}

    def forward_step(
        self,
        obstacle_pos: torch.Tensor | None,
        obstacle_vel: torch.Tensor | None,
        radii: torch.Tensor,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None = None,
        lidar: torch.Tensor | None = None,
        hidden: dict[str, torch.Tensor] | None = None,
        *,
        prev_context: torch.Tensor | None = None,
        use_posterior: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if lidar is not None:
            raise ValueError("StructuredObstacleRSSMModel does not support lidar inputs")
        if use_posterior and (obstacle_pos is None or obstacle_vel is None):
            raise ValueError("Posterior inference requires obstacle_pos and obstacle_vel")
        batch_size = action_index.shape[0]
        dtype = radii.dtype
        if hidden is None:
            hidden = self.initial_state(batch_size, device=radii.device, dtype=dtype)
        if prev_context is None:
            prev_context = self.zero_context(batch_size, device=radii.device, dtype=dtype)

        history = self._history_update(hidden["history"], hidden["latent"], prev_context)
        prior_latent = self._prior_from_history(history)
        posterior_latent = prior_latent
        if use_posterior:
            posterior_latent = self._posterior_from_observation(obstacle_pos, obstacle_vel, radii, history)

        current_context = self.encode_context(
            action_index=action_index,
            action_cont=action_cont,
            goal=goal,
        )
        outputs = self._decode(
            history=history,
            latent=posterior_latent,
            context=current_context,
        )
        outputs["posterior_latent"] = posterior_latent
        outputs["prior_latent"] = prior_latent
        next_state = {
            "history": history,
            "latent": posterior_latent if use_posterior else prior_latent,
        }
        return outputs, next_state

    def encode_context(
        self,
        *,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._encode_context(
            action_index=action_index,
            action_cont=action_cont,
            goal=goal,
        )

    def zero_context(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._zero_context(batch_size, device, dtype)

    def _encode_context(
        self,
        *,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.world_delta_only:
            return torch.zeros(action_index.shape[0], self.context_dim, device=action_index.device, dtype=action_cont.dtype)
        pieces = [self.action_embedding(action_index.long()), action_cont]
        if self.goal_encoder is not None:
            if goal is None:
                raise ValueError("Model was configured with use_goal=true but goal=None was provided")
            pieces.append(self.goal_encoder(goal))
        return torch.cat(pieces, dim=-1)

    def _zero_context(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.context_dim, device=device, dtype=dtype)

    def _history_update(
        self,
        history: torch.Tensor,
        previous_latent: torch.Tensor,
        previous_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_dyn, _ = history.shape
        context_expanded = previous_context[:, None, :].expand(batch_size, n_dyn, -1)
        cell_input = torch.cat([previous_latent, context_expanded], dim=-1)
        updated = self.history_cell(
            cell_input.reshape(batch_size * n_dyn, -1),
            history.reshape(batch_size * n_dyn, -1),
        )
        return updated.reshape(batch_size, n_dyn, self.hidden_dim)

    def _posterior_from_observation(
        self,
        obstacle_pos: torch.Tensor,
        obstacle_vel: torch.Tensor,
        radii: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_dyn, _ = obstacle_pos.shape
        radii_expanded = radii[:, :, None]
        obstacle_features = torch.cat([obstacle_pos, obstacle_vel, radii_expanded], dim=-1)
        obstacle_embed = self.obstacle_encoder(obstacle_features.reshape(batch_size * n_dyn, -1)).reshape(
            batch_size, n_dyn, -1
        )
        posterior_input = torch.cat([obstacle_embed, history], dim=-1)
        posterior = self.posterior_net(posterior_input.reshape(batch_size * n_dyn, -1))
        return posterior.reshape(batch_size, n_dyn, self.latent_dim)

    def _prior_from_history(self, history: torch.Tensor) -> torch.Tensor:
        batch_size, n_dyn, _ = history.shape
        prior = self.prior_net(history.reshape(batch_size * n_dyn, -1))
        return prior.reshape(batch_size, n_dyn, self.latent_dim)

    def _decode(
        self,
        *,
        history: torch.Tensor,
        latent: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, n_dyn, _ = history.shape
        context_expanded = context[:, None, :].expand(batch_size, n_dyn, -1)
        decoder_input = torch.cat([history, latent, context_expanded], dim=-1)
        body = self.decoder_body(decoder_input.reshape(batch_size * n_dyn, -1)).reshape(batch_size, n_dyn, -1)
        pred_delta_scaled = self.delta_head(body)
        pred_delta = pred_delta_scaled * self.delta_scale
        outputs = {
            "pred_delta_rel_scaled": pred_delta_scaled,
            "pred_delta_rel": pred_delta,
        }
        if self.world_delta_only:
            outputs["pred_delta_world_scaled"] = pred_delta_scaled
            outputs["pred_delta_world"] = pred_delta
            return outputs
        if self.next_velocity_head is None:
            raise RuntimeError("next_velocity_head is required for non-world-delta Version C models")
        pred_next_rel_vel_scaled = self.next_velocity_head(body)
        outputs.update(
            {
                "pred_next_rel_vel_scaled": pred_next_rel_vel_scaled,
                "pred_next_rel_vel": pred_next_rel_vel_scaled * self.velocity_scale,
            }
        )
        return outputs
