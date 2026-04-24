"""Simple GRU structured dynamics model for V4 obstacle-state prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class StructuredDynamicsConfig:
    num_dynamic_obstacles: int = 4
    obstacle_feature_dim: int = 5
    obstacle_embed_dim: int = 32
    action_vocab_size: int = 6
    action_embed_dim: int = 16
    action_cont_dim: int = 2
    goal_dim: int = 5
    lidar_dim: int = 241
    use_goal: bool = True
    use_lidar: bool = False
    goal_embed_dim: int = 32
    lidar_embed_dim: int = 64
    gru_hidden_dim: int = 128
    gru_layers: int = 1
    head_hidden_dim: int = 128
    dropout: float = 0.0
    delta_scale: float = 0.05


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


class StructuredDynamicsModel(nn.Module):
    """Predict robot-relative obstacle deltas and dynamic-risk labels."""

    def __init__(self, config: StructuredDynamicsConfig):
        super().__init__()
        self.config = config
        self.delta_scale = float(config.delta_scale)
        self.num_dynamic_obstacles = int(config.num_dynamic_obstacles)

        self.obstacle_encoder = _mlp(
            input_dim=int(config.obstacle_feature_dim),
            hidden_dim=int(config.obstacle_embed_dim),
            output_dim=int(config.obstacle_embed_dim),
            dropout=float(config.dropout),
        )
        self.action_embedding = nn.Embedding(int(config.action_vocab_size), int(config.action_embed_dim))

        context_dim = int(config.num_dynamic_obstacles) * int(config.obstacle_embed_dim)
        context_dim += int(config.action_embed_dim) + int(config.action_cont_dim)

        self.goal_encoder: nn.Module | None = None
        if bool(config.use_goal):
            self.goal_encoder = _mlp(
                input_dim=int(config.goal_dim),
                hidden_dim=int(config.goal_embed_dim),
                output_dim=int(config.goal_embed_dim),
                dropout=float(config.dropout),
            )
            context_dim += int(config.goal_embed_dim)

        self.lidar_encoder: nn.Module | None = None
        if bool(config.use_lidar):
            self.lidar_encoder = _mlp(
                input_dim=int(config.lidar_dim),
                hidden_dim=int(config.lidar_embed_dim),
                output_dim=int(config.lidar_embed_dim),
                dropout=float(config.dropout),
            )
            context_dim += int(config.lidar_embed_dim)

        self.gru = nn.GRU(
            input_size=context_dim,
            hidden_size=int(config.gru_hidden_dim),
            num_layers=int(config.gru_layers),
            batch_first=True,
            dropout=float(config.dropout) if int(config.gru_layers) > 1 else 0.0,
        )
        self.head_body = _mlp(
            input_dim=int(config.gru_hidden_dim),
            hidden_dim=int(config.head_hidden_dim),
            output_dim=int(config.head_hidden_dim),
            dropout=float(config.dropout),
        )
        self.delta_head = nn.Linear(int(config.head_hidden_dim), int(config.num_dynamic_obstacles) * 2)
        self.risk_head = nn.Linear(int(config.head_hidden_dim), 1)
        self.collision_next_head = nn.Linear(int(config.head_hidden_dim), 1)
        self.collision_within_3_head = nn.Linear(int(config.head_hidden_dim), 1)
        self.collision_within_5_head = nn.Linear(int(config.head_hidden_dim), 1)

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
        """Run a full sequence with a fresh GRU hidden state for each batch."""
        step_features = self._encode_sequence_inputs(
            obstacle_pos=obstacle_pos,
            obstacle_vel=obstacle_vel,
            radii=radii,
            action_index=action_index,
            action_cont=action_cont,
            goal=goal,
            lidar=lidar,
        )
        gru_out, _ = self.gru(step_features)
        return self._heads_from_hidden(gru_out)

    def forward_step(
        self,
        obstacle_pos: torch.Tensor,
        obstacle_vel: torch.Tensor,
        radii: torch.Tensor,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None = None,
        lidar: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run a single step, optionally continuing a provided GRU state."""
        if obstacle_pos.ndim != 3:
            raise ValueError("forward_step expects obstacle_pos shape (B, N_dyn, 2)")
        step_features = self._encode_sequence_inputs(
            obstacle_pos=obstacle_pos[:, None],
            obstacle_vel=obstacle_vel[:, None],
            radii=radii,
            action_index=action_index[:, None],
            action_cont=action_cont[:, None],
            goal=goal[:, None] if goal is not None else None,
            lidar=lidar[:, None] if lidar is not None else None,
        )
        gru_out, new_hidden = self.gru(step_features, hidden)
        outputs = self._heads_from_hidden(gru_out)
        squeezed = {key: value[:, 0] for key, value in outputs.items()}
        return squeezed, new_hidden

    def _encode_sequence_inputs(
        self,
        obstacle_pos: torch.Tensor,
        obstacle_vel: torch.Tensor,
        radii: torch.Tensor,
        action_index: torch.Tensor,
        action_cont: torch.Tensor,
        goal: torch.Tensor | None,
        lidar: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, n_dyn, _ = obstacle_pos.shape
        if n_dyn != self.num_dynamic_obstacles:
            raise ValueError(f"Expected {self.num_dynamic_obstacles} obstacles, got {n_dyn}")
        if radii.ndim != 2:
            raise ValueError("radii must have shape (B, N_dyn)")

        radii_bt = radii[:, None, :, None].expand(batch_size, seq_len, n_dyn, 1)
        obstacle_features = torch.cat([obstacle_pos, obstacle_vel, radii_bt], dim=-1)
        obstacle_flat = obstacle_features.reshape(batch_size * seq_len * n_dyn, -1)
        obstacle_emb = self.obstacle_encoder(obstacle_flat)
        obstacle_emb = obstacle_emb.reshape(batch_size, seq_len, n_dyn * int(self.config.obstacle_embed_dim))

        action_emb = self.action_embedding(action_index.long())
        pieces = [obstacle_emb, action_emb, action_cont]

        if self.goal_encoder is not None:
            if goal is None:
                raise ValueError("Model was configured with use_goal=true but goal=None was provided")
            goal_emb = self.goal_encoder(goal.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
            pieces.append(goal_emb)

        if self.lidar_encoder is not None:
            if lidar is None:
                raise ValueError("Model was configured with use_lidar=true but lidar=None was provided")
            lidar_emb = self.lidar_encoder(lidar.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
            pieces.append(lidar_emb)

        return torch.cat(pieces, dim=-1)

    def _heads_from_hidden(self, hidden_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = hidden_sequence.shape
        body = self.head_body(hidden_sequence)
        pred_delta_scaled = self.delta_head(body).reshape(batch_size, seq_len, self.num_dynamic_obstacles, 2)
        pred_delta = pred_delta_scaled * self.delta_scale
        raw_risk = self.risk_head(body).squeeze(-1)
        return {
            "pred_delta_rel_scaled": pred_delta_scaled,
            "pred_delta_rel": pred_delta,
            "raw_risk": raw_risk,
            "pred_risk": torch.sigmoid(raw_risk),
            "collision_next_logits": self.collision_next_head(body).squeeze(-1),
            "collision_within_3_logits": self.collision_within_3_head(body).squeeze(-1),
            "collision_within_5_logits": self.collision_within_5_head(body).squeeze(-1),
        }
