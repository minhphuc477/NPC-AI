"""GSPE model and utilities.

GSPE encodes structured game state into virtual prefix tokens that can be
prepended to model input embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from gspe.state_codec import FIELD_VOCABS, encode_game_state


@dataclass
class GSPEConfig:
    n_prefix_tokens: int = 20
    embed_dim: int = 128
    prefix_hidden: int = 64
    mlp_hidden_1: int = 512
    mlp_hidden_2: int = 512
    dropout: float = 0.1
    lm_hidden_dim: int = 3072

    @property
    def compressed_dim(self) -> int:
        return int(self.n_prefix_tokens) * int(self.prefix_hidden)


class GSPE(nn.Module):
    """Encode game-state IDs into virtual tokens (batch, n_prefix, hidden)."""

    def __init__(self, cfg: GSPEConfig):
        super().__init__()
        self.cfg = cfg

        self.field_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(len(vocab), cfg.embed_dim)
                for field, vocab in FIELD_VOCABS.items()
            }
        )
        total_embed_dim = len(FIELD_VOCABS) * cfg.embed_dim

        self.input_norm = nn.LayerNorm(total_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(total_embed_dim, cfg.mlp_hidden_1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_1, cfg.mlp_hidden_2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden_2, cfg.compressed_dim),
        )
        self.token_proj = nn.Linear(cfg.prefix_hidden, cfg.lm_hidden_dim)
        self.token_norm = nn.LayerNorm(cfg.lm_hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, field_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return virtual tokens with shape (batch, n_prefix_tokens, lm_hidden_dim)."""
        embeddings = []
        for field in sorted(FIELD_VOCABS.keys()):
            ids = field_ids[field]
            embeddings.append(self.field_embeddings[field](ids))

        x = torch.cat(embeddings, dim=-1)
        x = self.input_norm(x)
        x = self.mlp(x)
        x = x.view(-1, self.cfg.n_prefix_tokens, self.cfg.prefix_hidden)
        x = self.token_proj(x)
        x = self.token_norm(x)
        return x

    def encode_from_dict(self, game_state: Dict[str, object], device: torch.device) -> torch.Tensor:
        ids = encode_game_state(game_state)
        field_ids = {
            field: torch.tensor([idx], dtype=torch.long, device=device)
            for field, idx in ids.items()
        }
        with torch.no_grad():
            return self(field_ids)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
