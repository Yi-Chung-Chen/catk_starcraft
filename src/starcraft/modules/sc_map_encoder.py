"""StarCraft map encoder.

CNN-based encoder that processes map grids (pathing + height + creep) into
patch tokens for map-to-agent attention. No sparse self-attention — the CNN
receptive field handles spatial context within the well-structured grid.
"""

from typing import Dict

import torch
import torch.nn as nn

from src.smart.utils import weight_init


class SCMapEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Build CNN with num_layers stride-2 conv layers → total 2^num_layers downsampling.
        # Input: [B, 3, 200, 200] → Output: [B, hidden_dim, 25, 25] (for num_layers=3).
        # 3 input channels: pathing_grid, height_map, creep.
        channels = [3] + [min(32 * (2 ** i), hidden_dim) for i in range(num_layers - 1)] + [hidden_dim]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(1, channels[i + 1]))  # equivalent to LayerNorm for conv
            layers.append(nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, tokenized_map: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.cnn(tokenized_map["map_grid"])  # [B, hidden_dim, 25, 25]
        B = x.shape[0]
        # Reshape to flat patch tokens: [B*625, hidden_dim]
        x = x.permute(0, 2, 3, 1).reshape(B * 25 * 25, -1)

        valid_mask = tokenized_map["valid_mask"]  # [B*625]
        x[~valid_mask] = 0.0

        orient = torch.zeros(x.shape[0], device=x.device)  # fixed heading = 0 (x-axis)

        return {
            "pt_token": x,                              # [n_patch_total, hidden_dim]
            "position": tokenized_map["position"],      # [n_patch_total, 2]
            "orientation": orient,                      # [n_patch_total]
            "batch": tokenized_map["batch"],            # [n_patch_total]
            "valid_mask": valid_mask,                   # [n_patch_total]
        }
