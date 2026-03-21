"""StarCraft decoder. Agent-only (no map encoder)."""

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .sc_agent_decoder import SCAgentDecoder


class SCDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        a2a_radius: float,
        num_freq_bands: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
    ) -> None:
        super().__init__()
        self.agent_encoder = SCAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
        )

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return self.agent_encoder(tokenized_agent, tokenized_map)

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        return self.agent_encoder.inference(tokenized_agent, tokenized_map, sampling_scheme)
