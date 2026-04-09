"""StarCraft decoder. Map encoder (CNN) + agent decoder with pl2a attention."""

from typing import Dict, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .sc_agent_decoder import SCAgentDecoder
from .sc_map_encoder import SCMapEncoder


class SCDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        num_map_layers: int,
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        num_concepts: int = 16,
        use_aux_loss: bool = True,
        use_action_target_input: bool = False,
        closed_loop_oracle_intent_input: bool = False,
        num_action_classes: int = 11,
    ) -> None:
        super().__init__()
        self.map_encoder = SCMapEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = SCAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            num_concepts=num_concepts,
            use_aux_loss=use_aux_loss,
            use_action_target_input=use_action_target_input,
            closed_loop_oracle_intent_input=closed_loop_oracle_intent_input,
            num_action_classes=num_action_classes,
        )

    def forward(
        self, tokenized_map: Dict[str, Tensor], tokenized_agent: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder(tokenized_agent, map_feature)

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder.inference(tokenized_agent, map_feature, sampling_scheme)
