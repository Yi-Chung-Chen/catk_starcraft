"""StarCraft decoder. Map encoder (CNN) + agent decoder with pl2a attention."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from src.starcraft.tokens.sc_token_processor import (
    _AGENT_DIM0_KEYS,
    _PER_SCENARIO_KEYS,
    _SHARED_KEYS,
)

from .sc_agent_decoder import SCAgentDecoder
from .sc_map_encoder import SCMapEncoder


def _replicate_map_feature(
    map_feature: Dict[str, Tensor], n_rollouts: int, num_graphs: int,
) -> Dict[str, Tensor]:
    """Tile `map_feature` R times along dim 0; offset `batch` by B*r per replica.

    Cat-order: rows `[0, B*625)` belong to r=0, `[B*625, 2*B*625)` to r=1, ...
    The replicated `batch` values span `[0, B*R)`, keeping every replica on a
    disjoint scenario id so `radius(batch_x, batch_y)` edge builders treat
    replicas as independent graphs.
    """
    R = n_rollouts
    B = num_graphs
    out: Dict[str, Tensor] = {}
    for k, v in map_feature.items():
        if k == "batch":
            out[k] = torch.cat([v + B * r for r in range(R)], dim=0)
        else:
            out[k] = v.repeat(R, *([1] * (v.dim() - 1)))
    return out


def _replicate_tokenized_agent(
    tokenized_agent: Dict[str, Tensor],
    n_rollouts: int,
    teacher_force_mask: Optional[Tensor],
    visibility_gate: Optional[Tensor],
) -> Tuple[Dict[str, Tensor], Optional[Tensor], Optional[Tensor]]:
    """Replicate per-agent + per-scenario tensors R times; leave shared untouched.

    Classification:
      - `_AGENT_DIM0_KEYS`      → repeat R along dim 0 (cat-order).
      - `_PER_SCENARIO_KEYS`    → repeat R along dim 0 (cat-order).
      - `_SHARED_KEYS`          → passed through unchanged (shared vocabulary).
      - `batch`                 → concat with B*r offsets so replicas are disjoint.
      - `num_graphs` (scalar)   → multiplied by R.

    `teacher_force_mask [n_agent]` and `visibility_gate [n_agent, 18]` are
    replicated in the same cat-order as the per-agent tensors.
    """
    R = n_rollouts
    B = int(tokenized_agent["num_graphs"])
    out: Dict[str, Tensor] = {}
    for k, v in tokenized_agent.items():
        if k == "num_graphs":
            out[k] = B * R
        elif k == "batch":
            out[k] = torch.cat([v + B * r for r in range(R)], dim=0)
        elif k in _SHARED_KEYS:
            out[k] = v
        elif k in _AGENT_DIM0_KEYS or k in _PER_SCENARIO_KEYS:
            out[k] = v.repeat(R, *([1] * (v.dim() - 1)))
        else:
            raise KeyError(
                f"Unclassified tokenized_agent key '{k}' for rollout "
                "replication — add to _AGENT_DIM0_KEYS, _PER_SCENARIO_KEYS, "
                "or _SHARED_KEYS in sc_token_processor.py"
            )
    tfm_R = teacher_force_mask.repeat(R) if teacher_force_mask is not None else None
    vg_R = visibility_gate.repeat(R, 1) if visibility_gate is not None else None
    return out, tfm_R, vg_R


def _unbatch_rollouts(
    pred: Dict[str, Tensor], n_rollouts: int, n_agent: int,
) -> Dict[str, Tensor]:
    """Reshape agent-axis outputs `[n_agent*R, ...]` → `[n_agent, R, ...]`.

    Assumes the batched call was produced by cat-order replication so that
    rows `[r*n_agent : (r+1)*n_agent]` belong to replica r. Non-agent-axis
    values (scalars, or tensors whose leading dim isn't `n_agent*R`) pass
    through unchanged.
    """
    R = n_rollouts
    out: Dict[str, Tensor] = {}
    for k, v in pred.items():
        if isinstance(v, Tensor) and v.dim() >= 1 and v.shape[0] == n_agent * R:
            out[k] = v.view(R, n_agent, *v.shape[1:]).transpose(0, 1).contiguous()
        else:
            out[k] = v
    return out


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
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        visibility_gate: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        map_feature = self.map_encoder(tokenized_map)
        return self.agent_encoder(
            tokenized_agent, map_feature, visibility_gate=visibility_gate,
        )

    def inference(
        self,
        tokenized_map: Dict[str, Tensor],
        tokenized_agent: Dict[str, Tensor],
        sampling_scheme: DictConfig,
        teacher_force_mask: Optional[Tensor] = None,
        visibility_gate: Optional[Tensor] = None,
        n_rollouts: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Closed-loop rollout.

        When `n_rollouts is None` (default, training path), returns a dict of
        per-agent tensors with the original leading shape `[n_agent, ...]`.

        When `n_rollouts` is an int (closed-loop validation path), R rollout
        replicas are stitched together as disjoint graphs in a single
        `agent_encoder.inference` call. Per-agent tensors in the returned
        dict are reshaped to `[n_agent, R, ...]` — even for R=1, so callers
        see a uniform shape.
        """
        map_feature = self.map_encoder(tokenized_map)
        if n_rollouts is None:
            return self.agent_encoder.inference(
                tokenized_agent, map_feature, sampling_scheme,
                teacher_force_mask=teacher_force_mask,
                visibility_gate=visibility_gate,
            )

        R = int(n_rollouts)
        B = int(tokenized_agent["num_graphs"])
        n_agent = int(tokenized_agent["valid_mask"].shape[0])

        map_feature_R = _replicate_map_feature(map_feature, R, B)
        ta_R, tfm_R, vg_R = _replicate_tokenized_agent(
            tokenized_agent, R, teacher_force_mask, visibility_gate,
        )
        pred_batched = self.agent_encoder.inference(
            ta_R, map_feature_R, sampling_scheme,
            teacher_force_mask=tfm_R,
            visibility_gate=vg_R,
        )
        return _unbatch_rollouts(pred_batched, R, n_agent)
