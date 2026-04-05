"""Auxiliary loss for action classification and target position prediction."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class SCActionTargetLoss(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        has_action_weight: float = 1.0,
        has_target_pos_weight: float = 1.0,
        action_weight: float = 1.0,
        target_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.has_action_weight = has_action_weight
        self.has_target_pos_weight = has_target_pos_weight
        self.action_weight = action_weight
        self.target_weight = target_weight
        # Total weighted loss
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")
        # Per-component: each with its own count for correct averaging
        self.add_state("loss_ha_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_ha", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_htp_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_htp", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_action_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_action", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss_target_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_target", default=tensor(0.0), dist_reduce_fx="sum")
        self._batch_components = {k: tensor(0.0) for k in
                                  ("has_action", "has_target_pos", "action_class", "target_pos")}

    def update(
        self,
        has_action_logits: Tensor,       # [n_agent, 16]
        has_target_pos_logits: Tensor,   # [n_agent, 16]
        action_class_logits: Tensor,     # [n_agent, 16, n_action]
        target_pos_pred: Tensor,         # [n_agent, 16, 2]
        gt_has_action: Tensor,           # [n_agent, 16]
        gt_has_target_pos: Tensor,       # [n_agent, 16]
        gt_coarse_action: Tensor,        # [n_agent, 16] classes 0-10 (NO_OP excluded)
        gt_rel_target_pos: Tensor,       # [n_agent, 16, 2]
        next_token_valid: Tensor,        # [n_agent, 16]
        train_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        valid = next_token_valid
        if self.training and train_mask is not None:
            valid = valid & train_mask.unsqueeze(1)

        n_valid = valid.sum()
        if n_valid == 0:
            return

        total_loss = tensor(0.0, device=has_action_logits.device)

        # has_action: binary CE on all valid steps
        loss_ha = F.binary_cross_entropy_with_logits(
            has_action_logits[valid], gt_has_action[valid].float(), reduction="mean"
        )
        total_loss = total_loss + self.has_action_weight * loss_ha
        self.loss_ha_sum += loss_ha.detach() * n_valid
        self.count_ha += n_valid

        # has_target_pos: binary CE on all valid steps
        loss_htp = F.binary_cross_entropy_with_logits(
            has_target_pos_logits[valid], gt_has_target_pos[valid].float(), reduction="mean"
        )
        total_loss = total_loss + self.has_target_pos_weight * loss_htp
        self.loss_htp_sum += loss_htp.detach() * n_valid
        self.count_htp += n_valid

        # action_class: CE only where has_action=True (classes 0-10, NO_OP excluded)
        action_mask = valid & gt_has_action
        n_action = action_mask.sum()
        if n_action > 0:
            loss_action = F.cross_entropy(
                action_class_logits[action_mask], gt_coarse_action[action_mask].long(),
                reduction="mean",
            )
            total_loss = total_loss + self.action_weight * loss_action
            self.loss_action_sum += loss_action.detach() * n_action
            self.count_action += n_action

        # target_pos: L1 only where has_target_pos=True
        target_mask = valid & gt_has_target_pos
        n_target = target_mask.sum()
        if n_target > 0:
            loss_target = F.l1_loss(
                target_pos_pred[target_mask], gt_rel_target_pos[target_mask],
                reduction="mean",
            )
            total_loss = total_loss + self.target_weight * loss_target
            self.loss_target_sum += loss_target.detach() * n_target
            self.count_target += n_target

        self.loss_sum += total_loss * n_valid
        self.count += n_valid

        # Cache per-batch values for step-level logging
        self._batch_components = {
            "has_action": loss_ha.detach(),
            "has_target_pos": loss_htp.detach(),
            "action_class": loss_action.detach() if n_action > 0 else tensor(0.0, device=has_action_logits.device),
            "target_pos": loss_target.detach() if n_target > 0 else tensor(0.0, device=has_action_logits.device),
        }

    def compute(self) -> Tensor:
        return self.loss_sum / self.count

    def batch_components(self) -> Dict[str, Tensor]:
        """Return per-batch (last update) loss components for step-level logging."""
        return self._batch_components

    def compute_components(self) -> Dict[str, Tensor]:
        """Return correctly averaged individual loss components for epoch-level logging."""
        return {
            "has_action": self.loss_ha_sum / self.count_ha.clamp(min=1),
            "has_target_pos": self.loss_htp_sum / self.count_htp.clamp(min=1),
            "action_class": self.loss_action_sum / self.count_action.clamp(min=1),
            "target_pos": self.loss_target_sum / self.count_target.clamp(min=1),
        }
