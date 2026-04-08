"""StarCraft cross-entropy loss with contour-based token matching."""

from typing import Optional

import torch
from torch import Tensor, tensor
from torch.nn.functional import cross_entropy, one_hot
from torchmetrics.metric import Metric

from src.smart.metrics.utils import get_euclidean_targets
from src.smart.utils import cal_circular_contour


class SCCrossEntropy(Metric):

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        use_gt_raw: bool,
        gt_thresh_scale_length: float,
        label_smoothing: float,
        rollout_as_gt: bool,
    ) -> None:
        super().__init__()
        self.use_gt_raw = use_gt_raw
        self.gt_thresh_scale_length = gt_thresh_scale_length
        self.label_smoothing = label_smoothing
        self.rollout_as_gt = rollout_as_gt
        self.add_state("loss_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        next_token_logits: Tensor,  # [n_agent, 16, n_token]
        next_token_valid: Tensor,  # [n_agent, 16]
        pred_pos: Tensor,  # [n_agent, 18, 2]
        pred_head: Tensor,  # [n_agent, 18]
        pred_valid: Tensor,  # [n_agent, 18]
        gt_pos_raw: Tensor,  # [n_agent, 18, 2]
        gt_head_raw: Tensor,  # [n_agent, 18]
        gt_valid_raw: Tensor,  # [n_agent, 18]
        gt_pos: Tensor,  # [n_agent, 18, 2]
        gt_head: Tensor,  # [n_agent, 18]
        gt_valid: Tensor,  # [n_agent, 18]
        token_agent_shape: Tensor,  # [n_agent, 1]
        token_traj: Tensor,  # [n_token, 4, 2]
        train_mask: Optional[Tensor] = None,
        next_token_action: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        if self.use_gt_raw:
            gt_pos = gt_pos_raw
            gt_head = gt_head_raw
            gt_valid = gt_valid_raw

        if self.gt_thresh_scale_length > 0:
            dist = torch.norm(pred_pos - gt_pos, dim=-1)
            _thresh = token_agent_shape[:, 0] * self.gt_thresh_scale_length
            gt_valid = gt_valid & (dist < _thresh.unsqueeze(1))

        euclidean_target, euclidean_target_valid = get_euclidean_targets(
            pred_pos=pred_pos, pred_head=pred_head, pred_valid=pred_valid,
            gt_pos=gt_pos, gt_head=gt_head, gt_valid=gt_valid,
        )
        if self.rollout_as_gt and (next_token_action is not None):
            euclidean_target = next_token_action

        prob_target = _get_prob_targets_contour(
            target=euclidean_target,  # [n_agent, 16, 3] local x,y,yaw
            token_traj=token_traj,  # [n_token, 4, 2]
        )  # [n_agent, 16, n_token]

        loss = cross_entropy(
            next_token_logits.transpose(1, 2),
            prob_target.transpose(1, 2),
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        loss_weighting_mask = next_token_valid & euclidean_target_valid
        if train_mask is not None:
            loss_weighting_mask &= train_mask.unsqueeze(1)

        self.loss_sum += (loss * loss_weighting_mask).sum()
        self.count += (loss_weighting_mask > 0).sum()

    def compute(self) -> Tensor:
        if self.count == 0:
            return self.loss_sum.new_tensor(0.0)
        return self.loss_sum / self.count


@torch.no_grad()
def _get_prob_targets_contour(
    target: Tensor,  # [n_agent, n_step, 3] local x,y,yaw
    token_traj: Tensor,  # [n_token, 4, 2]
) -> Tensor:  # [n_agent, n_step, n_token]
    """Contour-based probability targets (sum L2 over 4 corners, circular radius=0.5)."""
    contour = cal_circular_contour(
        target[..., :2],  # [n_agent, n_step, 2]
        target[..., 2],  # [n_agent, n_step]
    )  # [n_agent, n_step, 4, 2]

    # [n_agent, n_step, 1, 4, 2] - [1, 1, n_token, 4, 2]
    dist = torch.norm(
        contour.unsqueeze(2) - token_traj[None, None], dim=-1
    ).sum(-1)  # [n_agent, n_step, n_token]

    target_token_index = dist.argmin(-1)  # [n_agent, n_step]
    prob_target = one_hot(target_token_index, num_classes=token_traj.shape[0])
    return prob_target.to(target.dtype)
