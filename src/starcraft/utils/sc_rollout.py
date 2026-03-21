"""Center+heading based rollout sampling for StarCraft."""

from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

from src.smart.utils import transform_to_global, wrap_angle


@torch.no_grad()
def sample_next_token_traj_center(
    token_traj_all: Tensor,  # [n_token, 9, 3]
    token_endpoint_xy: Tensor,  # [n_token, 2]
    token_endpoint_heading: Tensor,  # [n_token]
    sampling_scheme: DictConfig,
    next_token_logits: Tensor,  # [n_agent, n_token]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
    pos_next_gt: Tensor,  # [n_agent, 2]
    head_next_gt: Tensor,  # [n_agent]
    valid_next_gt: Tensor,  # [n_agent]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample next token using center-based distance.

    Returns:
        next_token_idx: [n_agent]
        next_pos: [n_agent, 2] global position at token endpoint
        next_head: [n_agent] global heading at token endpoint
        next_token_traj_all: [n_agent, 9, 3] local token trajectory
    """
    n_agent = next_token_logits.shape[0]
    range_a = torch.arange(n_agent, device=next_token_logits.device)
    logits = next_token_logits.detach()

    # Transform all token endpoints to global coords
    cos_h = torch.cos(head_now)  # [n_agent]
    sin_h = torch.sin(head_now)
    tx = token_endpoint_xy[:, 0]  # [n_token]
    ty = token_endpoint_xy[:, 1]
    # [n_agent, n_token]
    global_x = cos_h.unsqueeze(1) * tx.unsqueeze(0) - sin_h.unsqueeze(1) * ty.unsqueeze(0) + pos_now[:, 0:1]
    global_y = sin_h.unsqueeze(1) * tx.unsqueeze(0) + cos_h.unsqueeze(1) * ty.unsqueeze(0) + pos_now[:, 1:2]

    if (
        sampling_scheme.criterium == "topk_prob"
        or sampling_scheme.criterium == "topk_prob_sampled_with_dist"
    ):
        topk_logits, topk_indices = torch.topk(
            logits, sampling_scheme.num_k, dim=-1, sorted=False
        )
        if sampling_scheme.criterium == "topk_prob_sampled_with_dist":
            # Center distance to GT
            topk_gx = global_x[range_a.unsqueeze(1), topk_indices]  # [n_agent, K]
            topk_gy = global_y[range_a.unsqueeze(1), topk_indices]
            dist = (topk_gx - pos_next_gt[:, 0:1]) ** 2 + (topk_gy - pos_next_gt[:, 1:2]) ** 2
            dist = dist.sqrt()
            topk_logits = topk_logits.masked_fill(
                valid_next_gt.unsqueeze(1), 0.0
            ) - 1.0 * dist.masked_fill(~valid_next_gt.unsqueeze(1), 0.0)
    else:
        raise ValueError(f"Invalid criterium: {sampling_scheme.criterium}")

    topk_logits = topk_logits / sampling_scheme.temp
    samples = Categorical(logits=topk_logits).sample()
    next_token_idx = topk_indices[range_a, samples]  # [n_agent]

    # Compute next state
    next_pos = torch.stack(
        [global_x[range_a, next_token_idx], global_y[range_a, next_token_idx]], dim=-1
    )  # [n_agent, 2]
    next_head = wrap_angle(head_now + token_endpoint_heading[next_token_idx])  # [n_agent]

    # Get full token trajectory for the selected tokens
    next_token_traj_all = token_traj_all[next_token_idx]  # [n_agent, 9, 3]

    return next_token_idx, next_pos, next_head, next_token_traj_all
