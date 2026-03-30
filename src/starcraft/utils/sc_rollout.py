"""Contour-based rollout sampling for StarCraft."""

from typing import Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical

from src.smart.utils import cal_circular_contour, transform_to_global, wrap_angle


@torch.no_grad()
def sample_next_token_traj_contour(
    token_traj: Tensor,  # [n_token, 4, 2]
    token_traj_all: Tensor,  # [n_token, 9, 4, 2]
    sampling_scheme: DictConfig,
    next_token_logits: Tensor,  # [n_agent, n_token]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
    pos_next_gt: Tensor,  # [n_agent, 2]
    head_next_gt: Tensor,  # [n_agent]
    valid_next_gt: Tensor,  # [n_agent]
) -> Tuple[Tensor, Tensor]:
    """Sample next token using circular contour distance (radius=0.5).

    Returns:
        next_token_idx: [n_agent]
        next_token_traj_all: [n_agent, 9, 4, 2] in local coords
    """
    n_agent = next_token_logits.shape[0]
    range_a = torch.arange(n_agent, device=next_token_logits.device)
    logits = next_token_logits.detach()

    if (
        sampling_scheme.criterium == "topk_prob"
        or sampling_scheme.criterium == "topk_prob_sampled_with_dist"
    ):
        topk_logits, topk_indices = torch.topk(
            logits, sampling_scheme.num_k, dim=-1, sorted=False
        )
        if sampling_scheme.criterium == "topk_prob_sampled_with_dist":
            # GT contour: [n_agent, 4, 2] in global coord (circular, radius=0.5)
            gt_contour = cal_circular_contour(pos_next_gt, head_next_gt)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # Top-K token contours from shared vocab: [n_agent, K, 4, 2]
            topk_contours = token_traj[topk_indices]
            # Transform to global
            topk_contours_global = transform_to_global(
                pos_local=topk_contours.flatten(1, 2),  # [n_agent, K*4, 2]
                head_local=None,
                pos_now=pos_now,
                head_now=head_now,
            )[0].view(*topk_contours.shape)

            # dist: [n_agent, K]
            dist = torch.norm(topk_contours_global - gt_contour, dim=-1).mean(-1)
            topk_logits = topk_logits.masked_fill(
                valid_next_gt.unsqueeze(1), 0.0
            ) - 1.0 * dist.masked_fill(~valid_next_gt.unsqueeze(1), 0.0)
    else:
        raise ValueError(f"Invalid criterium: {sampling_scheme.criterium}")

    topk_logits = topk_logits / sampling_scheme.temp
    samples = Categorical(logits=topk_logits).sample()
    next_token_idx = topk_indices[range_a, samples]  # [n_agent]

    # Get full token trajectory for the selected tokens
    next_token_traj_all = token_traj_all[next_token_idx]  # [n_agent, 9, 4, 2]

    return next_token_idx, next_token_traj_all
