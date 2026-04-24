# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    if not finetune:
        return

    for p in model.parameters():
        p.requires_grad = False

    try:
        _unfreeze(model.agent_encoder.token_predict_head)
        log.info("Unfreezing token_predict_head")
    except AttributeError:
        log.info("No token_predict_head in model.agent_encoder")

    try:
        _unfreeze(model.agent_encoder.gmm_logits_head)
        _unfreeze(model.agent_encoder.gmm_pose_head)
        log.info("Unfreezing gmm heads")
    except AttributeError:
        log.info("No gmm heads in model.agent_encoder")

    # Attention stacks: SMART uses pt2a_attn_layers, SC uses pl2a_attn_layers.
    # Try both so one helper covers both model families.
    for attr in ("t_attn_layers", "pt2a_attn_layers", "pl2a_attn_layers", "a2a_attn_layers"):
        try:
            _unfreeze(getattr(model.agent_encoder, attr))
            log.info(f"Unfreezing {attr}")
        except AttributeError:
            pass

    # SC-only: concept attention layers
    try:
        _unfreeze(model.agent_encoder.concept_attn_layers)
        log.info("Unfreezing concept_attn_layers")
    except AttributeError:
        pass

    # SC-only: auxiliary prediction heads (present when use_aux_loss=true)
    for head in (
        "has_action_head",
        "has_target_pos_head",
        "action_class_head",
        "target_pos_head",
    ):
        try:
            _unfreeze(getattr(model.agent_encoder, head))
            log.info(f"Unfreezing {head}")
        except AttributeError:
            pass
