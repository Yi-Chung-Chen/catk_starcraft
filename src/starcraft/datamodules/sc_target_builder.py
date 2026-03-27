"""StarCraft target builder.

Player-owned units (P1/P2) alive at current_frame_idx with sufficient future are
training targets.  Neutral units are always excluded.  When the candidate count
exceeds *max_num*, moving units are prioritized over static ones.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from src.starcraft.utils.unit_type_map import (
    MOVING_UNIT_TYPE_IDS,
    SC2_ID_TO_INDEX,
)

CURRENT_FRAME_IDX = 16

# Remapped indices for moving unit types (used to match data["agent"]["type"])
_MOVING_TYPE_INDICES: frozenset[int] = frozenset(
    SC2_ID_TO_INDEX[sc2_id] for sc2_id in MOVING_UNIT_TYPE_IDS if sc2_id in SC2_ID_TO_INDEX
)

# Owner values: 1 = player 1, 2 = player 2, 16 = neutral
_PLAYER_OWNERS: frozenset[int] = frozenset({1, 2})


class SCTargetBuilderTrain(BaseTransform):
    def __init__(self, max_num: int, min_future_alive: int = 8) -> None:
        super().__init__()
        self.max_num = max_num
        self.min_future_alive = min_future_alive

    def forward(self, data) -> HeteroData:
        valid = data["agent"]["valid_mask"]  # [N, T]
        alive_now = valid[:, CURRENT_FRAME_IDX]  # [N]
        future_alive = valid[:, CURRENT_FRAME_IDX + 1 :].sum(-1)  # [N]
        owner = data["agent"]["owner"]  # [N]
        is_player = torch.tensor(
            [o.item() in _PLAYER_OWNERS for o in owner], dtype=torch.bool
        )
        train_mask = alive_now & (future_alive >= self.min_future_alive) & is_player

        if train_mask.sum() > self.max_num:
            indices = torch.where(train_mask)[0]
            agent_type = data["agent"]["type"]  # [N] remapped indices

            is_moving = torch.tensor(
                [agent_type[i].item() in _MOVING_TYPE_INDICES for i in indices],
                dtype=torch.bool,
            )

            # Priority: moving player units first, then static player units
            moving_mask = is_moving
            static_mask = ~is_moving

            selected = []
            for mask in (moving_mask, static_mask):
                if len(selected) >= self.max_num:
                    break
                pool = indices[mask]
                remaining = self.max_num - len(selected)
                if len(pool) > remaining:
                    pool = pool[torch.randperm(len(pool))[:remaining]]
                selected.append(pool)

            selected = torch.cat(selected) if selected else indices[:0]
            train_mask = torch.zeros_like(train_mask)
            train_mask[selected] = True

        data["agent"]["train_mask"] = train_mask
        return HeteroData(data)


class SCTargetBuilderVal(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data) -> HeteroData:
        N = data["agent"]["valid_mask"].shape[0]
        data["agent"]["train_mask"] = torch.ones(N, dtype=torch.bool)
        return HeteroData(data)
