"""StarCraft target builder.

Training targets: movable player-owned units (P1/P2) alive at current_frame_idx
with sufficient future frames.  Static buildings and neutral units are excluded
from the loss but remain in the attention graph as context.
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
    def __init__(self, min_future_alive: int = 8) -> None:
        super().__init__()
        self.min_future_alive = min_future_alive

    def forward(self, data) -> HeteroData:
        valid = data["agent"]["valid_mask"]  # [N, T]
        alive_now = valid[:, CURRENT_FRAME_IDX]  # [N]
        future_alive = valid[:, CURRENT_FRAME_IDX + 1 :].sum(-1)  # [N]
        owner = data["agent"]["owner"]  # [N]
        agent_type = data["agent"]["type"]  # [N] remapped indices

        is_player = torch.tensor(
            [o.item() in _PLAYER_OWNERS for o in owner], dtype=torch.bool
        )
        is_moving = torch.tensor(
            [t.item() in _MOVING_TYPE_INDICES for t in agent_type], dtype=torch.bool
        )

        # Only movable player units contribute to the loss.
        # Static buildings and neutrals remain in the attention graph as context.
        train_mask = alive_now & (future_alive >= self.min_future_alive) & is_player & is_moving

        data["agent"]["train_mask"] = train_mask
        return HeteroData(data)


class SCTargetBuilderVal(BaseTransform):
    """Val/test targets: movable player units (no min_future_alive requirement).

    Aligns val metrics with the training target definition so val loss
    reflects actual motion prediction quality, not trivially-static units.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data) -> HeteroData:
        owner = data["agent"]["owner"]
        agent_type = data["agent"]["type"]

        is_player = torch.tensor(
            [o.item() in _PLAYER_OWNERS for o in owner], dtype=torch.bool
        )
        is_moving = torch.tensor(
            [t.item() in _MOVING_TYPE_INDICES for t in agent_type], dtype=torch.bool
        )

        data["agent"]["train_mask"] = is_player & is_moving
        return HeteroData(data)
