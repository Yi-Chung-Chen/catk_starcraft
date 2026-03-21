"""StarCraft target builder.

All units alive at current_frame_idx with sufficient future are training targets.
No ego-centric filtering.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

CURRENT_FRAME_IDX = 16


class SCTargetBuilderTrain(BaseTransform):
    def __init__(self, max_num: int, min_future_alive: int = 8) -> None:
        super().__init__()
        self.max_num = max_num
        self.min_future_alive = min_future_alive

    def forward(self, data) -> HeteroData:
        valid = data["agent"]["valid_mask"]  # [N, T]
        alive_now = valid[:, CURRENT_FRAME_IDX]  # [N]
        future_alive = valid[:, CURRENT_FRAME_IDX + 1 :].sum(-1)  # [N]
        train_mask = alive_now & (future_alive >= self.min_future_alive)

        if train_mask.sum() > self.max_num:
            indices = torch.where(train_mask)[0]
            selected = indices[torch.randperm(indices.size(0))[: self.max_num]]
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
