"""StarCraft Motion dataset loader.

Reads per-scenario HDF5 files and produces agent tensors compatible with the
CAT-K training pipeline.
"""

from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset

from src.starcraft.utils.unit_type_map import remap_unit_type
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CURRENT_FRAME_IDX = 16
TOTAL_FRAMES = 145


class SCDataset(Dataset):
    """Dataset reading StarCraft scenario HDF5 files via a manifest."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        transform: Optional[Callable] = None,
        map_names: Optional[List[str]] = None,
    ) -> None:
        dataset_root = Path(dataset_root)
        manifest_path = dataset_root / "manifests" / f"{split}_manifest.txt"
        with open(manifest_path) as f:
            rel_paths = [line.strip() for line in f if line.strip()]
        self._file_paths = [str(dataset_root / p) for p in rel_paths]
        if map_names is not None:
            map_set = set(map_names)
            self._file_paths = [
                p for p in self._file_paths if Path(p).parent.name in map_set
            ]
        self._num_samples = len(self._file_paths)
        log.info(f"SCDataset [{split}]: {self._num_samples} scenarios"
                 f" (maps: {map_names if map_names else 'all'})")
        super().__init__(transform=transform, pre_transform=None, pre_filter=None)

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        path = self._file_paths[idx]
        scenario_id = Path(path).stem

        with h5py.File(path, "r") as f:
            map_name = f.attrs["map_name"]
            if isinstance(map_name, bytes):
                map_name = map_name.decode("utf-8")

            # Global unit data
            unit_tag = f["unit_data"]["global"]["unit_tag"][:]  # (N,)
            unit_owner = f["unit_data"]["global"]["unit_owner"][:]  # (N,)

            # Repeated unit data
            rep = f["unit_data"]["repeated"]
            is_alive = rep["is_alive"][:].astype(bool)  # (T, N)
            coordinate = rep["coordinate"][:].astype(np.float32)  # (T, N, 3)
            heading = rep["heading"][:].astype(np.float32)  # (T, N)
            unit_type = rep["unit_type"][:].astype(np.int64)  # (T, N)
            radius = rep["radius"][:].astype(np.float32)  # (T, N)
            visible_status = rep["visible_status"][:].astype(np.uint8)  # (T, N)
            is_flying = rep["is_flying"][:].astype(bool)  # (T, N)
            is_burrowed = rep["is_burrowed"][:].astype(bool)  # (T, N)
            is_carried = rep["is_carried"][:].astype(bool)  # (T, N)
            health = rep["health"][:].astype(np.float32)  # (T, N)
            health_max = rep["health_max"][:].astype(np.float32)  # (T, N)
            shield = rep["shield"][:].astype(np.float32)  # (T, N)
            energy = rep["energy"][:].astype(np.float32)  # (T, N)

            # Dynamic map data: creep at current frame
            map_frame_skip = int(f.attrs["map_frame_skip"])
            creep_all = f["map_data"]["dynamic"]["creep"][:]  # (T_map, H, W) bool
            creep_idx = min(CURRENT_FRAME_IDX // map_frame_skip, creep_all.shape[0] - 1)
            creep = creep_all[creep_idx].astype(np.float32)  # (H, W) float32, 0/1
            # Pad to fixed 200x200 so PyG can batch across different map sizes
            H_c, W_c = creep.shape
            creep = np.pad(creep, ((0, 200 - H_c), (0, 200 - W_c)), constant_values=0.0)

        T, N = is_alive.shape

        # Filter: keep units alive at any timestep
        ever_alive = is_alive.any(axis=0)  # (N,)
        keep_idx = np.where(ever_alive)[0]

        if len(keep_idx) == 0:
            # Edge case: no alive units — return minimal valid data
            keep_idx = np.array([0])

        # Slice to kept units
        is_alive = is_alive[:, keep_idx]  # (T, N')
        coordinate = coordinate[:, keep_idx]  # (T, N', 3)
        heading = heading[:, keep_idx]  # (T, N')
        unit_type = unit_type[:, keep_idx]  # (T, N')
        radius = radius[:, keep_idx]  # (T, N')
        visible_status = visible_status[:, keep_idx]  # (T, N')
        is_flying = is_flying[:, keep_idx]  # (T, N')
        is_burrowed = is_burrowed[:, keep_idx]  # (T, N')
        is_carried = is_carried[:, keep_idx]  # (T, N')
        health = health[:, keep_idx]  # (T, N')
        health_max = health_max[:, keep_idx]  # (T, N')
        shield = shield[:, keep_idx]  # (T, N')
        energy = energy[:, keep_idx]  # (T, N')
        unit_tag = unit_tag[keep_idx]  # (N',)
        unit_owner = unit_owner[keep_idx]  # (N',)
        N = len(keep_idx)

        # Transpose to (N, T, ...) to match Waymo convention
        valid_mask = torch.from_numpy(is_alive.T)  # (N, T)
        position = torch.from_numpy(coordinate.transpose(1, 0, 2))  # (N, T, 3)
        heading_t = torch.from_numpy(heading.T)  # (N, T)
        visible_status_t = torch.from_numpy(visible_status.T)  # (N, T)
        owner_t = torch.from_numpy(unit_owner.astype(np.int64))  # (N,)

        # Agent type: use type at current_frame_idx, remap to contiguous indices
        type_at_current = unit_type[min(CURRENT_FRAME_IDX, T - 1)]  # (N,)
        agent_type = torch.from_numpy(remap_unit_type(type_at_current)).to(torch.long)

        # Radius at current_frame_idx
        r = radius[min(CURRENT_FRAME_IDX, T - 1)]  # (N,)
        radius_t = torch.from_numpy(r).float()  # (N,)

        # Unit state at current frame: 0=grounded, 1=flying, 2=burrowed, 3=carried
        t_state = min(CURRENT_FRAME_IDX, T - 1)
        f, b, c = is_flying[t_state], is_burrowed[t_state], is_carried[t_state]
        assert not np.any(
            (f.astype(int) + b.astype(int) + c.astype(int)) > 1
        ), "is_flying, is_burrowed, is_carried must be mutually exclusive"
        unit_state = np.zeros(N, dtype=np.int64)  # default 0 = grounded
        unit_state[f] = 1
        unit_state[b] = 2
        unit_state[c] = 3
        unit_state_t = torch.from_numpy(unit_state)  # (N,)

        # Unit vitals at current frame: [health_frac, shield_norm, energy_norm]
        t_v = min(CURRENT_FRAME_IDX, T - 1)
        h, h_max = health[t_v], health_max[t_v]
        health_frac = np.divide(h, h_max, out=np.zeros_like(h), where=h_max > 0)
        shield_norm = shield[t_v] / 1000.0
        energy_norm = energy[t_v] / 200.0
        unit_vitals = torch.from_numpy(
            np.stack([health_frac, shield_norm, energy_norm], axis=-1)
        ).float()  # (N, 3)

        # Role: all False (no ego concept)
        role = torch.zeros(N, 3, dtype=torch.bool)

        # Agent ID
        agent_id = torch.from_numpy(unit_tag.astype(np.int64))

        # Pad or truncate to TOTAL_FRAMES if needed
        if T < TOTAL_FRAMES:
            pad = TOTAL_FRAMES - T
            valid_mask = torch.cat([valid_mask, torch.zeros(N, pad, dtype=torch.bool)], dim=1)
            position = torch.cat([position, torch.zeros(N, pad, 3)], dim=1)
            heading_t = torch.cat([heading_t, torch.zeros(N, pad)], dim=1)
            visible_status_t = torch.cat(
                [visible_status_t, torch.zeros(N, pad, dtype=torch.uint8)], dim=1
            )
        elif T > TOTAL_FRAMES:
            valid_mask = valid_mask[:, :TOTAL_FRAMES]
            position = position[:, :TOTAL_FRAMES]
            heading_t = heading_t[:, :TOTAL_FRAMES]
            visible_status_t = visible_status_t[:, :TOTAL_FRAMES]

        data = {
            "scenario_id": scenario_id,
            "map_name": map_name,
            "creep": torch.from_numpy(creep),  # (H, W) float32
            "agent": {
                "num_nodes": N,
                "valid_mask": valid_mask,
                "position": position,
                "heading": heading_t,
                "id": agent_id,
                "type": agent_type,
                "radius": radius_t,
                "role": role,
                "owner": owner_t,
                "visible_status": visible_status_t,
                "unit_state": unit_state_t,
                "unit_vitals": unit_vitals,
            },
        }
        return data
