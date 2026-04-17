"""StarCraft token processor.

Unified motion dictionary, contour-based matching, CNN-ready map tokenization.
"""

import os
import pickle
from typing import Dict, List, Tuple

import h5py
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import cal_circular_contour, transform_to_global, transform_to_local, wrap_angle
from src.starcraft.utils.coarse_action_mapping import ABILITY_ID_TO_COARSE_ACTION

_PADDED_SIZE = 200  # all maps padded to 200x200
_TARGET_POS_NORM = 200.0  # normalize relative target pos by map size

# Build a vectorized LUT for coarse action mapping (remap 255→11)
_MAX_ABILITY_ID = max(ABILITY_ID_TO_COARSE_ACTION.keys())
# Remap: original classes 1-10 → 0-9, UNKNOWN (255) → 10. NO_OP (0) excluded
# (supervised only when has_action=True, so NO_OP never appears as a target).
_NUM_ACTION_CLASSES = 11  # 10 action types + UNKNOWN
_COARSE_ACTION_LUT = torch.full((_MAX_ABILITY_ID + 2,), 10, dtype=torch.long)  # default: UNKNOWN→10
for _aid, _label in ABILITY_ID_TO_COARSE_ACTION.items():
    if _label == 0:
        _COARSE_ACTION_LUT[_aid] = 0  # NO_OP placeholder (won't be supervised)
    elif _label == 255:
        _COARSE_ACTION_LUT[_aid] = 10  # UNKNOWN
    else:
        _COARSE_ACTION_LUT[_aid] = _label - 1  # shift 1-10 → 0-9


def _vectorized_coarse_action(ability_ids: Tensor) -> Tensor:
    """Map raw ability_id tensor to coarse action labels (0-10, NO_OP excluded)."""
    clamped = ability_ids.clamp(0, _MAX_ABILITY_ID + 1)
    return _COARSE_ACTION_LUT.to(clamped.device)[clamped]
_PATCH_STRIDE = 8   # CNN stride-8 downsampling (3 layers) → 25x25 grid
_GRID_SIZE = _PADDED_SIZE // _PATCH_STRIDE  # 25

# --- Perspective-based agent filtering ---
# Keys with agent dimension (dim-0 = n_agent) that must be filtered by keep_mask.
_AGENT_DIM0_KEYS = frozenset({
    "type", "owner", "owner_idx", "ego_mask", "unit_state", "batch",
    "token_agent_shape", "unit_props",
    "valid_mask", "gt_idx", "sampled_idx", "sampled_pos", "sampled_heading",
    "gt_pos", "gt_heading", "gt_pos_raw", "gt_head_raw", "gt_valid_raw",
    "visible_status", "coarse_action", "has_action", "has_target_pos",
    "rel_target_pos",
    "gt_z_raw",  # only present during eval
    "is_observer",  # only present after filter_agents_for_perspective
})
# Keys shared across all agents (vocab tensors, scalars) — passed through unfiltered.
_SHARED_KEYS = frozenset({
    "num_graphs", "token_traj_all", "token_traj", "trajectory_token",
    "player_start_loc",
    "observer_start_loc",  # only present after filter_agents_for_perspective
})


_OPPONENT_KEEP_MODES = ("visible_now", "visible_ever", "all")


def filter_agents_for_perspective(
    tokenized_agent: Dict[str, Tensor],
    train_mask: Tensor,
    observer_player: int,
    min_future_visible: int = 8,
    opponent_keep_mode: str = "visible_now",
) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor]:
    """Filter agents to observer's perspective and build per-role train masks.

    ``opponent_keep_mode`` controls which opponent units survive the filter:

    - ``"visible_now"`` (default, used for training and open-loop validation):
      keep only opponents visible to the observer at the current frame
      (token step 1 = frame 16). Historical default.
    - ``"visible_ever"``: keep opponents visible at the current frame *or* at
      any future step (token steps 1..17). Permanently-unseen opponents are
      dropped. Intended for closed-loop teacher-forcing eval so late-observed
      opponents can enter attention when the observer would spot them.
    - ``"all"``: keep every opponent regardless of visibility. Debug /
      completeness option; leaves permanently-unseen opponents as zero-edge
      nodes after fog-of-war gating.

    Remaps ``owner_idx`` to 0=observer, 1=opponent, 2=neutral.  Zeros out
    intent fields for non-observer units.

    Returns:
        filtered_agent: tokenized_agent dict with non-kept agents removed.
        observer_train_mask: [n_filtered] bool — observer units eligible for
            trajectory + intent loss.
        opponent_train_mask: [n_filtered] bool — visible opponent units eligible
            for trajectory-only loss.
        keep_mask: [n_agent_original] bool — maps filtered indices back to
            original agent ordering (needed for validation target alignment).
        vis_to_obs: [n_filtered, 18] bool — per-step visibility of each kept
            agent to the observer. Observer's own units and neutrals are
            always True; opponents follow the raw ``visible_status`` decoding.
            Intended for fog-of-war gating in teacher-force rollout modes.
    """
    if opponent_keep_mode not in _OPPONENT_KEEP_MODES:
        raise ValueError(
            f"Unknown opponent_keep_mode '{opponent_keep_mode}'. "
            f"Valid options: {_OPPONENT_KEEP_MODES}."
        )

    owner = tokenized_agent["owner"]  # [n_agent], values {1, 2, 16}
    visible_status = tokenized_agent["visible_status"]  # [n_agent, 18]

    # Decode observer's visibility across all steps once; reused below.
    if observer_player == 1:
        observer_sees_per_step = (visible_status // 3) == 2
    else:
        observer_sees_per_step = (visible_status % 3) == 2
    # Current-frame visibility is stored at token step 1 (frame 16).
    observer_sees = observer_sees_per_step[:, 1]
    # Current or any future step (tokens 1..17).
    observer_sees_ever = observer_sees_per_step[:, 1:].any(dim=1)

    is_own = owner == observer_player
    is_neutral = owner == 16
    is_opponent = ~is_own & ~is_neutral
    if opponent_keep_mode == "visible_now":
        keep_mask = is_own | is_neutral | (is_opponent & observer_sees)
    elif opponent_keep_mode == "visible_ever":
        keep_mask = is_own | is_neutral | (is_opponent & observer_sees_ever)
    else:  # "all"
        keep_mask = is_own | is_neutral | is_opponent

    # --- Filter all per-agent tensors ---
    out: Dict[str, Tensor] = {}
    for k, v in tokenized_agent.items():
        if k in _SHARED_KEYS:
            out[k] = v
        elif k in _AGENT_DIM0_KEYS:
            out[k] = v[keep_mask]
        else:
            raise ValueError(
                f"Unknown tokenized_agent key '{k}' — "
                "add to _AGENT_DIM0_KEYS or _SHARED_KEYS in sc_token_processor.py"
            )

    # --- Remap owner_idx: 0=observer, 1=opponent, 2=neutral ---
    out["owner_idx"] = _remap_owner_perspective(out["owner"], observer_player)

    # --- is_observer flag for intent gating ---
    is_observer = out["owner"] == observer_player

    # --- Observer start location for concept attention ---
    obs_start_idx = 0 if observer_player == 1 else 1
    out["observer_start_loc"] = out["player_start_loc"][:, obs_start_idx]  # [B, 2]
    out["is_observer"] = is_observer

    # --- Zero out intent for non-observer (opponent + neutral) units ---
    non_obs = ~is_observer
    if non_obs.any():
        out["coarse_action"] = out["coarse_action"].clone()
        out["has_action"] = out["has_action"].clone()
        out["rel_target_pos"] = out["rel_target_pos"].clone()
        out["has_target_pos"] = out["has_target_pos"].clone()
        out["coarse_action"][non_obs] = 0
        out["has_action"][non_obs] = False
        out["rel_target_pos"][non_obs] = 0.0
        out["has_target_pos"][non_obs] = False

    # --- Build per-role train masks ---
    filtered_train_mask = train_mask[keep_mask]
    is_neutral_filtered = out["owner"] == 16
    observer_train_mask = filtered_train_mask & is_observer

    # Opponent train mask: additionally require future visibility >= threshold
    future_visible_count = observer_sees_per_step[keep_mask, 2:].sum(-1)  # [n_filtered]
    opponent_train_mask = (
        filtered_train_mask
        & ~is_observer
        & ~is_neutral_filtered
        & (future_visible_count >= min_future_visible)
    )

    # --- Per-step visibility to observer (for fog-of-war gating) ---
    # Reuse the decoded per-step visibility from before the filter and
    # restrict to kept agents; observer/neutral rows are forced True below.
    vis_to_obs = observer_sees_per_step[keep_mask]
    vis_to_obs = (
        vis_to_obs
        | is_observer.unsqueeze(1)
        | is_neutral_filtered.unsqueeze(1)
    )

    # --- Debug assertions ---
    assert out["batch"].max() < out["num_graphs"], "batch index out of range after filtering"
    assert out["owner"].shape[0] == out["valid_mask"].shape[0], "agent count mismatch"
    num_graphs = out["num_graphs"]
    for g in range(num_graphs):
        assert (out["batch"][is_observer] == g).any(), (
            f"graph {g} lost all observer units after filtering"
        )

    return out, observer_train_mask, opponent_train_mask, keep_mask, vis_to_obs


def _remap_owner_perspective(owner: Tensor, observer_player: int) -> Tensor:
    """Remap raw owner to perspective-relative indices: 0=observer, 1=opponent, 2=neutral."""
    idx = torch.full_like(owner, 2)  # default: neutral
    idx[owner == observer_player] = 0  # observer
    opponent_player = 2 if observer_player == 1 else 1
    idx[owner == opponent_player] = 1  # opponent
    return idx


class SCTokenProcessor(torch.nn.Module):

    def __init__(
        self,
        motion_dict_file: str,
        map_data_dir: str,
        agent_token_sampling: DictConfig,
    ) -> None:
        super().__init__()
        self.agent_token_sampling = agent_token_sampling
        self.map_data_dir = map_data_dir
        self._map_cache: Dict[str, Dict[str, Tensor]] = {}
        self.shift = 8
        self.current_frame_idx = 16
        self.dt = 1.0 / 16.0  # 16 fps

        self._init_agent_token(motion_dict_file)
        self.n_token_agent = self.agent_token_all.shape[0]

    def _init_agent_token(self, path: str) -> None:
        data = pickle.load(open(path, "rb"))
        contours = torch.tensor(data["cluster_centers"], dtype=torch.float32)
        # contours: [n_token, 9, 4, 2] where 4 = corners (LF, RF, RB, LB), 2 = [x, y]
        assert contours.ndim == 4 and contours.shape[-2:] == (4, 2), (
            f"Expected motion dict shape (n_token, 9, 4, 2), got {contours.shape}. "
            "The dictionary file may still be in the old center-based (n_token, 9, 3) format."
        )
        self.register_buffer("agent_token_all", contours, persistent=False)
        # endpoint contour: [n_token, 4, 2] — last frame of each token
        self.register_buffer(
            "agent_token_endpoint", contours[:, -1].contiguous(), persistent=False
        )

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self._tokenize_map(data)
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_map, tokenized_agent

    # ------------------------------------------------------------------
    # Map tokenization
    # ------------------------------------------------------------------

    def _load_and_preprocess_map(self, map_name: str) -> Dict[str, Tensor]:
        """Load static map HDF5, preprocess, and cache. Only 7 maps total."""
        if map_name in self._map_cache:
            return self._map_cache[map_name]

        path = os.path.join(self.map_data_dir, f"{map_name}.h5")
        with h5py.File(path, "r") as f:
            pathing = torch.from_numpy(f["pathing_grid"][:].astype("float32"))  # [H, W]
            height = torch.from_numpy(f["height_map"][:].astype("float32"))     # [H, W]

        # Invert pathing: raw SC2 API uses 0 = walkable, 1 = blocked.
        # We flip to 1 = walkable, 0 = blocked so padding (0) = blocked.
        pathing = 1.0 - pathing
        # Height normalized to [-1, 1] before padding so pad=0 is neutral midpoint.
        height = height / 255.0 * 2.0 - 1.0
        grid = torch.stack([pathing, height], dim=0)  # [2, H, W]
        H, W = grid.shape[1], grid.shape[2]

        # Valid mask before padding (marks original area)
        valid_2d = torch.ones(H, W, dtype=torch.bool)

        # Pad: pathing 0 = blocked, height 0 = neutral midpoint
        grid = F.pad(grid, (0, _PADDED_SIZE - W, 0, _PADDED_SIZE - H), value=0.0)

        # Normalize pathing channel to [-1, 1]; height already in [-1, 1]
        grid[0] = grid[0] * 2.0 - 1.0
        valid_2d = F.pad(valid_2d, (0, _PADDED_SIZE - W, 0, _PADDED_SIZE - H), value=False)

        # Per-patch valid mask (25x25): valid if any cell in the 8x8 patch is in original area
        valid_patches = valid_2d.unfold(0, _PATCH_STRIDE, _PATCH_STRIDE).unfold(
            1, _PATCH_STRIDE, _PATCH_STRIDE
        )  # [25, 25, 8, 8]
        valid_mask = valid_patches.reshape(_GRID_SIZE * _GRID_SIZE, -1).any(dim=1)  # [625]

        # Patch center positions in game coordinates (col→X, row→Y flipped:
        # row 0 = top of map = high Y in game coords)
        row_idx = torch.arange(_GRID_SIZE, dtype=torch.float32).unsqueeze(1).expand(_GRID_SIZE, _GRID_SIZE)
        col_idx = torch.arange(_GRID_SIZE, dtype=torch.float32).unsqueeze(0).expand(_GRID_SIZE, _GRID_SIZE)
        position = torch.stack([
            col_idx.reshape(-1) * _PATCH_STRIDE + _PATCH_STRIDE / 2.0,  # X
            H - (row_idx.reshape(-1) * _PATCH_STRIDE + _PATCH_STRIDE / 2.0),  # Y
        ], dim=-1)  # [625, 2]

        result = {"map_grid": grid, "position": position, "valid_mask": valid_mask}
        self._map_cache[map_name] = result
        return result

    def _tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:
        """Load and batch static map data for all scenarios in the batch."""
        device = data["agent"]["position"].device

        # data["map_name"] is a list of strings after PyG batching
        map_names: List[str] = data["map_name"]
        if isinstance(map_names, str):
            map_names = [map_names]

        batch_size = len(map_names)
        n_patches = _GRID_SIZE * _GRID_SIZE  # 625

        grids = []
        positions = []
        valid_masks = []
        batch_indices = []

        # data["creep"] is (B*200, 200) after PyG concatenation (pre-padded in dataset)
        # Move to CPU for cat with cached static grids; entire dict moves to device at return
        creep_batch = data["creep"].cpu().view(batch_size, _PADDED_SIZE, _PADDED_SIZE)  # [B, 200, 200]

        for i, name in enumerate(map_names):
            m = self._load_and_preprocess_map(name)
            static_grid = m["map_grid"]  # [2, 200, 200]
            # Concatenate creep as 3rd channel (already 0/1, padded with 0),
            # then normalize to [-1, 1]
            creep_ch = creep_batch[i].unsqueeze(0) * 2.0 - 1.0
            grids.append(torch.cat([static_grid, creep_ch], dim=0))  # [3, 200, 200]
            positions.append(m["position"])
            valid_masks.append(m["valid_mask"])
            batch_indices.append(torch.full((n_patches,), i, dtype=torch.long))

        return {
            "map_grid": torch.stack(grids, dim=0).to(device),          # [B, 3, 200, 200]
            "position": torch.cat(positions, dim=0).to(device),        # [B*625, 2]
            "valid_mask": torch.cat(valid_masks, dim=0).to(device),    # [B*625]
            "batch": torch.cat(batch_indices, dim=0).to(device),       # [B*625]
        }

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        valid = data["agent"]["valid_mask"].clone()  # [n_agent, T]
        heading = data["agent"]["heading"].clone()  # [n_agent, T]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, T, 2]

        # Clean heading discontinuities
        heading = self._clean_heading(valid, heading)

        # Stationary-fill extrapolation (no velocity needed)
        valid, pos, heading = self._extrapolate_stationary(valid, pos, heading)

        radius = data["agent"]["radius"]  # [n_agent]

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent], all False for SC
            "token_agent_shape": radius.unsqueeze(-1),  # [n_agent, 1]
            "batch": data["agent"]["batch"],
            # Ownership and visibility for fog-of-war edge filtering
            "owner": data["agent"]["owner"],  # [n_agent]
            "owner_idx": self._remap_owner(data["agent"]["owner"]),  # [n_agent] 0=P1,1=P2,2=neutral
            "unit_state": data["agent"]["unit_state"],  # [n_agent]
            "unit_props": torch.cat([
                radius.unsqueeze(-1),            # [n_agent, 1]
                data["agent"]["unit_vitals"],     # [n_agent, 3] health, shield, energy
            ], dim=-1),  # [n_agent, 4]
            "visible_status": data["agent"]["visible_status"][:, self.shift :: self.shift],  # [n_agent, 18]
            "player_start_loc": data["player_start_loc"].view(data.num_graphs, 2, 2),  # [B, 2, 2]
            # Token vocabulary (shared across all agents/types)
            "token_traj_all": self.agent_token_all,  # [n_token, 9, 4, 2]
            "token_traj": self.agent_token_endpoint,  # [n_token, 4, 2]
            "trajectory_token": self.agent_token_endpoint.flatten(-2, -1),  # [n_token, 8]
            # GT at token boundaries: steps {8, 16, 24, ..., 144}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, 18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, 18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, 18]
        }

        # Action & target labels at token boundaries (last frame of each window)
        ability_steps = data["agent"]["ability_id"][:, self.shift :: self.shift]  # [n_agent, 18]
        target_steps = data["agent"]["target_pos"][:, self.shift :: self.shift]  # [n_agent, 18, 2]
        unit_pos_steps = pos[:, self.shift :: self.shift]  # [n_agent, 18, 2]

        tokenized_agent["coarse_action"] = _vectorized_coarse_action(ability_steps)  # [n_agent, 18] 0-10
        tokenized_agent["has_action"] = (ability_steps != 0)  # [n_agent, 18]
        tokenized_agent["has_target_pos"] = (target_steps.abs().sum(-1) > 0)  # [n_agent, 18]
        rel_target = (target_steps - unit_pos_steps) / _TARGET_POS_NORM  # [n_agent, 18, 2]
        rel_target[~tokenized_agent["has_target_pos"]] = 0.0
        tokenized_agent["rel_target_pos"] = rel_target  # [n_agent, 18, 2]

        if not self.training:
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][
                :, self.current_frame_idx, 2
            ]

        # Match tokens
        token_dict = self._match_agent_token_contour(
            valid=valid,
            pos=pos,
            heading=heading,
            token_traj=self.agent_token_endpoint,  # [n_token, 4, 2]
        )
        tokenized_agent.update(token_dict)
        return tokenized_agent

    def _match_agent_token_contour(
        self,
        valid: Tensor,  # [n_agent, T]
        pos: Tensor,  # [n_agent, T, 2]
        heading: Tensor,  # [n_agent, T]
        token_traj: Tensor,  # [n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        """Contour-based token matching using circular contour (radius=0.5)."""
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent, device=valid.device)
        n_token = token_traj.shape[0]

        prev_pos = pos[:, 0]  # [n_agent, 2]
        prev_head = heading[:, 0]  # [n_agent]
        prev_pos_sample = pos[:, 0].clone()
        prev_head_sample = heading[:, 0].clone()

        out = {
            "valid_mask": [],
            "gt_idx": [],
            "gt_pos": [],
            "gt_heading": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):
            _valid = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid = ~_valid
            out["valid_mask"].append(_valid)

            # GT contour: [n_agent, 4, 2] in global coord (circular, radius=0.5)
            gt_contour = cal_circular_contour(pos[:, i], heading[:, i])
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # Expand shared vocab for transform_to_global
            token_expanded = token_traj.unsqueeze(0).expand(n_agent, -1, -1, -1)
            # [n_agent, n_token, 4, 2]
            token_world_gt = transform_to_global(
                pos_local=token_expanded.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,
                head_now=prev_head,
            )[0].view(n_agent, n_token, 4, 2)

            # Match: sum L2 over corners
            token_idx_gt = torch.argmin(
                torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1), dim=-1
            )  # [n_agent]

            # Extract pos/head from matched contour
            token_contour_gt = token_world_gt[range_a, token_idx_gt]  # [n_agent, 4, 2]

            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 2]  # front - back
            prev_head[_valid] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid]
            prev_pos = pos[:, i].clone()
            prev_pos[_valid] = token_contour_gt.mean(1)[_valid]

            out["gt_idx"].append(token_idx_gt)
            out["gt_pos"].append(prev_pos.masked_fill(_invalid.unsqueeze(1), 0))
            out["gt_heading"].append(prev_head.masked_fill(_invalid, 0))

            # Sampled rollout
            if num_k == 1:
                out["sampled_idx"].append(out["gt_idx"][-1])
                out["sampled_pos"].append(out["gt_pos"][-1])
                out["sampled_heading"].append(out["gt_heading"][-1])
            else:
                # Transform using sampled state
                token_world_sample = transform_to_global(
                    pos_local=token_expanded.flatten(1, 2),
                    head_local=None,
                    pos_now=prev_pos_sample,
                    head_now=prev_head_sample,
                )[0].view(n_agent, n_token, 4, 2)

                # dist: [n_agent, n_token]
                dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
                topk_dists, topk_indices = torch.topk(
                    dist, num_k, dim=-1, largest=False, sorted=False
                )

                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()
                token_idx_s = topk_indices[range_a, _samples]
                token_contour_s = token_world_sample[range_a, token_idx_s]

                prev_head_sample = heading[:, i].clone()
                dxy = token_contour_s[:, 0] - token_contour_s[:, 2]  # front - back
                prev_head_sample[_valid] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid]
                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample[_valid] = token_contour_s.mean(1)[_valid]

                out["sampled_idx"].append(token_idx_s)
                out["sampled_pos"].append(
                    prev_pos_sample.masked_fill(_invalid.unsqueeze(1), 0)
                )
                out["sampled_heading"].append(
                    prev_head_sample.masked_fill(_invalid, 0)
                )

        return {k: torch.stack(v, dim=1) for k, v in out.items()}

    def _extrapolate_stationary(
        self,
        valid: Tensor,  # [n_agent, T]
        pos: Tensor,  # [n_agent, T, 2]
        heading: Tensor,  # [n_agent, T]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Stationary fill: copy first-alive state backwards to previous token boundary."""
        first_valid_step = torch.max(valid, dim=1).indices  # [n_agent]

        for i, t in enumerate(first_valid_step):
            t = t.item()
            n_fill = t % self.shift
            if t == self.current_frame_idx and not valid[i, self.current_frame_idx - self.shift]:
                n_fill = self.shift

            if n_fill > 0:
                valid[i, t - n_fill : t] = True
                pos[i, t - n_fill : t] = pos[i, t]
                heading[i, t - n_fill : t] = heading[i, t]

        return valid, pos, heading

    @staticmethod
    def _remap_owner(owner: Tensor) -> Tensor:
        """Remap raw owner values (1=P1, 2=P2, 16=neutral) to contiguous indices (0, 1, 2)."""
        idx = torch.zeros_like(owner)
        idx[owner == 2] = 1
        idx[owner == 16] = 2
        return idx

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        """Fix heading discontinuities (>1.5 rad jumps)."""
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading
