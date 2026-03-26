"""StarCraft token processor.

Unified motion dictionary, contour-based matching, no map tokenization.
"""

import pickle
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import cal_polygon_contour, transform_to_global, transform_to_local, wrap_angle


class SCTokenProcessor(torch.nn.Module):

    def __init__(
        self,
        motion_dict_file: str,
        agent_token_sampling: DictConfig,
    ) -> None:
        super().__init__()
        self.agent_token_sampling = agent_token_sampling
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
        tokenized_map = self._tokenize_map_dummy(data)
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_map, tokenized_agent

    def _tokenize_map_dummy(self, data: HeteroData) -> Dict[str, Tensor]:
        """Return an empty map dict. Map support is deferred to phase 2."""
        return {}

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        valid = data["agent"]["valid_mask"]  # [n_agent, T]
        heading = data["agent"]["heading"].clone()  # [n_agent, T]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, T, 2]

        # Clean heading discontinuities
        heading = self._clean_heading(valid, heading)

        # Stationary-fill extrapolation (no velocity needed)
        valid, pos, heading = self._extrapolate_stationary(valid, pos, heading)

        agent_shape = data["agent"]["shape"][:, :2]  # [n_agent, 2]

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent], all False for SC
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            # Ownership and visibility for fog-of-war edge filtering
            "owner": data["agent"]["owner"],  # [n_agent]
            "visible_status": data["agent"]["visible_status"][:, self.shift :: self.shift],  # [n_agent, 18]
            # Token vocabulary (shared across all agents/types)
            "token_traj_all": self.agent_token_all,  # [n_token, 9, 4, 2]
            "token_traj": self.agent_token_endpoint,  # [n_token, 4, 2]
            "trajectory_token": self.agent_token_endpoint.flatten(-2, -1),  # [n_token, 8]
            # GT at token boundaries: steps {8, 16, 24, ..., 144}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, 18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, 18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, 18]
        }

        if not self.training:
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][
                :, self.current_frame_idx, 2
            ]

        # Match tokens
        token_dict = self._match_agent_token_contour(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=self.agent_token_endpoint,  # [n_token, 4, 2]
        )
        tokenized_agent.update(token_dict)
        return tokenized_agent

    def _match_agent_token_contour(
        self,
        valid: Tensor,  # [n_agent, T]
        pos: Tensor,  # [n_agent, T, 2]
        heading: Tensor,  # [n_agent, T]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        """Contour-based token matching following SMART pattern."""
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

            # GT contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
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
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
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
                dxy = token_contour_s[:, 0] - token_contour_s[:, 3]
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
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        """Fix heading discontinuities (>1.5 rad jumps)."""
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading
