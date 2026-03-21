"""StarCraft token processor.

Unified motion dictionary, center+heading matching, no map tokenization.
"""

import pickle
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import transform_to_global, transform_to_local, wrap_angle


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
        centers = torch.tensor(data["cluster_centers"], dtype=torch.float32)
        # centers: [n_token, 9, 3] where 3 = [rel_x, rel_y, rel_heading]
        self.register_buffer("agent_token_all", centers, persistent=False)
        # endpoint: [n_token, 3] — last frame of each token
        self.register_buffer(
            "agent_token_endpoint", centers[:, -1].contiguous(), persistent=False
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

        # Token endpoint for matching: [n_token, 2] (xy only)
        token_endpoint_xy = self.agent_token_endpoint[:, :2]  # [n_token, 2]
        token_endpoint_heading = self.agent_token_endpoint[:, 2]  # [n_token]

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent], all False for SC
            "token_agent_shape": data["agent"]["shape"][:, :2],  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            # Token vocabulary (shared across all agents/types)
            "token_traj_all": self.agent_token_all,  # [n_token, 9, 3]
            "token_traj": self.agent_token_endpoint,  # [n_token, 3]
            "trajectory_token": self.agent_token_endpoint,  # [n_token, 3]
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
        token_dict = self._match_agent_token_center(
            valid=valid,
            pos=pos,
            heading=heading,
            token_endpoint_xy=token_endpoint_xy,
            token_endpoint_heading=token_endpoint_heading,
        )
        tokenized_agent.update(token_dict)
        return tokenized_agent

    def _match_agent_token_center(
        self,
        valid: Tensor,  # [n_agent, T]
        pos: Tensor,  # [n_agent, T, 2]
        heading: Tensor,  # [n_agent, T]
        token_endpoint_xy: Tensor,  # [n_token, 2]
        token_endpoint_heading: Tensor,  # [n_token]
    ) -> Dict[str, Tensor]:
        """Center-based token matching. No polygon contour needed."""
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent, device=valid.device)
        n_token = token_endpoint_xy.shape[0]

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

            gt_pos_i = pos[:, i]  # [n_agent, 2]

            # Transform token endpoints from local to global using prev_pos/prev_head
            # token_endpoint_xy: [n_token, 2] -> broadcast to [n_agent, n_token, 2]
            cos_h = torch.cos(prev_head)  # [n_agent]
            sin_h = torch.sin(prev_head)  # [n_agent]
            # Rotate: global = R(heading) @ local + pos
            tx = token_endpoint_xy[:, 0]  # [n_token]
            ty = token_endpoint_xy[:, 1]  # [n_token]
            # [n_agent, n_token]
            global_x = cos_h.unsqueeze(1) * tx.unsqueeze(0) - sin_h.unsqueeze(1) * ty.unsqueeze(0) + prev_pos[:, 0:1]
            global_y = sin_h.unsqueeze(1) * tx.unsqueeze(0) + cos_h.unsqueeze(1) * ty.unsqueeze(0) + prev_pos[:, 1:2]

            # Distance: [n_agent, n_token]
            dist = (global_x - gt_pos_i[:, 0:1]) ** 2 + (global_y - gt_pos_i[:, 1:2]) ** 2

            # GT: argmin distance
            token_idx_gt = torch.argmin(dist, dim=-1)  # [n_agent]

            # Update prev state using matched token
            matched_global_x = global_x[range_a, token_idx_gt]  # [n_agent]
            matched_global_y = global_y[range_a, token_idx_gt]  # [n_agent]
            matched_heading_delta = token_endpoint_heading[token_idx_gt]  # [n_agent]

            new_pos_gt = torch.stack([matched_global_x, matched_global_y], dim=-1)
            new_head_gt = wrap_angle(prev_head + matched_heading_delta)

            # For invalid agents, fall back to raw GT
            prev_pos = pos[:, i].clone()
            prev_pos[_valid] = new_pos_gt[_valid]
            prev_head = heading[:, i].clone()
            prev_head[_valid] = new_head_gt[_valid]

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
                cos_s = torch.cos(prev_head_sample)
                sin_s = torch.sin(prev_head_sample)
                g_x = cos_s.unsqueeze(1) * tx.unsqueeze(0) - sin_s.unsqueeze(1) * ty.unsqueeze(0) + prev_pos_sample[:, 0:1]
                g_y = sin_s.unsqueeze(1) * tx.unsqueeze(0) + cos_s.unsqueeze(1) * ty.unsqueeze(0) + prev_pos_sample[:, 1:2]

                dist_s = (g_x - gt_pos_i[:, 0:1]) ** 2 + (g_y - gt_pos_i[:, 1:2]) ** 2

                topk_dists, topk_indices = torch.topk(
                    dist_s, num_k, dim=-1, largest=False, sorted=False
                )
                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()
                token_idx_s = topk_indices[range_a, _samples]

                m_gx = g_x[range_a, token_idx_s]
                m_gy = g_y[range_a, token_idx_s]
                m_dh = token_endpoint_heading[token_idx_s]

                new_pos_s = torch.stack([m_gx, m_gy], dim=-1)
                new_head_s = wrap_angle(prev_head_sample + m_dh)

                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample[_valid] = new_pos_s[_valid]
                prev_head_sample = heading[:, i].clone()
                prev_head_sample[_valid] = new_head_s[_valid]

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
