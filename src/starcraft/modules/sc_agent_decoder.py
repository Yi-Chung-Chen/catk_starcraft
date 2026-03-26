"""StarCraft agent decoder.

No map-to-agent attention. Unified token embedding. Contour-based rollout.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, transform_to_global, weight_init, wrap_angle
from src.starcraft.utils.sc_rollout import sample_next_token_traj_contour
from src.starcraft.utils.unit_type_map import NUM_UNIT_TYPES


class SCAgentDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 8
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_a2a = 3
        input_dim_token = 8  # [4 corners * 2 coords], flattened endpoint contour

        self.type_a_emb = nn.Embedding(NUM_UNIT_TYPES, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)

        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape

        # Unified token embedding (no per-type branching)
        agent_token_emb_all = self.token_emb(trajectory_token)  # [n_token, hidden_dim]
        agent_token_emb = agent_token_emb_all[agent_token_index]  # [n_agent, n_step, hidden_dim]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(n_agent, 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        ).view(-1, n_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return feat_a, agent_token_emb, agent_token_emb_all, categorical_embs
        return feat_a

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, mask, inference_mask=None):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                (edge_index_t[0] - edge_index_t[1]).float(),
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],
            trajectory_token=tokenized_agent["trajectory_token"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
        )

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a, mask=mask
        )

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
            batch_s=batch_s, mask=mask,
        )

        # Attention layers: temporal + agent-to-agent (no map-to-agent)
        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        next_token_logits = self.token_predict_head(feat_a)

        return {
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],
            "pred_pos": tokenized_agent["sampled_pos"],
            "pred_head": tokenized_agent["sampled_heading"],
            "pred_valid": tokenized_agent["valid_mask"],
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],
            "gt_head_raw": tokenized_agent["gt_head_raw"],
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],
            "gt_pos": tokenized_agent["gt_pos"],
            "gt_head": tokenized_agent["gt_heading"],
            "gt_valid": tokenized_agent["valid_mask"],
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, torch.Tensor]:
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_native = self.num_future_steps  # 128
        n_step_future_2hz = n_step_future_native // self.shift  # 16
        step_current_native = self.num_historical_steps - 1  # 16
        step_current_2hz = step_current_native // self.shift  # 2

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()

        feat_a, agent_token_emb, agent_token_emb_all, categorical_embs = (
            self.agent_token_embedding(
                agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
                trajectory_token=tokenized_agent["trajectory_token"],
                pos_a=pos_a,
                head_vector_a=head_vector_a,
                agent_type=tokenized_agent["type"],
                agent_shape=tokenized_agent["shape"],
                inference=True,
            )
        )

        # Token data for rollout
        token_traj_all = tokenized_agent["token_traj_all"]  # [n_token, 9, 4, 2]
        token_traj = tokenized_agent["token_traj"]  # [n_token, 4, 2]
        token_agent_shape = tokenized_agent["token_agent_shape"]  # [n_agent, 2]

        if not self.training:
            pred_traj_native = torch.zeros(
                [n_agent, n_step_future_native, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_native = torch.zeros(
                [n_agent, n_step_future_native], dtype=pos_a.dtype, device=pos_a.device
            )

        pred_valid = tokenized_agent["valid_mask"].clone()
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}

        for t in range(n_step_future_2hz):
            t_now = step_current_2hz - 1 + t
            n_step = t_now + 1

            if t == 0:
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * s
                        for s in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],
                head_a=head_a[:, -hist_step:],
                head_vector_a=head_vector_a[:, -hist_step:],
                batch_s=batch_s,
                mask=inference_mask[:, -hist_step:],
            )

            for i in range(self.num_layers):
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(0, 1)
                    feat_a_now = _feat_temporal[:, -1]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal
                else:
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t, edge_index_t,
                    )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)

            next_token_idx, next_token_traj_all = (
                sample_next_token_traj_contour(
                    token_traj=token_traj,
                    token_traj_all=token_traj_all,
                    sampling_scheme=sampling_scheme,
                    next_token_logits=next_token_logits,
                    pos_now=pos_a[:, t_now],
                    head_now=head_a[:, t_now],
                    pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],
                    head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],
                    valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],
                    token_agent_shape=token_agent_shape,
                )
            )  # next_token_traj_all: [n_agent, 9, 4, 2] in local coords

            # next_token_action from local endpoint (before global transform)
            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )
            )

            # Transform entire trajectory to global at once
            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 9*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],
                head_now=head_a[:, t_now],
            )[0].view(n_agent, 9, 4, 2)

            if not self.training:
                # Native-fps: all substeps at once
                pred_traj_native[:, t * self.shift : (t + 1) * self.shift] = (
                    token_traj_global[:, 1:].mean(2)
                )
                diff_xy_sub = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_native[:, t * self.shift : (t + 1) * self.shift] = (
                    torch.atan2(diff_xy_sub[:, :, 1], diff_xy_sub[:, :, 0])
                )

            # Extract next_pos/next_head from global endpoint
            next_pos = token_traj_global[:, -1].mean(dim=1)  # [n_agent, 2]
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            next_head = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            pred_idx[:, n_step] = next_token_idx
            pred_valid[:, n_step] = pred_valid[:, t_now]

            pos_a = torch.cat([pos_a, next_pos.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, next_head.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [next_head.cos(), next_head.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            agent_token_emb_next = agent_token_emb_all[next_token_idx]  # [n_agent, hidden_dim]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],
            "pred_pos": pos_a,
            "pred_head": head_a,
            "pred_valid": pred_valid,
            "pred_idx": pred_idx,
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],
            "gt_head_raw": tokenized_agent["gt_head_raw"],
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],
            "gt_pos": tokenized_agent["gt_pos"],
            "gt_head": tokenized_agent["gt_heading"],
            "gt_valid": tokenized_agent["valid_mask"],
            "next_token_action": torch.stack(next_token_action_list, dim=1),
        }

        if not self.training:
            out_dict["pred_traj_native"] = pred_traj_native
            out_dict["pred_head_native"] = pred_head_native
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)
            out_dict["pred_z_native"] = pred_z.expand(-1, pred_traj_native.shape[1])

        return out_dict
