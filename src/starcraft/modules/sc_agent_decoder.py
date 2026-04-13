"""StarCraft agent decoder.

Map-to-agent + agent-to-agent attention. Unified token embedding. Contour-based rollout.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, transform_to_global, weight_init, wrap_angle
from src.starcraft.layers.concept_attention import ConceptAttentionLayer
from src.starcraft.utils.sc_rollout import sample_next_token_traj_contour
from src.starcraft.utils.unit_type_map import NUM_UNIT_TYPES


class SCAgentDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        num_concepts: int = 16,
        use_aux_loss: bool = True,
        use_action_target_input: bool = False,
        closed_loop_oracle_intent_input: bool = False,
        num_action_classes: int = 11,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.num_concepts = num_concepts
        self.shift = 8
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pl2a = 3  # [distance, angle, rel_orient]
        input_dim_r_a2a = 3
        input_dim_token = 8  # [4 corners * 2 coords], flattened endpoint contour

        self.type_a_emb = nn.Embedding(NUM_UNIT_TYPES, hidden_dim)
        self.unit_state_emb = nn.Embedding(4, hidden_dim)  # grounded/flying/burrowed/carried
        self.owner_emb = nn.Embedding(3, hidden_dim)  # 0=P1, 1=P2, 2=neutral
        self.unit_props_emb = MLPLayer(4, hidden_dim, hidden_dim)  # radius, health, shield, energy

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
        self.r_pl2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2a,
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
        self.pl2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
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
        self.cross_player_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        if num_concepts > 0:
            self.concept_pos_emb = FourierEmbedding(
                input_dim=3, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
            )
            self.concept_attn_layers = nn.ModuleList(
                [
                    ConceptAttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        num_concepts=num_concepts,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )

        # Auxiliary prediction heads: action & target
        self.use_aux_loss = use_aux_loss
        self.num_action_classes = num_action_classes
        if use_aux_loss:
            self.has_action_head = MLPLayer(hidden_dim, hidden_dim, 1)
            self.has_target_pos_head = MLPLayer(hidden_dim, hidden_dim, 1)
            self.action_class_head = MLPLayer(hidden_dim, hidden_dim, num_action_classes)
            self.target_pos_head = MLPLayer(hidden_dim, hidden_dim, 2)

        # Optional action/target input embeddings for ablation
        if closed_loop_oracle_intent_input and not use_action_target_input:
            raise ValueError(
                "closed_loop_oracle_intent_input requires use_action_target_input=True "
                "(needs action/target input embedding layers)"
            )
        if use_action_target_input and not use_aux_loss and not closed_loop_oracle_intent_input:
            raise ValueError(
                "use_action_target_input requires use_aux_loss=True when "
                "closed_loop_oracle_intent_input is False "
                "(predicted variant needs auxiliary prediction heads)"
            )
        self.use_action_target_input = use_action_target_input
        self.closed_loop_oracle_intent_input = closed_loop_oracle_intent_input
        if self.use_action_target_input:
            self.action_input_emb = nn.Embedding(num_action_classes, hidden_dim)
            self.target_input_emb = MLPLayer(2, hidden_dim, hidden_dim)
            self.aux_fusion = MLPLayer(hidden_dim * 3, hidden_dim, hidden_dim)

        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        unit_state,  # [n_agent]
        owner_idx,  # [n_agent] 0=P1, 1=P2, 2=neutral
        unit_props,  # [n_agent, 4] (radius, health, shield, energy)
        inference=False,
        prev_action=None,  # [n_agent, n_step] coarse action class
        prev_target_pos=None,  # [n_agent, n_step, 2] relative target pos
        prev_has_action=None,  # [n_agent, n_step] bool
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
        # Symmetry: randomly swap P1/P2 embedding indices during training
        owner_for_emb = owner_idx
        if self.training and torch.rand(1).item() < 0.5:
            owner_for_emb = owner_idx.clone()
            owner_for_emb[owner_idx == 0] = 1
            owner_for_emb[owner_idx == 1] = 0

        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.unit_state_emb(unit_state.long()),
            self.owner_emb(owner_for_emb.long()),
            self.unit_props_emb(unit_props),
        ]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        ).view(-1, n_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if self.use_action_target_input and prev_action is not None:
            action_emb = self.action_input_emb(prev_action.long())  # [n_agent, n_step, hidden_dim]
            # Zero out action embedding for idle steps (has_action=False)
            if prev_has_action is not None:
                action_emb = action_emb * prev_has_action.unsqueeze(-1).float()
            target_emb = self.target_input_emb(prev_target_pos)  # [n_agent, n_step, hidden_dim]
            feat_a = self.aux_fusion(
                torch.cat([feat_a, action_emb, target_emb], dim=-1)
            )

        if inference:
            return feat_a, agent_token_emb, agent_token_emb_all, categorical_embs
        return feat_a

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, mask, owner=None, inference_mask=None):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        # Exclude neutral units from temporal attention
        if owner is not None:
            mask = mask.clone()
            mask[owner == 16] = False

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

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask,
                               owner=None):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        # Exclude neutral units from same-player a2a
        if owner is not None:
            n_step_local = pos_a.shape[1]
            neutral_s = (owner == 16).unsqueeze(0).expand(n_step_local, -1).reshape(-1)
            mask = mask & ~neutral_s

        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]

        # Same-player only: keep edges where src and tgt belong to the same player
        if owner is not None:
            n_step_local = pos_a.shape[1]
            owner_s = owner.unsqueeze(0).expand(n_step_local, -1).reshape(-1)
            same_player = owner_s[edge_index_a2a[0]] == owner_s[edge_index_a2a[1]]
            edge_index_a2a = edge_index_a2a[:, same_player]

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

    def build_map2agent_edge(
        self,
        pos_pl,          # [n_pl, 2]
        orient_pl,       # [n_pl]
        valid_pl,        # [n_pl] bool — patch validity (False for padding)
        pos_a,           # [n_agent, n_step, 2]
        head_a,          # [n_agent, n_step]
        head_vector_a,   # [n_agent, n_step, 2]
        mask,            # [n_agent, n_step]
        batch_s,         # [n_agent*n_step]
        batch_pl,        # [n_pl*n_step]
        owner=None,      # [n_agent] raw ownership (1, 2, 16)
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        # Exclude neutral units from map-to-agent attention
        if owner is not None:
            neutral_s = (owner == 16).unsqueeze(0).expand(n_step, -1).reshape(-1)
            mask_pl2a = mask_pl2a & ~neutral_s
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        valid_pl_rep = valid_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        # Filter: agent must be valid AND patch must be valid (non-padded)
        edge_index_pl2a = edge_index_pl2a[
            :, mask_pl2a[edge_index_pl2a[1]] & valid_pl_rep[edge_index_pl2a[0]]
        ]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def build_cross_player_edge(self, pos_a, head_a, head_vector_a, batch_s, mask,
                                owner, visible_status):
        """Build cross-player edges with fog-of-war. Source=frozen, target=player units.

        Edges: P1↔P2, P1↔Neutral(src), P2↔Neutral(src). Neutrals are source-only.
        Uses step-major layout matching a2a edges.
        """
        n_agent, n_step = pos_a.shape[:2]
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        owner_s = owner.unsqueeze(0).expand(n_step, -1).reshape(-1)
        vis_s = visible_status.transpose(0, 1).reshape(-1).to(pos_s.device)

        edge_index = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index = subgraph(subset=mask_s, edge_index=edge_index)[0]

        src = edge_index[0]
        tgt = edge_index[1]
        src_owner = owner_s[src]
        tgt_owner = owner_s[tgt]

        # Cross-player: different owners, target must be player (not neutral)
        cross = (src_owner != tgt_owner) & (tgt_owner != 16)

        # Fog-of-war: observer (target) must currently see the source unit
        src_vs = vis_s[src]
        p1_state = src_vs // 3
        p2_state = src_vs % 3
        vis_ok = torch.ones(src.shape[0], dtype=torch.bool, device=src.device)
        vis_ok[tgt_owner == 1] = (p1_state[tgt_owner == 1] == 2)
        vis_ok[tgt_owner == 2] = (p2_state[tgt_owner == 2] == 2)

        edge_index_cross = edge_index[:, cross & vis_ok]

        # Edge features: [distance, bearing, heading_diff]
        rel_pos = pos_s[edge_index_cross[0]] - pos_s[edge_index_cross[1]]
        rel_head = wrap_angle(head_s[edge_index_cross[0]] - head_s[edge_index_cross[1]])
        r_cross = torch.stack(
            [
                torch.norm(rel_pos[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_cross[1]],
                    nbr_vector=rel_pos[:, :2],
                ),
                rel_head,
            ],
            dim=-1,
        )
        r_cross = self.r_a2a_emb(continuous_inputs=r_cross, categorical_embs=None)
        return edge_index_cross, r_cross

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
            unit_state=tokenized_agent["unit_state"],
            owner_idx=tokenized_agent["owner_idx"],
            unit_props=tokenized_agent["unit_props"],
            prev_action=tokenized_agent.get("coarse_action"),
            prev_target_pos=tokenized_agent.get("rel_target_pos"),
            prev_has_action=tokenized_agent.get("has_action"),
        )

        # Frozen initial features for cross-player attention (step-major view).
        # feat_a is reassigned (not mutated) in the loop, so this view stays frozen.
        feat_a_initial_s = feat_a.transpose(0, 1).flatten(0, 1)

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a, mask=mask,
            owner=tokenized_agent["owner"],
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
            owner=tokenized_agent["owner"],
        )

        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],
            orient_pl=map_feature["orientation"],
            valid_pl=map_feature["valid_mask"],
            pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
            mask=mask, batch_s=batch_s, batch_pl=batch_pl,
            owner=tokenized_agent["owner"],
        )

        # Cross-player edges (frozen source → player targets)
        edge_index_cross, r_cross = self.build_cross_player_edge(
            pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
            batch_s=batch_s, mask=mask,
            owner=tokenized_agent["owner"],
            visible_status=tokenized_agent["visible_status"],
        )

        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

        # Pre-build concept attention edges (reused across all layers)
        num_graphs = tokenized_agent["num_graphs"]
        if self.num_concepts > 0:
            concept_edge_data = self.concept_attn_layers[0].build_concept_edges(
                owner=tokenized_agent["owner"],
                visible_status=tokenized_agent["visible_status"],
                valid_mask=tokenized_agent["valid_mask"],
                batch=tokenized_agent["batch"],
                n_step=n_step,
                num_graphs=num_graphs,
                pos_a=pos_a,
                player_start_loc=tokenized_agent["player_start_loc"],
                pos_emb=self.concept_pos_emb,
            )

        # Attention layers: temporal + map-to-agent + agent-to-agent + concept
        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pl2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = self.cross_player_attn_layers[i](
                (feat_a_initial_s, feat_a), r_cross, edge_index_cross
            )
            # feat_a is [n_step*n_agent, hidden_dim] (step-major)
            if self.num_concepts > 0:
                feat_a = self.concept_attn_layers[i](feat_a, **concept_edge_data, frozen_feat=feat_a_initial_s)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        next_token_logits = self.token_predict_head(feat_a)

        out = {
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

        if self.use_aux_loss:
            has_action_logits = self.has_action_head(feat_a).squeeze(-1)
            has_target_pos_logits = self.has_target_pos_head(feat_a).squeeze(-1)
            action_class_logits = self.action_class_head(feat_a)
            target_pos_pred = self.target_pos_head(feat_a)
            out.update({
                "has_action_logits": has_action_logits[:, 1:-1],
                "has_target_pos_logits": has_target_pos_logits[:, 1:-1],
                "action_class_logits": action_class_logits[:, 1:-1],
                "target_pos_pred": target_pos_pred[:, 1:-1],
                "gt_has_action": tokenized_agent["has_action"][:, 2:],
                "gt_has_target_pos": tokenized_agent["has_target_pos"][:, 2:],
                "gt_coarse_action": tokenized_agent["coarse_action"][:, 2:],
                "gt_rel_target_pos": tokenized_agent["rel_target_pos"][:, 2:],
            })

        return out

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
                unit_state=tokenized_agent["unit_state"],
                owner_idx=tokenized_agent["owner_idx"],
                unit_props=tokenized_agent["unit_props"],
                inference=True,
                prev_action=tokenized_agent["coarse_action"][:, :step_current_2hz] if self.use_action_target_input else None,
                prev_target_pos=tokenized_agent["rel_target_pos"][:, :step_current_2hz] if self.use_action_target_input else None,
                prev_has_action=tokenized_agent["has_action"][:, :step_current_2hz] if self.use_action_target_input else None,
            )
        )

        # Token data for rollout
        token_traj_all = tokenized_agent["token_traj_all"]  # [n_token, 9, 4, 2]
        token_traj = tokenized_agent["token_traj"]  # [n_token, 4, 2]
        token_agent_shape = tokenized_agent["token_agent_shape"]  # [n_agent, 1] (radius)

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
        if self.use_aux_loss:
            aux_has_action_list = []
            aux_has_target_pos_list = []
            aux_action_list = []
            aux_target_pos_list = []
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
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * s
                        for s in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    owner=tokenized_agent["owner"],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a, head_a=head_a, head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    owner=tokenized_agent["owner"],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],
                orient_pl=map_feature["orientation"],
                valid_pl=map_feature["valid_mask"],
                pos_a=pos_a[:, -hist_step:],
                head_a=head_a[:, -hist_step:],
                head_vector_a=head_vector_a[:, -hist_step:],
                mask=inference_mask[:, -hist_step:],
                batch_s=batch_s,
                batch_pl=batch_pl,
                owner=tokenized_agent["owner"],
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],
                head_a=head_a[:, -hist_step:],
                head_vector_a=head_vector_a[:, -hist_step:],
                batch_s=batch_s,
                mask=inference_mask[:, -hist_step:],
                owner=tokenized_agent["owner"],
            )

            # Cross-player edges and frozen source features (same hist_step window)
            vis_idx_end = min(n_step, 18)
            vis_idx_start = vis_idx_end - hist_step
            edge_index_cross, r_cross = self.build_cross_player_edge(
                pos_a=pos_a[:, -hist_step:],
                head_a=head_a[:, -hist_step:],
                head_vector_a=head_vector_a[:, -hist_step:],
                batch_s=batch_s,
                mask=pred_valid[:, n_step - hist_step:n_step] if t == 0 else inference_mask[:, -hist_step:],
                owner=tokenized_agent["owner"],
                visible_status=tokenized_agent["visible_status"][:, vis_idx_start:vis_idx_end],
            )
            # Frozen source: initial embeddings from feat_a (never overwritten by attention)
            feat_cross_src = feat_a[:, -hist_step:].transpose(0, 1).flatten(0, 1)

            # Build concept attention edges for this rollout step
            if self.num_concepts > 0:
                vis_idx_end = min(n_step, 18)
                vis_idx_start = vis_idx_end - hist_step
                concept_edge_data = self.concept_attn_layers[0].build_concept_edges(
                    owner=tokenized_agent["owner"],
                    visible_status=tokenized_agent["visible_status"][:, vis_idx_start:vis_idx_end],
                    valid_mask=pred_valid[:, n_step - hist_step:n_step],
                    batch=tokenized_agent["batch"],
                    n_step=hist_step,
                    num_graphs=tokenized_agent["num_graphs"],
                    pos_a=pos_a[:, -hist_step:],
                    player_start_loc=tokenized_agent["player_start_loc"],
                    pos_emb=self.concept_pos_emb,
                )

            for i in range(self.num_layers):
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    _feat_map = (
                        map_feature["pt_token"]
                        .unsqueeze(0)
                        .expand(hist_step, -1, -1)
                        .flatten(0, 1)
                    )
                    _feat_temporal = self.pl2a_attn_layers[i](
                        (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                    )
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = self.cross_player_attn_layers[i](
                        (feat_cross_src, _feat_temporal), r_cross, edge_index_cross
                    )
                    # _feat_temporal is [n_step*n_agent, d] (step-major)
                    if self.num_concepts > 0:
                        _feat_temporal = self.concept_attn_layers[i](
                            _feat_temporal, **concept_edge_data, frozen_feat=feat_cross_src
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
                    feat_a_now = self.pl2a_attn_layers[i](
                        (map_feature["pt_token"], feat_a_now), r_pl2a, edge_index_pl2a
                    )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )
                    feat_a_now = self.cross_player_attn_layers[i](
                        (feat_cross_src, feat_a_now), r_cross, edge_index_cross
                    )
                    # feat_a_now is [n_agent, d] (single timestep)
                    if self.num_concepts > 0:
                        feat_a_now = self.concept_attn_layers[i](
                            feat_a_now, **concept_edge_data, frozen_feat=feat_cross_src
                        )
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)

            # Auxiliary predictions at this rollout step
            if self.use_aux_loss:
                aux_has_action_list.append(self.has_action_head(feat_a_now).squeeze(-1))
                aux_has_target_pos_list.append(self.has_target_pos_head(feat_a_now).squeeze(-1))
                aux_action_logits = self.action_class_head(feat_a_now)
                aux_action_list.append(aux_action_logits)
                aux_target_pos_list.append(self.target_pos_head(feat_a_now))

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
                )
            )  # next_token_traj_all: [n_agent, 9, 4, 2] in local coords

            # next_token_action from local endpoint (before global transform)
            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 2]  # front - back
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
                diff_xy_sub = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 2]  # front - back
                pred_head_native[:, t * self.shift : (t + 1) * self.shift] = (
                    torch.atan2(diff_xy_sub[:, :, 1], diff_xy_sub[:, :, 0])
                )

            # Extract next_pos/next_head from global endpoint
            next_pos = token_traj_global[:, -1].mean(dim=1)  # [n_agent, 2]
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 2]  # front - back
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

            # Feed action/target as input for next step
            if self.use_action_target_input:
                if self.closed_loop_oracle_intent_input:
                    # Oracle: use GT intent at step n_step (the position being appended)
                    oracle_action = tokenized_agent["coarse_action"][:, n_step]
                    oracle_has_action = tokenized_agent["has_action"][:, n_step]
                    oracle_target_pos = tokenized_agent["rel_target_pos"][:, n_step]
                    oracle_has_tp = tokenized_agent["has_target_pos"][:, n_step]

                    action_emb_next = self.action_input_emb(oracle_action.long())
                    action_emb_next[~oracle_has_action] = 0.0

                    target_pos = oracle_target_pos.clone()
                    target_pos[~oracle_has_tp] = 0.0
                    target_emb_next = self.target_input_emb(target_pos)
                else:
                    # Predicted: use aux head predictions from current step
                    has_action_pred = (aux_has_action_list[-1] > 0)  # [n_agent]
                    action_pred = aux_action_logits.argmax(-1)  # [n_agent]
                    action_emb_next = self.action_input_emb(action_pred)  # [n_agent, hidden_dim]
                    action_emb_next[~has_action_pred] = 0.0

                    has_tp_pred = (aux_has_target_pos_list[-1] > 0)  # [n_agent]
                    target_pred = aux_target_pos_list[-1].clone()  # [n_agent, 2]
                    target_pred[~has_tp_pred] = 0.0
                    target_emb_next = self.target_input_emb(target_pred)  # [n_agent, hidden_dim]

                feat_a_next = self.aux_fusion(
                    torch.cat([feat_a_next, action_emb_next.unsqueeze(1), target_emb_next.unsqueeze(1)], dim=-1)
                )

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

        if self.use_aux_loss:
            out_dict.update({
                "has_action_logits": torch.stack(aux_has_action_list, dim=1),
                "has_target_pos_logits": torch.stack(aux_has_target_pos_list, dim=1),
                "action_class_logits": torch.stack(aux_action_list, dim=1),
                "target_pos_pred": torch.stack(aux_target_pos_list, dim=1),
                "gt_has_action": tokenized_agent["has_action"][:, 2:],
                "gt_has_target_pos": tokenized_agent["has_target_pos"][:, 2:],
                "gt_coarse_action": tokenized_agent["coarse_action"][:, 2:],
                "gt_rel_target_pos": tokenized_agent["rel_target_pos"][:, 2:],
            })

        if not self.training:
            out_dict["pred_traj_native"] = pred_traj_native
            out_dict["pred_head_native"] = pred_head_native
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)
            out_dict["pred_z_native"] = pred_z.expand(-1, pred_traj_native.shape[1])

        return out_dict
