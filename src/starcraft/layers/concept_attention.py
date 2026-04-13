"""Q-former-style concept attention for global battlefield awareness.

Two-stage cross-attention via PyG MessagePassing:
  Stage 1 — K shared concept queries aggregate from all visible units (per player).
  Stage 2 — Player-owned units attend to updated concepts (pos on Q via to_q_r).
"""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.utils import weight_init


class ConceptAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        num_concepts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim

        # Shared concept queries — symmetric for both players
        self.concept_queries = nn.Parameter(torch.randn(num_concepts, hidden_dim))

        # Stage 1: concepts ← visible units (pos on K,V via has_pos_emb)
        self.s1_attn = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=True,
            has_pos_emb=True,
        )

        # Stage 2: owned units ← concepts (pos on Q via has_pos_emb_q)
        self.s2_attn = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=True,
            has_pos_emb=False,
            has_pos_emb_q=True,
        )

        self.apply(weight_init)
        # Re-init concept queries after weight_init
        nn.init.normal_(self.concept_queries, std=0.02)

    def forward(
        self,
        feat_a: Tensor,
        edge_index_s1: Tensor,
        edge_index_s2: Tensor,
        r_s1: Tensor,
        pos_q: Tensor,
        neutral_mask: Tensor,
        num_concepts_total: int,
        frozen_feat: Tensor = None,
        s1_is_cross_player: Tensor = None,
    ) -> Tensor:
        """Apply concept attention.

        Args:
            feat_a: Unit features [n_step*n_agent, hidden_dim] (step-major).
            edge_index_s1: Stage 1 edges [2, E1] (unit → concept).
            edge_index_s2: Stage 2 edges [2, E2] (concept → owned unit).
            r_s1: Per-edge pos encoding for Stage 1 [E1, hidden_dim].
            pos_q: Per-node pos encoding for Stage 2 Q [n_step*n_agent, hidden_dim].
            neutral_mask: Boolean mask [n_step*n_agent], True for neutral units.
            num_concepts_total: Total concept nodes (n_step * 2 * B * K).

        Returns:
            Updated feat_a with same shape.
        """
        K = self.num_concepts

        # Expand shared queries to all (timestep, scenario, player) groups
        n_groups = num_concepts_total // K
        concepts = self.concept_queries.unsqueeze(0).expand(n_groups, K, -1)
        concepts = concepts.reshape(num_concepts_total, self.hidden_dim)

        # Stage 1: concepts aggregate from visible units
        # Cross-player sources use frozen initial features to prevent info leakage
        if frozen_feat is not None and s1_is_cross_player is not None:
            combined_src = torch.cat([feat_a, frozen_feat], dim=0)
            edge_index_s1 = edge_index_s1.clone()
            edge_index_s1[0, s1_is_cross_player] += feat_a.shape[0]
            concepts = self.s1_attn((combined_src, concepts), r_s1, edge_index_s1)
        else:
            concepts = self.s1_attn((feat_a, concepts), r_s1, edge_index_s1)

        # Stage 2: owned units attend to concepts (pos on Q)
        feat_a_before = feat_a
        feat_a = self.s2_attn(
            (concepts, feat_a), None, edge_index_s2, r_q=pos_q
        )
        # Restore neutral units — update() gating modifies even zero-input nodes
        feat_a = torch.where(neutral_mask.unsqueeze(-1), feat_a_before, feat_a)

        return feat_a

    def build_concept_edges(
        self,
        owner: Tensor,
        visible_status: Tensor,
        valid_mask: Tensor,
        batch: Tensor,
        n_step: int,
        num_graphs: int,
        pos_a: Tensor,
        player_start_loc: Tensor,
        pos_emb: nn.Module,
    ) -> Dict[str, object]:
        """Build edges and positional encodings for concept attention.

        All outputs are in step-major layout. Called once before the decoder
        layer loop and reused across all layers.

        Args:
            owner: [n_agent] unit ownership (1=P1, 2=P2, 16=neutral).
            visible_status: [n_agent, n_step] uint8 visibility per timestep.
            valid_mask: [n_agent, n_step] bool, True if unit is alive.
            batch: [n_agent] batch index per agent.
            n_step: Number of timesteps.
            num_graphs: Batch size B.
            pos_a: [n_agent, n_step, 2] unit positions.
            player_start_loc: [B, 2, 2] per-player starting locations.

        Returns:
            Dict with keys: edge_index_s1, edge_index_s2, r_s1, pos_q,
            neutral_mask, num_concepts_total.
        """
        K = self.num_concepts
        n_agent = owner.shape[0]
        B = num_graphs
        device = owner.device

        # Alive mask per timestep: [n_step, n_agent]
        alive = valid_mask.T  # [n_step, n_agent]

        # Decode visibility per timestep: [n_step, n_agent]
        vis = visible_status.T  # [n_step, n_agent]
        p1_vis = (vis // 3 == 2)  # P1 can currently see this unit
        p2_vis = (vis % 3 == 2)   # P2 can currently see this unit

        # A unit is "visible" to player P if P sees it OR the unit belongs to P
        # AND the unit is alive at that timestep
        is_p1 = (owner == 1)  # [n_agent]
        is_p2 = (owner == 2)
        vis_to_p1 = (p1_vis | is_p1.unsqueeze(0)) & alive  # [n_step, n_agent]
        vis_to_p2 = (p2_vis | is_p2.unsqueeze(0)) & alive

        # --- Stage 1 edges: visible units → concepts ---
        # Player 1 edges
        t1, a1 = torch.where(vis_to_p1)  # (t_idx, agent_idx) pairs
        unit_flat_1 = t1 * n_agent + a1
        concept_base_1 = t1 * (2 * B * K) + batch[a1] * (2 * K)  # P1 offset = 0
        src_1 = unit_flat_1.repeat_interleave(K)
        dst_1 = (concept_base_1.unsqueeze(1) + torch.arange(K, device=device)).flatten()

        # Player 2 edges
        t2, a2 = torch.where(vis_to_p2)
        unit_flat_2 = t2 * n_agent + a2
        concept_base_2 = t2 * (2 * B * K) + batch[a2] * (2 * K) + K  # P2 offset = K
        src_2 = unit_flat_2.repeat_interleave(K)
        dst_2 = (concept_base_2.unsqueeze(1) + torch.arange(K, device=device)).flatten()

        edge_index_s1 = torch.stack([
            torch.cat([src_1, src_2]),
            torch.cat([dst_1, dst_2]),
        ])

        # Cross-player mask: source unit not owned by the concept's player
        cross_p1 = (owner[a1] != 1).repeat_interleave(K)  # P2/neutral → P1 concepts
        cross_p2 = (owner[a2] != 2).repeat_interleave(K)  # P1/neutral → P2 concepts
        s1_is_cross_player = torch.cat([cross_p1, cross_p2])

        # --- Stage 1 positional encoding (per-edge) ---
        # For P1 edges: pos relative to P1's start; for P2: relative to P2's start
        pos_flat = pos_a.transpose(0, 1).reshape(n_step * n_agent, 2)  # step-major

        p1_start = player_start_loc[batch[a1], 0]  # [n_vis_p1, 2]
        p1_unit_pos = pos_flat[unit_flat_1]         # [n_vis_p1, 2]
        p1_rel = p1_unit_pos - p1_start
        p1_dist = p1_rel.norm(dim=-1, keepdim=True)
        p1_r_input = torch.cat([p1_rel, p1_dist], dim=-1)  # [n_vis_p1, 3]
        p1_r = p1_r_input.repeat_interleave(K, dim=0)      # [n_vis_p1 * K, 3]

        p2_start = player_start_loc[batch[a2], 1]
        p2_unit_pos = pos_flat[unit_flat_2]
        p2_rel = p2_unit_pos - p2_start
        p2_dist = p2_rel.norm(dim=-1, keepdim=True)
        p2_r_input = torch.cat([p2_rel, p2_dist], dim=-1)
        p2_r = p2_r_input.repeat_interleave(K, dim=0)

        r_s1_input = torch.cat([p1_r, p2_r], dim=0)  # [E1, 3]
        r_s1 = pos_emb(continuous_inputs=r_s1_input, categorical_embs=None)

        # --- Stage 2 edges: concepts → owned units (alive only) ---
        # Player 1 owned units alive at each timestep
        p1_alive = is_p1.unsqueeze(0) & alive  # [n_step, n_agent]
        t_p1, a_p1 = torch.where(p1_alive)
        unit_flat_p1 = t_p1 * n_agent + a_p1
        concept_base_p1 = t_p1 * (2 * B * K) + batch[a_p1] * (2 * K)  # P1 offset = 0
        s2_src_1 = (concept_base_p1.unsqueeze(1) + torch.arange(K, device=device)).flatten()
        s2_dst_1 = unit_flat_p1.repeat_interleave(K)

        # Player 2 owned units alive at each timestep
        p2_alive = is_p2.unsqueeze(0) & alive  # [n_step, n_agent]
        t_p2, a_p2 = torch.where(p2_alive)
        unit_flat_p2 = t_p2 * n_agent + a_p2
        concept_base_p2 = t_p2 * (2 * B * K) + batch[a_p2] * (2 * K) + K  # P2 offset
        s2_src_2 = (concept_base_p2.unsqueeze(1) + torch.arange(K, device=device)).flatten()
        s2_dst_2 = unit_flat_p2.repeat_interleave(K)

        edge_index_s2 = torch.stack([
            torch.cat([s2_src_1, s2_src_2]),
            torch.cat([s2_dst_1, s2_dst_2]),
        ])

        # --- Stage 2 positional encoding (per-node, for Q) ---
        # Each unit's pos relative to its own player's start
        # Neutral and dead units get zero pos encoding
        pos_q_raw = torch.zeros(n_step * n_agent, 3, device=device)

        # P1 units
        p1_unit_idx = torch.where(is_p1)[0]  # [n_p1]
        if p1_unit_idx.numel() > 0:
            p1_pos = pos_a[p1_unit_idx]  # [n_p1, n_step, 2]
            p1_s = player_start_loc[batch[p1_unit_idx], 0]  # [n_p1, 2]
            p1_rel_all = p1_pos - p1_s.unsqueeze(1)  # [n_p1, n_step, 2]
            p1_dist_all = p1_rel_all.norm(dim=-1, keepdim=True)  # [n_p1, n_step, 1]
            p1_feat = torch.cat([p1_rel_all, p1_dist_all], dim=-1)  # [n_p1, n_step, 3]
            # Write into step-major layout
            p1_flat_idx = (
                torch.arange(n_step, device=device).unsqueeze(1) * n_agent
                + p1_unit_idx.unsqueeze(0)
            ).flatten()  # [n_step * n_p1]
            pos_q_raw[p1_flat_idx] = p1_feat.transpose(0, 1).reshape(-1, 3)

        # P2 units
        p2_unit_idx = torch.where(is_p2)[0]  # [n_p2]
        if p2_unit_idx.numel() > 0:
            p2_pos = pos_a[p2_unit_idx]
            p2_s = player_start_loc[batch[p2_unit_idx], 1]
            p2_rel_all = p2_pos - p2_s.unsqueeze(1)
            p2_dist_all = p2_rel_all.norm(dim=-1, keepdim=True)
            p2_feat = torch.cat([p2_rel_all, p2_dist_all], dim=-1)
            p2_flat_idx = (
                torch.arange(n_step, device=device).unsqueeze(1) * n_agent
                + p2_unit_idx.unsqueeze(0)
            ).flatten()
            pos_q_raw[p2_flat_idx] = p2_feat.transpose(0, 1).reshape(-1, 3)

        pos_q = pos_emb(continuous_inputs=pos_q_raw, categorical_embs=None)

        # --- Neutral mask ---
        neutral_mask = (owner == 16).unsqueeze(0).expand(n_step, -1).reshape(-1)

        num_concepts_total = n_step * 2 * B * K

        return {
            "edge_index_s1": edge_index_s1,
            "edge_index_s2": edge_index_s2,
            "r_s1": r_s1,
            "pos_q": pos_q,
            "neutral_mask": neutral_mask,
            "num_concepts_total": num_concepts_total,
            "s1_is_cross_player": s1_is_cross_player,
        }
