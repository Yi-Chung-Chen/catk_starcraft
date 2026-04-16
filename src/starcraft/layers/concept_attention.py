"""Observer-only concept attention for global battlefield awareness.

Two-stage cross-attention via PyG MessagePassing:
  Stage 1 — K shared concept queries aggregate from all observer-visible units.
  Stage 2 — All non-neutral units attend to updated concepts (pos on Q).

All position encodings use the observer's start location as reference.
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

        # Shared concept queries
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

        # Stage 2: units ← concepts (pos on Q via has_pos_emb_q)
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
    ) -> Tensor:
        """Apply observer-only concept attention.

        Args:
            feat_a: Unit features [n_step*n_agent, hidden_dim] (step-major).
            edge_index_s1: Stage 1 edges [2, E1] (unit → concept).
            edge_index_s2: Stage 2 edges [2, E2] (concept → unit).
            r_s1: Per-edge pos encoding for Stage 1 [E1, hidden_dim].
            pos_q: Per-node pos encoding for Stage 2 Q [n_step*n_agent, hidden_dim].
            neutral_mask: Boolean mask [n_step*n_agent], True for neutral units.
            num_concepts_total: Total concept nodes (n_step * B * K).

        Returns:
            Updated feat_a with same shape.
        """
        K = self.num_concepts

        # Expand shared queries to all (timestep, scenario) groups
        n_groups = num_concepts_total // K
        concepts = self.concept_queries.unsqueeze(0).expand(n_groups, K, -1)
        concepts = concepts.reshape(num_concepts_total, self.hidden_dim)

        # Stage 1: concepts aggregate from observer-visible units (live features)
        concepts = self.s1_attn((feat_a, concepts), r_s1, edge_index_s1)

        # Stage 2: non-neutral units attend to concepts (pos on Q)
        feat_a_before = feat_a
        feat_a = self.s2_attn(
            (concepts, feat_a), None, edge_index_s2, r_q=pos_q
        )
        # Restore neutral units — update() gating modifies even zero-input nodes
        feat_a = torch.where(neutral_mask.unsqueeze(-1), feat_a_before, feat_a)

        return feat_a

    def build_concept_edges(
        self,
        owner_idx: Tensor,
        valid_mask: Tensor,
        batch: Tensor,
        n_step: int,
        num_graphs: int,
        pos_a: Tensor,
        observer_start_loc: Tensor,
        pos_emb: nn.Module,
    ) -> Dict[str, object]:
        """Build edges and positional encodings for observer-only concept attention.

        All outputs are in step-major layout. Called once before the decoder
        layer loop and reused across all layers.

        Args:
            owner_idx: [n_agent] 0=observer, 1=opponent, 2=neutral.
            valid_mask: [n_agent, n_step] bool, True if unit is alive.
            batch: [n_agent] batch index per agent.
            n_step: Number of timesteps.
            num_graphs: Batch size B.
            pos_a: [n_agent, n_step, 2] unit positions.
            observer_start_loc: [B, 2] observer's starting location.
            pos_emb: FourierEmbedding module for positional encoding.

        Returns:
            Dict with keys: edge_index_s1, edge_index_s2, r_s1, pos_q,
            neutral_mask, num_concepts_total.
        """
        K = self.num_concepts
        n_agent = owner_idx.shape[0]
        B = num_graphs
        device = owner_idx.device

        alive = valid_mask.T  # [n_step, n_agent]

        # All non-neutral alive units are visible to the observer
        # (non-visible opponents already filtered before forward pass)
        is_player = (owner_idx != 2)  # [n_agent]
        vis_to_observer = is_player.unsqueeze(0) & alive  # [n_step, n_agent]

        # --- Stage 1 edges: visible units → observer concepts ---
        t1, a1 = torch.where(vis_to_observer)
        unit_flat_1 = t1 * n_agent + a1
        # Single concept bank: n_step * B * K
        concept_base_1 = t1 * (B * K) + batch[a1] * K
        src_1 = unit_flat_1.repeat_interleave(K)
        dst_1 = (concept_base_1.unsqueeze(1) + torch.arange(K, device=device)).flatten()

        edge_index_s1 = torch.stack([src_1, dst_1])

        # --- Stage 1 positional encoding: all relative to observer's start ---
        pos_flat = pos_a.transpose(0, 1).reshape(n_step * n_agent, 2)  # step-major
        obs_start = observer_start_loc[batch[a1]]  # [n_vis, 2]
        unit_pos = pos_flat[unit_flat_1]  # [n_vis, 2]
        rel = unit_pos - obs_start
        dist = rel.norm(dim=-1, keepdim=True)
        r_input = torch.cat([rel, dist], dim=-1)  # [n_vis, 3]
        r_s1 = pos_emb(
            continuous_inputs=r_input.repeat_interleave(K, dim=0),
            categorical_embs=None,
        )

        # --- Stage 2 edges: observer concepts → all non-neutral alive units ---
        # Same set as Stage 1 (symmetric: same units aggregate and attend)
        t_all, a_all = t1, a1  # reuse
        unit_flat_all = unit_flat_1
        concept_base_all = concept_base_1
        s2_src = (concept_base_all.unsqueeze(1) + torch.arange(K, device=device)).flatten()
        s2_dst = unit_flat_all.repeat_interleave(K)

        edge_index_s2 = torch.stack([s2_src, s2_dst])

        # --- Stage 2 positional encoding: per-node, for Q ---
        pos_q_raw = torch.zeros(n_step * n_agent, 3, device=device)
        player_idx = torch.where(is_player)[0]
        if player_idx.numel() > 0:
            p_pos = pos_a[player_idx]  # [n_player, n_step, 2]
            p_s = observer_start_loc[batch[player_idx]]  # [n_player, 2]
            p_rel = p_pos - p_s.unsqueeze(1)  # [n_player, n_step, 2]
            p_dist = p_rel.norm(dim=-1, keepdim=True)
            p_feat = torch.cat([p_rel, p_dist], dim=-1)  # [n_player, n_step, 3]
            flat_idx = (
                torch.arange(n_step, device=device).unsqueeze(1) * n_agent
                + player_idx.unsqueeze(0)
            ).flatten()
            pos_q_raw[flat_idx] = p_feat.transpose(0, 1).reshape(-1, 3)

        pos_q = pos_emb(continuous_inputs=pos_q_raw, categorical_embs=None)

        # --- Neutral mask ---
        neutral_mask = (owner_idx == 2).unsqueeze(0).expand(n_step, -1).reshape(-1)

        num_concepts_total = n_step * B * K

        return {
            "edge_index_s1": edge_index_s1,
            "edge_index_s2": edge_index_s2,
            "r_s1": r_s1,
            "pos_q": pos_q,
            "neutral_mask": neutral_mask,
            "num_concepts_total": num_concepts_total,
        }
