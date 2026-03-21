"""Verify the StarCraft pipeline end-to-end: load → tokenize → forward → loss."""

import sys
import torch

def main():
    print("=" * 60)
    print("StarCraft Pipeline Verification")
    print("=" * 60)

    # 1. Load one scenario
    print("\n[1/5] Loading dataset...")
    from src.starcraft.datasets.sc_dataset import SCDataset
    from src.starcraft.datamodules.sc_target_builder import SCTargetBuilderTrain
    ds = SCDataset("datasets/StarCraftMotion_split_v1_medium", "train")
    raw = ds[0]
    print(f"  scenario_id: {raw['scenario_id']}")
    print(f"  agents: {raw['agent']['valid_mask'].shape[0]}")

    # 2. Apply target builder
    print("\n[2/5] Applying target builder...")
    tb = SCTargetBuilderTrain(max_num=64, min_future_alive=8)
    data = tb(raw)
    train_mask = data["agent"]["train_mask"]
    print(f"  train_mask: {train_mask.sum().item()} / {train_mask.shape[0]} agents")

    # 3. Batch via DataLoader (single sample)
    print("\n[3/5] Batching...")
    from torch_geometric.loader import DataLoader
    ds_with_transform = SCDataset(
        "datasets/StarCraftMotion_split_v1_medium", "train", transform=tb
    )
    loader = DataLoader(ds_with_transform, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    n_agent = batch["agent"]["valid_mask"].shape[0]
    print(f"  Batch agents: {n_agent}")
    print(f"  Batch graphs: {batch.num_graphs}")

    # 4. Tokenize
    print("\n[4/5] Tokenizing...")
    from src.starcraft.tokens.sc_token_processor import SCTokenProcessor
    from omegaconf import OmegaConf
    tp = SCTokenProcessor(
        motion_dict_file="/local/scratch/a/bai116/datasets/motion_dict_out/motion_dictionary.pkl",
        agent_token_sampling=OmegaConf.create({"num_k": 1, "temp": 1.0}),
    )
    tp.eval()
    tokenized_map, tokenized_agent = tp(batch)

    print(f"  n_token_agent: {tp.n_token_agent}")
    for k, v in tokenized_agent.items():
        if isinstance(v, torch.Tensor):
            print(f"  tokenized_agent[{k}]: {v.shape} {v.dtype}")

    # Verify key shapes
    assert tokenized_agent["gt_idx"].shape == (n_agent, 18), \
        f"gt_idx shape mismatch: {tokenized_agent['gt_idx'].shape}"
    assert tokenized_agent["valid_mask"].shape == (n_agent, 18), \
        f"valid_mask shape mismatch: {tokenized_agent['valid_mask'].shape}"
    assert tokenized_agent["gt_idx"].min() >= 0, "Negative token index"
    assert tokenized_agent["gt_idx"].max() < tp.n_token_agent, "Token index out of range"
    print("  Shape checks passed!")

    # 5. Forward pass + loss
    print("\n[5/5] Forward pass + loss...")
    from src.starcraft.modules.sc_decoder import SCDecoder

    decoder = SCDecoder(
        hidden_dim=64,
        num_historical_steps=17,
        num_future_steps=128,
        time_span=30,
        a2a_radius=30,
        num_freq_bands=32,
        num_agent_layers=2,  # small for quick test
        num_heads=4,
        head_dim=16,
        dropout=0.0,
        hist_drop_prob=0.0,
        n_token_agent=tp.n_token_agent,
    )
    decoder.train()

    pred = decoder(tokenized_map, tokenized_agent)
    print(f"  next_token_logits: {pred['next_token_logits'].shape}")
    print(f"  next_token_valid: {pred['next_token_valid'].shape}")

    assert pred["next_token_logits"].shape == (n_agent, 16, tp.n_token_agent), \
        f"Logits shape mismatch: {pred['next_token_logits'].shape}"

    # Loss
    from src.starcraft.metrics.sc_cross_entropy import SCCrossEntropy
    loss_fn = SCCrossEntropy(
        use_gt_raw=True,
        gt_thresh_scale_length=-1.0,
        label_smoothing=0.1,
        rollout_as_gt=False,
    )
    loss_fn.train()
    loss_fn.update(
        **pred,
        token_agent_shape=tokenized_agent["token_agent_shape"],
        token_traj=tokenized_agent["token_traj"],
        train_mask=batch["agent"]["train_mask"],
    )
    loss_val = loss_fn.compute()
    print(f"  Loss: {loss_val.item():.4f}")
    assert torch.isfinite(loss_val), "Loss is not finite!"

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
