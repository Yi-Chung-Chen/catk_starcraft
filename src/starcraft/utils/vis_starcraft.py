"""StarCraft scenario visualization: GT + predicted trajectories as animated GIF."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

TOTAL_FRAMES = 145
FPS = 16  # native frame rate


def extract_scenario_data(data, pred_traj, scenario_idx, rollout_idx):
    """Extract per-scenario numpy arrays from batched PyG data.

    Args:
        data: Batched PyG HeteroData with data["agent"] fields.
        pred_traj: (N_batch, n_rollout, n_future_steps, 2) predicted trajectories.
        scenario_idx: Which scenario in the batch to extract.
        rollout_idx: Which rollout to extract.

    Returns:
        Dict with numpy arrays for one scenario + one rollout.
    """
    batch = data["agent"]["batch"]
    mask = batch == scenario_idx

    return {
        "gt_positions": data["agent"]["position"][mask, :, :2].cpu().numpy(),
        "gt_valid": data["agent"]["valid_mask"][mask].cpu().numpy(),
        "pred_positions": pred_traj[mask, rollout_idx].cpu().numpy(),
        "owner": data["agent"]["owner"][mask].cpu().numpy(),
        "scenario_id": data["scenario_id"][scenario_idx],
    }


def save_scenario_gif(
    gt_positions,
    gt_valid,
    pred_positions,
    owner,
    scenario_id,
    save_path,
    num_historical_steps=17,
    frame_step=2,
    trail_length=16,
    gif_fps=8,
):
    """Render an animated GIF showing GT and predicted trajectories.

    Args:
        gt_positions: (N, T, 2) ground-truth XY positions for all T frames.
        gt_valid: (N, T) boolean alive mask.
        pred_positions: (N, T_pred, 2) predicted XY, starting from frame num_historical_steps.
        owner: (N,) player ownership (1=P1, 2=P2, 16=Neutral).
        scenario_id: String identifier for the scenario.
        save_path: Output GIF path.
        num_historical_steps: First predicted frame index in game time.
        frame_step: Sample every N frames for the GIF.
        trail_length: Number of frames to show as trailing line.
        gif_fps: GIF playback speed.
    """
    N, T = gt_positions.shape[0], gt_positions.shape[1]

    p1_mask = owner == 1
    p2_mask = owner == 2
    neutral_mask = owner == 16

    # Compute total displacement per unit for mobile detection
    total_disp = np.zeros(N)
    for u in range(N):
        alive_frames = np.where(gt_valid[u])[0]
        if len(alive_frames) > 1:
            total_disp[u] = np.linalg.norm(
                np.diff(gt_positions[u, alive_frames], axis=0), axis=-1
            ).sum()
    mobile = total_disp > 0.5

    # Spatial bounds from alive GT positions
    alive_any = gt_valid.any(axis=1)  # (N,)
    all_alive_pos = []
    for u in np.where(alive_any)[0]:
        af = gt_valid[u]
        all_alive_pos.append(gt_positions[u, af])
    if all_alive_pos:
        all_pos = np.concatenate(all_alive_pos, axis=0)
        x_min, y_min = all_pos.min(axis=0) - 5
        x_max, y_max = all_pos.max(axis=0) + 5
    else:
        x_min, y_min, x_max, y_max = -50, -50, 50, 50

    # --- Set up figure ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Neutral units (static, from frame 0)
    neutral_alive_0 = gt_valid[:, 0] & neutral_mask
    if neutral_alive_0.any():
        ax.scatter(
            gt_positions[neutral_alive_0, 0, 0],
            gt_positions[neutral_alive_0, 0, 1],
            c="#aaaaaa", s=10, alpha=0.3, marker="s", label="Neutral", zorder=1,
        )

    # GT scatter artists
    scat_gt_p1 = ax.scatter(
        [], [], c="dodgerblue", s=30, edgecolors="navy",
        linewidths=0.5, marker="o", label="P1 (GT)", zorder=4,
    )
    scat_gt_p2 = ax.scatter(
        [], [], c="tomato", s=30, edgecolors="darkred",
        linewidths=0.5, marker="o", label="P2 (GT)", zorder=4,
    )

    # Pred scatter artists
    scat_pred_p1 = ax.scatter(
        [], [], c="dodgerblue", s=40, edgecolors="navy",
        linewidths=0.8, marker="D", alpha=0.6, label="P1 (Pred)", zorder=5,
    )
    scat_pred_p2 = ax.scatter(
        [], [], c="tomato", s=40, edgecolors="darkred",
        linewidths=0.8, marker="D", alpha=0.6, label="P2 (Pred)", zorder=5,
    )

    # GT trail lines (solid) for mobile player units
    gt_trail_lines = []
    for u in np.where(mobile & p1_mask)[0]:
        (line,) = ax.plot(
            [], [], "-", color="dodgerblue", alpha=0.4, linewidth=1.2, zorder=2,
        )
        gt_trail_lines.append((u, line))
    for u in np.where(mobile & p2_mask)[0]:
        (line,) = ax.plot(
            [], [], "-", color="tomato", alpha=0.4, linewidth=1.2, zorder=2,
        )
        gt_trail_lines.append((u, line))

    # Pred trail lines (dashed) for mobile player units
    pred_trail_lines = []
    for u in np.where(mobile & p1_mask)[0]:
        (line,) = ax.plot(
            [], [], "--", color="cornflowerblue", alpha=0.3, linewidth=1.2, zorder=3,
        )
        pred_trail_lines.append((u, line))
    for u in np.where(mobile & p2_mask)[0]:
        (line,) = ax.plot(
            [], [], "--", color="salmon", alpha=0.3, linewidth=1.2, zorder=3,
        )
        pred_trail_lines.append((u, line))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X (game units)")
    ax.set_ylabel("Y (game units)")
    ax.set_title(f"Scenario: {scenario_id}", fontsize=11)

    time_text = ax.text(
        0.5, 0.02, "", transform=ax.transAxes, fontsize=10,
        ha="center", va="bottom", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        zorder=10,
    )

    def update(frame_idx):
        # --- GT positions ---
        gt_p1_alive = gt_valid[:, frame_idx] & p1_mask
        gt_p2_alive = gt_valid[:, frame_idx] & p2_mask

        scat_gt_p1.set_offsets(gt_positions[gt_p1_alive, frame_idx])
        scat_gt_p2.set_offsets(gt_positions[gt_p2_alive, frame_idx])

        # GT trails
        t_start = max(0, frame_idx - trail_length)
        for u, line in gt_trail_lines:
            if gt_valid[u, frame_idx]:
                seg = gt_valid[u, t_start : frame_idx + 1]
                tc = gt_positions[u, t_start : frame_idx + 1]
                line.set_data(tc[seg, 0], tc[seg, 1])
            else:
                line.set_data([], [])

        # --- Predicted positions ---
        if frame_idx >= num_historical_steps:
            pred_idx = frame_idx - num_historical_steps
            # Show pred for agents alive at current frame in GT
            pred_p1_alive = gt_valid[:, frame_idx] & p1_mask
            pred_p2_alive = gt_valid[:, frame_idx] & p2_mask

            scat_pred_p1.set_offsets(pred_positions[pred_p1_alive, pred_idx])
            scat_pred_p2.set_offsets(pred_positions[pred_p2_alive, pred_idx])

            # Pred trails
            pred_t_start = max(0, pred_idx - trail_length)
            for u, line in pred_trail_lines:
                if gt_valid[u, frame_idx]:
                    # For pred trails, we don't have a per-frame valid, use gt as proxy
                    seg_start = max(0, pred_idx - trail_length)
                    seg = pred_positions[u, seg_start : pred_idx + 1]
                    line.set_data(seg[:, 0], seg[:, 1])
                else:
                    line.set_data([], [])
        else:
            scat_pred_p1.set_offsets(np.empty((0, 2)))
            scat_pred_p2.set_offsets(np.empty((0, 2)))
            for _, line in pred_trail_lines:
                line.set_data([], [])

        # Phase text
        if frame_idx < num_historical_steps - 1:
            phase = "HISTORY"
            bg = "wheat"
        elif frame_idx == num_historical_steps - 1:
            phase = ">>> CURRENT <<<"
            bg = "#aaffaa"
        else:
            phase = "PREDICTION"
            bg = "#ffe0b2"

        n_p1 = gt_p1_alive.sum()
        n_p2 = gt_p2_alive.sum()
        time_text.set_text(
            f"{phase}  |  Frame {frame_idx}/{T - 1}  |  "
            f"{frame_idx / FPS:.2f}s  |  P1:{n_p1} P2:{n_p2}"
        )
        time_text.get_bbox_patch().set_facecolor(bg)

        artists = [scat_gt_p1, scat_gt_p2, scat_pred_p1, scat_pred_p2, time_text]
        artists += [l for _, l in gt_trail_lines]
        artists += [l for _, l in pred_trail_lines]
        return artists

    frame_indices = list(range(0, T, frame_step))
    ani = FuncAnimation(fig, update, frames=frame_indices, blit=False, interval=125)
    ani.save(save_path, writer=PillowWriter(fps=gif_fps), dpi=100)
    plt.close(fig)
