"""StarCraft scenario visualization: GT + predicted trajectories as animated GIF."""

import matplotlib

matplotlib.use("Agg")

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

TOTAL_FRAMES = 145
FPS = 16  # native frame rate


def extract_scenario_data(data, pred_traj, scenario_idx, rollout_idx,
                          num_historical_steps=17, aux_target_data=None,
                          map_data_dir=None, pred_valid_mask=None,
                          observer_player=None, target_mask=None,
                          pred_head=None):
    """Extract per-scenario numpy arrays from batched PyG data.

    Args:
        data: Batched PyG HeteroData with data["agent"] fields.
        pred_traj: (N_batch, n_rollout, n_future_steps, 2) predicted trajectories.
        scenario_idx: Which scenario in the batch to extract.
        rollout_idx: Which rollout to extract.
        num_historical_steps: Agents must be alive at frame
            (num_historical_steps - 1) to have valid predictions.
        aux_target_data: Optional dict with target_pos_pred, has_target_pos_logits,
            gt_rel_target_pos, gt_has_target_pos tensors (batch-level).
        observer_player: Optional int (1 or 2). When provided, aux intent
            fields (`gt_target_rel`, `gt_has_target`, `pred_target_rel`,
            `pred_has_target`) are zeroed for non-observer rows. Non-observer
            GT is already zeroed by filter_agents_for_perspective, but
            non-observer aux head predictions are untrained garbage and must
            not be rendered.
        target_mask: Optional (N_total,) bool. When provided, `pred_agent_mask`
            is further narrowed to this mask — diamond markers and pred
            trails then appear only for units of interest under the current
            observer/mode (the same rows saved by the rollout I/O layer).

    Returns:
        Dict with numpy arrays for one scenario + one rollout.
    """
    batch = data["agent"]["batch"]
    mask = batch == scenario_idx
    valid = data["agent"]["valid_mask"][mask]

    # Agents alive at the current frame have valid model predictions.
    # Agents spawning later started from (0,0) and must be excluded.
    # `pred_valid_mask` additionally excludes agents the model never ran on
    # (e.g. opponents hidden by fog of war), whose predictions are zero-filled.
    current_frame = num_historical_steps - 1
    pred_agent_mask = valid[:, current_frame].cpu().numpy()
    if pred_valid_mask is not None:
        pred_agent_mask = pred_agent_mask & pred_valid_mask[mask].cpu().numpy()
    if target_mask is not None:
        pred_agent_mask = pred_agent_mask & target_mask[mask].cpu().numpy()

    out = {
        "gt_positions": data["agent"]["position"][mask, :, :2].cpu().numpy(),
        "gt_headings": data["agent"]["heading"][mask].cpu().numpy(),
        "gt_valid": valid.cpu().numpy(),
        "pred_positions": pred_traj[mask, rollout_idx].cpu().numpy(),
        "pred_headings": (
            pred_head[mask, rollout_idx].cpu().numpy() if pred_head is not None else None
        ),
        "pred_agent_mask": pred_agent_mask,
        "owner": data["agent"]["owner"][mask].cpu().numpy(),
        "scenario_id": data["scenario_id"][scenario_idx],
    }

    if aux_target_data is not None:
        pred_target_rel = aux_target_data["target_pos_pred"][mask].cpu().numpy()
        pred_has_target = (aux_target_data["has_target_pos_logits"][mask] > 0).cpu().numpy()
        gt_target_rel = aux_target_data["gt_rel_target_pos"][mask].cpu().numpy()
        gt_has_target = aux_target_data["gt_has_target_pos"][mask].cpu().numpy()
        # Non-observer rows: aux heads run on every agent but are untrained
        # for opponents/neutrals (loss uses obs_mask), so their predictions
        # are meaningless. GT is already zeroed by filter_agents_for_perspective
        # but we zero it here too for symmetry.
        if observer_player is not None:
            is_observer = data["agent"]["owner"][mask].cpu().numpy() == observer_player
            pred_target_rel[~is_observer] = 0.0
            pred_has_target[~is_observer] = False
            gt_target_rel[~is_observer] = 0.0
            gt_has_target[~is_observer] = False
        out["pred_target_rel"] = pred_target_rel
        out["pred_has_target"] = pred_has_target
        out["gt_target_rel"] = gt_target_rel
        out["gt_has_target"] = gt_has_target

    if map_data_dir is not None:
        map_names = data["map_name"]
        map_name = map_names[scenario_idx] if isinstance(map_names, (list, tuple)) else map_names
        with h5py.File(os.path.join(map_data_dir, f"{map_name}.h5"), "r") as f:
            out["pathing_grid"] = f["pathing_grid"][:].astype(np.float32)  # [H, W] before padding

    return out


_TARGET_POS_NORM = 200.0


def _local_to_world_offset(local_xy, head):
    """Rotate a single 2-D local-frame offset into world coords by +head."""
    cos_h, sin_h = np.cos(head), np.sin(head)
    return np.array([
        cos_h * local_xy[0] - sin_h * local_xy[1],
        sin_h * local_xy[0] + cos_h * local_xy[1],
    ])


def save_scenario_gif(
    gt_positions,
    gt_headings,
    gt_valid,
    pred_positions,
    pred_headings,
    pred_agent_mask,
    owner,
    scenario_id,
    save_path,
    num_historical_steps=17,
    frame_step=2,
    trail_length=16,
    gif_fps=8,
    pred_target_rel=None,
    pred_has_target=None,
    gt_target_rel=None,
    gt_has_target=None,
    pathing_grid=None,
):
    """Render an animated GIF showing GT and predicted trajectories.

    Args:
        gt_positions: (N, T, 2) ground-truth XY positions for all T frames.
        gt_valid: (N, T) boolean alive mask.
        pred_positions: (N, T_pred, 2) predicted XY, starting from frame num_historical_steps.
        pred_agent_mask: (N,) bool — True for agents alive at the current frame
            whose predictions are valid.  Agents spawning later have garbage
            predictions (origin) and are excluded.
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

    # Pathing grid background (before padding, original H x W)
    if pathing_grid is not None:
        H_map, W_map = pathing_grid.shape
        ax.imshow(
            pathing_grid, cmap="Greys_r", alpha=0.3,
            extent=[0, W_map, 0, H_map],  # [left, right, bottom, top]
            origin="upper", zorder=0,
        )

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

    # Pred trail lines (dashed) for mobile player units with valid predictions
    pred_trail_lines = []
    for u in np.where(mobile & p1_mask & pred_agent_mask)[0]:
        (line,) = ax.plot(
            [], [], "--", color="cornflowerblue", alpha=0.3, linewidth=1.2, zorder=3,
        )
        pred_trail_lines.append((u, line))
    for u in np.where(mobile & p2_mask & pred_agent_mask)[0]:
        (line,) = ax.plot(
            [], [], "--", color="salmon", alpha=0.3, linewidth=1.2, zorder=3,
        )
        pred_trail_lines.append((u, line))

    # Target position lines: unit → target (only for mobile player units)
    has_targets = gt_target_rel is not None
    target_shift = 8  # target data sampled every 8 native frames
    gt_target_lines = []
    pred_target_lines = []
    if has_targets:
        for u in np.where(mobile & (p1_mask | p2_mask) & pred_agent_mask)[0]:
            color_gt = "dodgerblue" if p1_mask[u] else "tomato"
            color_pred = "cornflowerblue" if p1_mask[u] else "salmon"
            (gl,) = ax.plot([], [], "-", color=color_gt, alpha=0.5, linewidth=0.8, zorder=6)
            gt_target_lines.append((u, gl))
            (pl,) = ax.plot([], [], "--", color=color_pred, alpha=0.5, linewidth=0.8, zorder=6)
            pred_target_lines.append((u, pl))
        scat_gt_target = ax.scatter(
            [], [], c="limegreen", s=20, marker="x", linewidths=1.0,
            label="GT Target", zorder=7,
        )
        scat_pred_target = ax.scatter(
            [], [], c="orange", s=20, marker="x", linewidths=1.0,
            label="Pred Target", zorder=7,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, ncol=3 if not has_targets else 4)
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

        # --- Predicted positions (only for agents with valid initial state) ---
        if frame_idx >= num_historical_steps:
            pred_idx = frame_idx - num_historical_steps
            pred_p1_alive = gt_valid[:, frame_idx] & p1_mask & pred_agent_mask
            pred_p2_alive = gt_valid[:, frame_idx] & p2_mask & pred_agent_mask

            scat_pred_p1.set_offsets(pred_positions[pred_p1_alive, pred_idx])
            scat_pred_p2.set_offsets(pred_positions[pred_p2_alive, pred_idx])

            # Pred trails
            for u, line in pred_trail_lines:
                if gt_valid[u, frame_idx]:
                    seg_start = max(0, pred_idx - trail_length)
                    seg = pred_positions[u, seg_start : pred_idx + 1]
                    line.set_data(seg[:, 0], seg[:, 1])
                else:
                    line.set_data([], [])

            # --- Target position arrows ---
            if has_targets:
                step_2hz = min(pred_idx // target_shift, gt_target_rel.shape[1] - 1)
                # Use unit position at the 2Hz boundary frame (not current frame)
                # to keep target fixed for the whole 8-frame window.
                # 2Hz steps 0..15 correspond to native frames 24,32,...,144
                boundary_frame = (step_2hz + 3) * target_shift
                boundary_frame = min(boundary_frame, T - 1)
                boundary_pred_idx = boundary_frame - num_historical_steps
                gt_tgt_pts = []
                pred_tgt_pts = []
                for u, gl in gt_target_lines:
                    if gt_valid[u, frame_idx] and gt_has_target[u, step_2hz]:
                        anchor = gt_positions[u, boundary_frame]
                        local_offset = gt_target_rel[u, step_2hz] * _TARGET_POS_NORM
                        tgt = anchor + _local_to_world_offset(
                            local_offset, gt_headings[u, boundary_frame]
                        )
                        gl.set_data([gt_positions[u, frame_idx, 0], tgt[0]],
                                    [gt_positions[u, frame_idx, 1], tgt[1]])
                        gt_tgt_pts.append(tgt)
                    else:
                        gl.set_data([], [])
                for u, pl in pred_target_lines:
                    if gt_valid[u, frame_idx] and pred_has_target[u, step_2hz]:
                        anchor = pred_positions[u, boundary_pred_idx]
                        local_offset = pred_target_rel[u, step_2hz] * _TARGET_POS_NORM
                        tgt = anchor + _local_to_world_offset(
                            local_offset, pred_headings[u, boundary_pred_idx]
                        )
                        pl.set_data([pred_positions[u, pred_idx, 0], tgt[0]],
                                    [pred_positions[u, pred_idx, 1], tgt[1]])
                        pred_tgt_pts.append(tgt)
                    else:
                        pl.set_data([], [])
                scat_gt_target.set_offsets(np.array(gt_tgt_pts) if gt_tgt_pts else np.empty((0, 2)))
                scat_pred_target.set_offsets(np.array(pred_tgt_pts) if pred_tgt_pts else np.empty((0, 2)))
        else:
            scat_pred_p1.set_offsets(np.empty((0, 2)))
            scat_pred_p2.set_offsets(np.empty((0, 2)))
            for _, line in pred_trail_lines:
                line.set_data([], [])
            if has_targets:
                for _, gl in gt_target_lines:
                    gl.set_data([], [])
                for _, pl in pred_target_lines:
                    pl.set_data([], [])
                scat_gt_target.set_offsets(np.empty((0, 2)))
                scat_pred_target.set_offsets(np.empty((0, 2)))

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
        if has_targets:
            artists += [l for _, l in gt_target_lines]
            artists += [l for _, l in pred_target_lines]
            artists += [scat_gt_target, scat_pred_target]
        return artists

    frame_indices = list(range(0, T, frame_step))
    ani = FuncAnimation(fig, update, frames=frame_indices, blit=False, interval=125)
    ani.save(save_path, writer=PillowWriter(fps=gif_fps), dpi=100)
    plt.close(fig)
