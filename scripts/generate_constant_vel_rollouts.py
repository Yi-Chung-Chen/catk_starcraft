"""Constant-velocity baseline rollout generator for StarCraft closed-loop eval.

Extrapolates each agent linearly from its last two observed native-rate frames
(Waymo sim-agents ConstantVelocity baseline). Writes HDF5 files in the **same
schema** as SCSMART's `save_rollout_batch`, so the output directory plugs into
`scripts/eval_sc_rollouts.py` with zero changes.

Invoked via Hydra. Dataset / map / batch / worker knobs are Hydra overrides —
identical syntax to `scripts/local_test_sc.sh`.

The SC experiment (`experiment=sc_pre_bc`) is pre-selected at argv level so
the plain invocation works out of the box; override with
`experiment=<other_sc_experiment>` if needed.

Shorthand `maps=<token>` is translated at argv level to the matching
`data.test_map_names=[...]` Hydra override (same alphabet as
`scripts/local_test_sc.sh`):
    - `maps=id`   → the five id maps (default)
    - `maps=ood`  → the two held-out maps
    - `maps=all`  → no filter (uses the data config's default)
    - `maps=Foo,Bar` → custom comma list (with quoting for parens if needed)

Example:
    python scripts/generate_constant_vel_rollouts.py           # id maps, default task_name
    python scripts/generate_constant_vel_rollouts.py maps=ood  # held-out maps

Output: `${hydra.runtime.output_dir}/rollouts/{scenario_id}.h5` — one file
per scenario, four groups (`obs_p{1,2}/{own,opponent}`).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Ensure repo root is importable when invoked as a plain script (Hydra apps
# don't get the `python -m src.run` treatment).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# configs/run.yaml still defaults to the Waymo/SMART experiment. Force the
# StarCraft experiment so the plain `python scripts/generate_constant_vel_rollouts.py`
# invocation works without the caller remembering `experiment=sc_pre_bc`.
# The user can still override (e.g., a future `experiment=sc_clsft`).
if not any(
    a.startswith("experiment=") or a.startswith("+experiment=")
    for a in sys.argv[1:]
):
    sys.argv.insert(1, "experiment=sc_pre_bc")

# Default the output dir name so logs/ stays organized — this script only
# generates the one (constant_velocity) baseline method.
if not any(
    a.startswith("task_name=") or a.startswith("+task_name=")
    for a in sys.argv[1:]
):
    sys.argv.insert(1, "task_name=sc_closed_test_constant_velocity")

# Translate `maps=<token>` shortcuts to the full `data.test_map_names=[...]`
# Hydra override, mirroring `scripts/local_test_sc.sh`. Default to `id` when
# absent so the baseline ships with the same eval split as the test runs.
_MAP_PRESETS = {
    "id": "[Abyssal_Reef_LE,Acolyte_LE,Ascension_to_Aiur_LE,Interloper_LE,Mech_Depot_LE]",
    "ood": "['Catallena_LE_(Void)',Odyssey_LE]",
    # "all": no filter — explicitly drop any maps= shortcut so Hydra falls back
    # to the data config's test_map_names (null → every map in the test split).
}
_map_arg = next(
    (a for a in sys.argv[1:] if a.startswith("maps=")),
    None,
)
if _map_arg is None:
    # Default to the id maps, matching scripts/local_test_sc.sh.
    sys.argv.insert(1, f"data.test_map_names={_MAP_PRESETS['id']}")
else:
    # Always strip the `maps=` shortcut — it isn't a real Hydra config key,
    # so leaving it in argv would fail with "Key 'maps' is not in struct".
    sys.argv.remove(_map_arg)
    if any(
        a.startswith("data.test_map_names=") or a.startswith("+data.test_map_names=")
        for a in sys.argv[1:]
    ):
        # User provided an explicit override too — that wins. Drop the
        # shortcut silently.
        pass
    else:
        _token = _map_arg.split("=", 1)[1]
        if _token == "all":
            pass  # no filter → use data config default (null = every map)
        elif _token in _MAP_PRESETS:
            sys.argv.insert(1, f"data.test_map_names={_MAP_PRESETS[_token]}")
        else:
            # Custom comma list, e.g. maps='Catallena_LE_(Void),Odyssey_LE'.
            sys.argv.insert(1, f"data.test_map_names=[{_token}]")

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.starcraft.tokens.sc_token_processor import (  # noqa: E402
    SCTokenProcessor,
    filter_agents_for_perspective,
)
from src.starcraft.utils.sc_rollout_io import save_rollout_batch  # noqa: E402

_NUM_HISTORICAL_STEPS = 17
_NUM_FUTURE_STEPS = 128
_NATIVE_FPS = 16


def _build_constant_velocity_rollout(data) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute [N_total, 1, 128, 2] pred_traj and [N_total, 1, 128] pred_head
    from raw native-rate GT history.

    Velocity is `pos[:, 16] - pos[:, 15]`, per native frame. Falls back to
    zero velocity (hold position) when either frame 15 or frame 16 is
    invalid, so dead/late-spawned units extrapolate sensibly.
    """
    pos = data["agent"]["position"][..., :2]  # [N, 145, 2]
    valid = data["agent"]["valid_mask"]        # [N, 145]
    heading = data["agent"]["heading"]         # [N, 145]

    t_now = _NUM_HISTORICAL_STEPS - 1   # = 16
    t_prev = t_now - 1                  # = 15

    vel_valid = valid[:, t_prev] & valid[:, t_now]  # [N]
    vel = torch.where(
        vel_valid.unsqueeze(-1),
        pos[:, t_now] - pos[:, t_prev],
        torch.zeros_like(pos[:, t_now]),
    )  # [N, 2]

    # Future frames are t_now+1 .. t_now+128. Step index k ∈ [0, 128),
    # displacement is (k+1) * vel.
    k = torch.arange(
        1, _NUM_FUTURE_STEPS + 1, device=pos.device, dtype=pos.dtype
    ).view(1, _NUM_FUTURE_STEPS, 1)
    pred_pos = pos[:, t_now].unsqueeze(1) + k * vel.unsqueeze(1)  # [N, 128, 2]
    pred_head = heading[:, t_now].unsqueeze(-1).expand(-1, _NUM_FUTURE_STEPS)  # [N, 128]

    return pred_pos.unsqueeze(1), pred_head.unsqueeze(1)  # add rollout axis (R=1)


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_printoptions(precision=3)

    # Datamodule: has a real _target_ in configs/data/starcraft.yaml.
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    # Token processor: the token_processor block in configs/model/sc_smart.yaml
    # has no _target_, so construct directly. Pass `agent_token_sampling` as
    # DictConfig (not a plain dict) — the processor uses attribute access
    # (`.num_k`, `.temp`) on it, which plain dicts don't support.
    tp_cfg = cfg.model.model_config.token_processor
    token_processor = SCTokenProcessor(
        motion_dict_file=tp_cfg.motion_dict_file,
        map_data_dir=tp_cfg.map_data_dir,
        agent_token_sampling=tp_cfg.agent_token_sampling,
    )

    save_dir = Path(cfg.paths.output_dir) / "rollouts"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Replicate the single deterministic rollout N_ROLLOUTS times (Waymo
    # sim-agents convention). Matters because the offline NLL metrics
    # (kinematic_nll, interaction_nll) fit a Silverman-bandwidth KDE over R
    # samples and reject R < 2 outright. The replicas carry no extra
    # information — the KDE reduces to a single Gaussian centered at the
    # baseline prediction — but this keeps the metric pipeline apples-to-
    # apples with the model's rollouts.
    n_rollouts = int(cfg.model.model_config.n_rollout_closed_val)
    if n_rollouts < 2:
        raise ValueError(
            f"n_rollout_closed_val must be >= 2 for the offline NLL metrics "
            f"(got {n_rollouts}). Override via "
            "`model.model_config.n_rollout_closed_val=16`."
        )

    # Match SCSMART's file_attrs schema so load_rollout / offline harness
    # treat baseline files indistinguishably from model files.
    dataset_version = Path(cfg.data.dataset_root).name
    # A hash of "constant_velocity" makes the hash-field populated (keeps
    # downstream CSVs from special-casing an empty field), while the
    # string itself doesn't pretend to be a real model fingerprint.
    model_config_hash = hashlib.sha256(b"constant_velocity").hexdigest()[:16]
    file_attrs = {
        "num_historical_steps": _NUM_HISTORICAL_STEPS,
        "num_future_steps": _NUM_FUTURE_STEPS,
        "n_rollouts": n_rollouts,
        "native_fps": _NATIVE_FPS,
        "dataset_version": dataset_version,
        "model_config_hash": model_config_hash,
        "stage": "test",
    }

    precision = str(
        cfg.model.model_config.get("rollout_save_precision", "fp16")
    )

    print(
        f"[constant_vel] dataset={dataset_version} "
        f"maps={list(cfg.data.test_map_names) if cfg.data.test_map_names else 'all'} "
        f"n_rollouts={n_rollouts} save_dir={save_dir}"
    )

    for data in tqdm(dataloader, desc="generate", unit="batch"):
        pred_traj_all, pred_head_all = _build_constant_velocity_rollout(data)
        # Replicate the deterministic rollout across R samples (see
        # n_rollouts comment above). .expand is a zero-copy view;
        # .contiguous() materializes so downstream `.numpy()` in the saver
        # sees a normal-stride tensor.
        pred_traj_all = pred_traj_all.expand(-1, n_rollouts, -1, -1).contiguous()
        pred_head_all = pred_head_all.expand(-1, n_rollouts, -1).contiguous()

        tokenized_agent = token_processor.tokenize_agent(data)
        train_mask = data["agent"]["train_mask"]

        for observer_player in [1, 2]:
            filt_e, obs_mask_e, opp_mask_e, keep_mask_e, vis_to_obs = (
                filter_agents_for_perspective(
                    tokenized_agent, train_mask, observer_player,
                    opponent_keep_mode="visible_ever",
                )
            )
            is_observer = filt_e["is_observer"]
            is_neutral = filt_e["owner"] == 16
            is_opponent = ~is_observer & ~is_neutral
            obs_sees_current = vis_to_obs[:, 1]

            # Slice the observer-agnostic rollout down to kept rows.
            pred_traj_obs = pred_traj_all[keep_mask_e]   # [n_filt, 1, 128, 2]
            pred_head_obs = pred_head_all[keep_mask_e]   # [n_filt, 1, 128]

            # GT overlay target (same slice _run_closed_loop_modes uses).
            target = data["agent"]["position"][
                keep_mask_e, _NUM_HISTORICAL_STEPS:, :2
            ]  # [n_filt, 128, 2]

            for mode in ("own", "opponent"):
                if mode == "own":
                    teacher_force_mask = ~is_observer
                    metric_scope = obs_mask_e
                else:
                    is_late_obs_opp = is_opponent & ~obs_sees_current
                    teacher_force_mask = is_observer | is_neutral | is_late_obs_opp
                    metric_scope = opp_mask_e & obs_sees_current

                # Clone so the GT overlay for mode="own" doesn't leak into
                # the mode="opponent" pass of the same observer.
                pred_traj = pred_traj_obs.clone()
                pred_head = pred_head_obs.clone()

                tf_target_gt = target[teacher_force_mask]
                if tf_target_gt.shape[0] > 0:
                    pred_traj[teacher_force_mask] = tf_target_gt.unsqueeze(1).expand(
                        -1, pred_traj.shape[1], -1, -1,
                    )

                save_rollout_batch(
                    save_dir=save_dir,
                    data=data,
                    filt_e=filt_e,
                    keep_mask_e=keep_mask_e,
                    metric_scope=metric_scope,
                    teacher_force_mask=teacher_force_mask,
                    pred_traj=pred_traj,
                    pred_head=pred_head,
                    vis_to_obs=vis_to_obs,
                    aux_target_list=None,
                    observer_player=observer_player,
                    mode=mode,
                    file_attrs=file_attrs,
                    precision=precision,
                    global_rank=0,
                    world_size=1,
                )

    print(f"[constant_vel] done. rollouts at: {save_dir}")


if __name__ == "__main__":
    main()
