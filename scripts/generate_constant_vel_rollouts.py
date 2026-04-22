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

Noise (Waymo sim-agents style, per-step action noise on velocity / heading):
    +noise_vel_sigma=<σ_v>   per-axis velocity noise std, cells/frame (default 0.0)
    +noise_head_sigma=<σ_h>  heading noise std, radians/frame  (default 0.0)
    The `+` prefix is required — these are new Hydra config keys.
    Recommended first setting: `+noise_vel_sigma=0.06`.

Example:
    python scripts/generate_constant_vel_rollouts.py                         # deterministic CV
    python scripts/generate_constant_vel_rollouts.py maps=ood                # held-out maps
    python scripts/generate_constant_vel_rollouts.py +noise_vel_sigma=0.06   # stochastic, non-degenerate NLL

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

def _noise_sigma_from_argv(key: str) -> float:
    """Argv-level scan for a sigma override of the form
    `+<key>=<float>` or `<key>=<float>`. Returns 0.0 if absent or malformed."""
    for a in sys.argv[1:]:
        if a.startswith(f"+{key}=") or a.startswith(f"{key}="):
            try:
                return float(a.split("=", 1)[1])
            except ValueError:
                return 0.0
    return 0.0


# Default the output dir name so logs/ stays organized — this script only
# generates the one (constant_velocity) baseline method. If either sigma
# is nonzero, use a distinct task_name so noise/no-noise runs don't clobber.
if not any(
    a.startswith("task_name=") or a.startswith("+task_name=")
    for a in sys.argv[1:]
):
    _has_noise = (
        _noise_sigma_from_argv("noise_vel_sigma") > 0
        or _noise_sigma_from_argv("noise_head_sigma") > 0
    )
    _default_task_name = (
        "sc_closed_test_constant_velocity_noise" if _has_noise
        else "sc_closed_test_constant_velocity"
    )
    sys.argv.insert(1, f"task_name={_default_task_name}")

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


def _build_rollout(
    data,
    n_rollouts: int,
    noise_vel_sigma: float,
    noise_head_sigma: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute `[N, R, 128, 2]` `pred_traj` and `[N, R, 128]` `pred_head` from
    raw native-rate GT history.

    Base velocity `v = pos[:, 16] - pos[:, 15]`, zero when either frame is
    invalid (hold position for dead / late-spawned units).

    When both sigmas are zero, takes a deterministic fast path — the R
    replicas are byte-identical copies of a single CV trajectory (matching
    the pre-noise behavior exactly).

    When either sigma > 0, uses Waymo's per-step action-noise form:
        state[r, k+1] = state[r, k] + v + eps_v[r, k]   eps_v ~ N(0, σ_v²)
        head [r, k+1] = head [r, k]       + eps_h[r, k] eps_h ~ N(0, σ_h²)
    Each replica is a random walk on velocity / heading, so positional and
    angular variance grow with horizon — non-degenerate for NLL metrics.
    """
    pos = data["agent"]["position"][..., :2]  # [N, 145, 2]
    valid = data["agent"]["valid_mask"]        # [N, 145]
    heading = data["agent"]["heading"]         # [N, 145]
    N = pos.shape[0]
    R = n_rollouts
    K = _NUM_FUTURE_STEPS

    t_now = _NUM_HISTORICAL_STEPS - 1   # = 16
    t_prev = t_now - 1                  # = 15

    vel_valid = valid[:, t_prev] & valid[:, t_now]  # [N]
    vel = torch.where(
        vel_valid.unsqueeze(-1),
        pos[:, t_now] - pos[:, t_prev],
        torch.zeros_like(pos[:, t_now]),
    )  # [N, 2]

    if noise_vel_sigma == 0 and noise_head_sigma == 0:
        # Deterministic fast path — preserves byte-for-byte output vs. the
        # pre-noise implementation.
        k_idx = torch.arange(
            1, K + 1, device=pos.device, dtype=pos.dtype
        ).view(1, K, 1)
        pred_pos = pos[:, t_now].unsqueeze(1) + k_idx * vel.unsqueeze(1)  # [N, K, 2]
        pred_head = heading[:, t_now].unsqueeze(-1).expand(-1, K)         # [N, K]
        pred_traj = pred_pos.unsqueeze(1).expand(-1, R, -1, -1).contiguous()
        pred_head = pred_head.unsqueeze(1).expand(-1, R, -1).contiguous()
        return pred_traj, pred_head

    # Noisy path: per-step velocity/heading action noise, integrated by
    # cumulative sum (vectorized random walk).
    if noise_vel_sigma > 0:
        eps_v = torch.randn(N, R, K, 2, generator=generator) * noise_vel_sigma
    else:
        eps_v = torch.zeros(N, R, K, 2)
    v_eff = vel.view(N, 1, 1, 2) + eps_v                        # [N, R, K, 2]
    displacement = v_eff.cumsum(dim=2)                           # [N, R, K, 2]
    pred_traj = pos[:, t_now].view(N, 1, 1, 2) + displacement    # [N, R, K, 2]

    if noise_head_sigma > 0:
        eps_h = torch.randn(N, R, K, generator=generator) * noise_head_sigma
        head_displacement = eps_h.cumsum(dim=2)                  # [N, R, K]
        pred_head = heading[:, t_now].view(N, 1, 1) + head_displacement
    else:
        pred_head = (
            heading[:, t_now].view(N, 1, 1).expand(N, R, K).contiguous()
        )

    return pred_traj, pred_head


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

    # Noise parameters (new keys; users pass via `+noise_vel_sigma=...` at CLI).
    # OmegaConf.select tolerates missing keys and the `+`-prefix add form.
    noise_vel_sigma = float(OmegaConf.select(cfg, "noise_vel_sigma", default=0.0))
    noise_head_sigma = float(OmegaConf.select(cfg, "noise_head_sigma", default=0.0))

    # Seeded CPU generator for reproducibility — same seed + same sigmas →
    # byte-identical output. cfg.seed is plumbed via configs/run.yaml:29.
    noise_generator = torch.Generator()
    noise_generator.manual_seed(int(cfg.seed))

    # Match SCSMART's file_attrs schema so load_rollout / offline harness
    # treat baseline files indistinguishably from model files.
    dataset_version = Path(cfg.data.dataset_root).name
    # Keep the deterministic hash stable when no noise is configured, so the
    # σ=0 output remains byte-identical to the pre-noise implementation.
    if noise_vel_sigma == 0 and noise_head_sigma == 0:
        hash_input = "constant_velocity"
    else:
        hash_input = (
            "constant_velocity_step_noise"
            f"|vel_sigma={noise_vel_sigma}|head_sigma={noise_head_sigma}"
        )
    model_config_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
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

    # `trainer.limit_test_batches` is a Lightning knob and has no effect when
    # we iterate the dataloader directly. Honor it here so sweep drivers that
    # pass the same override work for both model runs and this baseline.
    limit_test_batches = OmegaConf.select(
        cfg, "trainer.limit_test_batches", default=None,
    )
    max_batches = None
    if limit_test_batches is not None:
        try:
            lt = float(limit_test_batches)
            if 0 < lt < 1:
                # Fractional form: fraction of the total dataloader length.
                max_batches = max(1, int(lt * len(dataloader)))
            elif lt >= 1:
                max_batches = int(lt)
        except (TypeError, ValueError):
            max_batches = None

    print(
        f"[constant_vel] dataset={dataset_version} "
        f"maps={list(cfg.data.test_map_names) if cfg.data.test_map_names else 'all'} "
        f"n_rollouts={n_rollouts} "
        f"noise_vel_sigma={noise_vel_sigma} noise_head_sigma={noise_head_sigma} "
        f"limit_test_batches={max_batches if max_batches is not None else 'none'} "
        f"save_dir={save_dir}"
    )

    for batch_idx, data in enumerate(tqdm(dataloader, desc="generate", unit="batch")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pred_traj_all, pred_head_all = _build_rollout(
            data, n_rollouts, noise_vel_sigma, noise_head_sigma, noise_generator,
        )

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
