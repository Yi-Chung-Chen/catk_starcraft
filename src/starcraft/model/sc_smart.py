"""StarCraft SMART model (LightningModule)."""

import hashlib
import logging
import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import TokenCls, minADE
from src.starcraft.metrics.sc_action_target_loss import SCActionTargetLoss
from src.starcraft.metrics.sc_cross_entropy import SCCrossEntropy
from src.starcraft.modules.sc_decoder import SCDecoder
from src.starcraft.tokens.sc_token_processor import SCTokenProcessor, filter_agents_for_perspective
from src.starcraft.utils.sc_rollout_io import save_rollout_batch
from src.starcraft.utils.vis_starcraft import extract_scenario_data, save_scenario_gif
from src.smart.utils.finetune import set_model_for_finetuning

log = logging.getLogger(__name__)


class SCSMART(LightningModule):

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop

        self.token_processor = SCTokenProcessor(**model_config.token_processor)
        self.use_aux_loss = model_config.use_aux_loss
        self.encoder = SCDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
            use_aux_loss=self.use_aux_loss,
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.training_loss = SCCrossEntropy(**model_config.training_loss)
        if self.use_aux_loss:
            self.action_target_loss = SCActionTargetLoss(**model_config.action_target_loss)
        self.minADE = minADE()
        self.minADE_own_rollout = minADE()
        self.minADE_opp_rollout = minADE()
        self.TokenCls = TokenCls(max_guesses=5)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self.closed_loop_rollout_modes = list(model_config.closed_loop_rollout_modes)

        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.vis_observer_player = model_config.vis_observer_player

        hydra_output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        self.gif_dir = hydra_output_dir / "gifs"

        # Closed-loop rollout save (offline metric harness)
        self.save_closed_rollouts = bool(
            getattr(model_config, "save_closed_rollouts", False)
        )
        rollout_save_dir_cfg = getattr(model_config, "rollout_save_dir", None)
        self.rollout_save_dir = (
            Path(rollout_save_dir_cfg) if rollout_save_dir_cfg
            else hydra_output_dir / "rollouts"
        )
        self.save_rollout_modes = list(
            getattr(model_config, "save_rollout_modes", ["own", "opponent"])
        )
        self.save_rollout_observers = [
            int(o) for o in getattr(model_config, "save_rollout_observers", [1, 2])
        ]
        self.rollout_save_precision = str(
            getattr(model_config, "rollout_save_precision", "fp16")
        )
        rdv = getattr(model_config, "rollout_dataset_version", None)
        self.dataset_version = Path(rdv).name if rdv else "unknown"
        self.model_config_hash = hashlib.sha256(
            OmegaConf.to_yaml(model_config, resolve=True).encode()
        ).hexdigest()[:16]
        self.rollout_stage = "val"  # set in setup()

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        train_mask = data["agent"]["train_mask"]

        # Randomly choose one observer perspective per step.
        # opponent_keep_mode="visible_ever" keeps late-observed opponents in
        # the roster so the observer's input matches what an observer would
        # actually know mid-window (same roster the closed-loop evaluator
        # uses); vis_to_obs gates their attention edges per step.
        observer_player = 1 if torch.rand(1).item() < 0.5 else 2
        filtered_agent, obs_mask, opp_mask, _, vis_to_obs = (
            filter_agents_for_perspective(
                tokenized_agent, train_mask, observer_player,
                opponent_keep_mode="visible_ever",
            )
        )

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(
                tokenized_map, filtered_agent, visibility_gate=vis_to_obs,
            )
        else:
            pred = self.encoder.inference(
                tokenized_map, filtered_agent,
                sampling_scheme=self.training_rollout_sampling,
                visibility_gate=vis_to_obs,
            )

        # Trajectory loss: observer + opponent units
        combined_mask = obs_mask | opp_mask
        loss_motion = self.training_loss(
            **pred,
            token_agent_shape=filtered_agent["token_agent_shape"],
            token_traj=filtered_agent["token_traj"],
            train_mask=combined_mask,
        )
        # Aux loss: observer only (opponent has no intent)
        if self.use_aux_loss:
            loss_aux = self.action_target_loss(
                **pred, train_mask=obs_mask,
            )
            loss = loss_motion + loss_aux
            for k, v in self.action_target_loss.batch_components().items():
                self.log(f"train/loss_{k}", v, on_step=True, batch_size=1)
        else:
            loss = loss_motion
        self.log("train/loss", loss, on_step=True, batch_size=1)
        self.log("train/loss_motion", loss_motion, on_step=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        return self._shared_eval_step(data, batch_idx, stage="val")

    def test_step(self, data, batch_idx):
        return self._shared_eval_step(data, batch_idx, stage="test")

    def _shared_eval_step(self, data, batch_idx, stage: str):
        tokenized_map, tokenized_agent = self.token_processor(data)
        train_mask = data["agent"]["train_mask"]

        # Run both perspectives for complete metrics
        for observer_player in [1, 2]:
            # Expanded roster (matches training) so open-loop val measures the
            # same context the training signal was built on.
            filtered_agent, obs_mask, opp_mask, _, vis_to_obs = (
                filter_agents_for_perspective(
                    tokenized_agent, train_mask, observer_player,
                    opponent_keep_mode="visible_ever",
                )
            )
            combined_mask = obs_mask | opp_mask

            if self.val_open_loop:
                pred = self.encoder(
                    tokenized_map, filtered_agent, visibility_gate=vis_to_obs,
                )
                loss = self.training_loss(
                    **pred,
                    token_agent_shape=filtered_agent["token_agent_shape"],
                    token_traj=filtered_agent["token_traj"],
                    train_mask=combined_mask,
                )
                if self.use_aux_loss:
                    self.action_target_loss(**pred, train_mask=obs_mask)
                    for k, v in self.action_target_loss.batch_components().items():
                        self.log(f"{stage}_open/loss_{k}", v, on_epoch=True, sync_dist=True, batch_size=1)

                # pred_valid uses source-step validity (the step the
                # prediction is made from). target_valid uses next-step
                # validity (the step the GT token actually represents, after
                # fog gating) so we don't score accuracy against tokens of
                # dead or fog-hidden agents.
                pred_valid = pred["next_token_valid"] & combined_mask.unsqueeze(1)
                target_valid = (
                    filtered_agent["valid_mask"][:, 2:]
                    & vis_to_obs[:, 2:]
                    & combined_mask.unsqueeze(1)
                )
                self.TokenCls.update(
                    pred=pred["next_token_logits"],
                    pred_valid=pred_valid,
                    target=filtered_agent["gt_idx"][:, 2:],
                    target_valid=target_valid,
                )
                self.log(f"{stage}_open/acc", self.TokenCls, on_epoch=True, sync_dist=True, batch_size=1)
                self.log(f"{stage}_open/loss_motion", loss, on_epoch=True, sync_dist=True, batch_size=1)

            if self.val_closed_loop:
                self._run_closed_loop_modes(
                    data=data,
                    tokenized_map=tokenized_map,
                    tokenized_agent=tokenized_agent,
                    train_mask=train_mask,
                    observer_player=observer_player,
                    batch_idx=batch_idx,
                    stage=stage,
                )

        if self.val_closed_loop:
            if "own" in self.closed_loop_rollout_modes:
                self.log(
                    f"{stage}_closed/ADE_own_rollout", self.minADE_own_rollout,
                    on_epoch=True, sync_dist=True, batch_size=1,
                )
            if "opponent" in self.closed_loop_rollout_modes:
                self.log(
                    f"{stage}_closed/ADE_opp_rollout", self.minADE_opp_rollout,
                    on_epoch=True, sync_dist=True, batch_size=1,
                )

    def _run_closed_loop_modes(
        self, data, tokenized_map, tokenized_agent, train_mask,
        observer_player, batch_idx, stage: str = "val",
    ):
        """Run the closed-loop rollout for each enabled mode (own/opponent).

        Uses the expanded roster (opponent_keep_mode="visible_ever") so
        late-observed opponents can serve as GT-teacher-forced context.
        Fog-of-war visibility is threaded into the decoder's inference()
        via visibility_gate.
        """
        # Expanded filter: keep opponents visible at frame 16 or any future
        # step so late-observed opponents can enter attention post-reveal.
        # Permanently-unseen opponents are still dropped (they would be
        # zero-edge dead nodes under fog-of-war gating anyway).
        filt_e, obs_mask_e, opp_mask_e, keep_mask_e, vis_to_obs = (
            filter_agents_for_perspective(
                tokenized_agent, train_mask, observer_player,
                opponent_keep_mode="visible_ever",
            )
        )
        is_observer = filt_e["is_observer"]
        is_neutral = filt_e["owner"] == 16
        is_opponent = ~is_observer & ~is_neutral
        obs_sees_current = vis_to_obs[:, 1]  # visibility at frame 16

        N_total = data["agent"]["position"].shape[0]
        gt_native = data["agent"]["position"][
            :, self.num_historical_steps:, :2
        ]  # [N_total, n_step, 2]
        alive_at_current = data["agent"]["valid_mask"][
            keep_mask_e, self.num_historical_steps - 1
        ]
        target_valid_full = (
            data["agent"]["valid_mask"][keep_mask_e, self.num_historical_steps:]
            & alive_at_current.unsqueeze(1)
        )
        target = data["agent"]["position"][
            keep_mask_e, self.num_historical_steps:, :2
        ]

        # Predictions: all 4 heads (extended from prior 2-key list so the save
        # helper can decode action class + has_action). GT keys remain for the
        # GIF visualization block; not saved to disk in v1.
        _AUX_KEYS = (
            "target_pos_pred", "has_target_pos_logits",
            "action_class_logits", "has_action_logits",
            "gt_rel_target_pos", "gt_has_target_pos",
        )

        for mode in self.closed_loop_rollout_modes:
            if mode == "own":
                teacher_force_mask = ~is_observer  # opp + neutrals -> GT
                metric_scope = obs_mask_e
                metric_meter = self.minADE_own_rollout
            elif mode == "opponent":
                is_late_obs_opp = is_opponent & ~obs_sees_current
                teacher_force_mask = is_observer | is_neutral | is_late_obs_opp
                metric_scope = opp_mask_e & obs_sees_current
                metric_meter = self.minADE_opp_rollout
            else:
                raise ValueError(
                    f"Unknown closed_loop rollout mode '{mode}'. "
                    "Valid options: 'own', 'opponent'."
                )

            pred_traj_list = []
            pred_head_list = []
            aux_target_list = []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, filt_e, self.validation_rollout_sampling,
                    teacher_force_mask=teacher_force_mask,
                    visibility_gate=vis_to_obs,
                )
                pred_traj_list.append(pred["pred_traj_native"])
                pred_head_list.append(pred["pred_head_native"])
                if self.use_aux_loss:
                    aux_target_list.append({k: pred[k] for k in _AUX_KEYS})

            pred_traj = torch.stack(
                pred_traj_list, dim=1
            )  # [n_filtered, n_rollout, n_step, 2]
            pred_head = torch.stack(
                pred_head_list, dim=1
            )  # [n_filtered, n_rollout, n_step]

            # Overlay native-fps GT onto teacher-forced rows so the teacher-forced
            # side is pixel-exact (the in-decoder override is only 2Hz-accurate).
            # NOTE: save_rollout_batch reads only metric_scope rows, which are
            # disjoint from teacher_force_mask, so pred_head intentionally has
            # no parallel overlay — see assertion inside the helper.
            tf_target_gt = target[teacher_force_mask]  # [n_tf, n_step, 2]
            if tf_target_gt.shape[0] > 0:
                pred_traj[teacher_force_mask] = tf_target_gt.unsqueeze(1).expand(
                    -1, pred_traj.shape[1], -1, -1
                )

            # --- Save rollouts to disk for offline metric harness ---
            if (
                self.save_closed_rollouts
                and observer_player in self.save_rollout_observers
                and mode in self.save_rollout_modes
            ):
                save_rollout_batch(
                    save_dir=self.rollout_save_dir,
                    data=data,
                    filt_e=filt_e,
                    keep_mask_e=keep_mask_e,
                    metric_scope=metric_scope,
                    teacher_force_mask=teacher_force_mask,
                    pred_traj=pred_traj,
                    pred_head=pred_head,
                    vis_to_obs=vis_to_obs,
                    aux_target_list=aux_target_list if self.use_aux_loss else None,
                    observer_player=observer_player,
                    mode=mode,
                    file_attrs={
                        "num_historical_steps": self.num_historical_steps,
                        "num_future_steps": int(pred_traj.shape[2]),
                        "n_rollouts": int(pred_traj.shape[1]),
                        "native_fps": 16,
                        "dataset_version": self.dataset_version,
                        "model_config_hash": self.model_config_hash,
                        "stage": self.rollout_stage,
                    },
                    precision=self.rollout_save_precision,
                    global_rank=self.global_rank,
                    world_size=int(self.trainer.world_size),
                )

            scope_valid = target_valid_full & metric_scope.unsqueeze(1)
            if scope_valid.any():
                metric_meter.update(
                    pred=pred_traj, target=target, target_valid=scope_valid,
                )

            vis_enabled_this_pass = (
                self.vis_observer_player == "both"
                or (self.vis_observer_player == "p1" and observer_player == 1)
                or (self.vis_observer_player == "p2" and observer_player == 2)
            )
            if (
                vis_enabled_this_pass
                and self.global_rank == 0
                and batch_idx < self.n_vis_batch
            ):
                full_pred_traj = torch.zeros(
                    N_total, *pred_traj.shape[1:],
                    dtype=pred_traj.dtype, device=pred_traj.device,
                )
                full_pred_traj[keep_mask_e] = pred_traj

                full_pred_head = torch.zeros(
                    N_total, *pred_head.shape[1:],
                    dtype=pred_head.dtype, device=pred_head.device,
                )
                full_pred_head[keep_mask_e] = pred_head

                # Units of interest for this (observer, mode): the same
                # metric_scope rows the rollout I/O saves. Scattered back
                # onto the full N_total roster so extract_scenario_data can
                # narrow diamond/pred-trail rendering to these agents.
                target_mask_full = torch.zeros(
                    N_total, dtype=torch.bool, device=metric_scope.device,
                )
                target_mask_full[keep_mask_e] = metric_scope

                n_scenarios = min(self.n_vis_scenario, data.num_graphs)
                n_rollouts_vis = min(self.n_vis_rollout, full_pred_traj.shape[1])
                obs_tag = f"obs_p{observer_player}"
                mode_tag = f"mode_{mode}"
                for i_sc in range(n_scenarios):
                    for i_roll in range(n_rollouts_vis):
                        aux_data = None
                        if aux_target_list:
                            aux_data = {}
                            for k, v in aux_target_list[i_roll].items():
                                full_v = torch.zeros(
                                    N_total, *v.shape[1:],
                                    dtype=v.dtype, device=v.device,
                                )
                                full_v[keep_mask_e] = v
                                aux_data[k] = full_v
                        sc_data = extract_scenario_data(
                            data, full_pred_traj, i_sc, i_roll,
                            num_historical_steps=self.num_historical_steps,
                            aux_target_data=aux_data,
                            map_data_dir=self.token_processor.map_data_dir,
                            pred_valid_mask=keep_mask_e,
                            observer_player=observer_player,
                            target_mask=target_mask_full,
                            pred_head=full_pred_head,
                        )
                        save_dir = (
                            self.gif_dir
                            / f"batch_{batch_idx:02d}"
                            / f"scenario_{i_sc:02d}"
                            / obs_tag
                            / mode_tag
                        )
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_scenario_gif(
                            **sc_data,
                            save_path=str(save_dir / f"rollout_{i_roll:02d}.gif"),
                            num_historical_steps=self.num_historical_steps,
                        )

    def setup(self, stage):
        # Lightning calls setup with stage in {"fit", "validate", "test", "predict"}
        # or sometimes None (e.g. trainer.tune). Guard the rollout dir for both
        # eval stages; every rank checks (shared FS → same answer → consistent
        # raise across ranks → clean DDP shutdown without barrier hang).
        if stage in ("validate", "test", None) and self.save_closed_rollouts:
            self.rollout_stage = "test" if stage == "test" else "val"
            self.rollout_save_dir.mkdir(parents=True, exist_ok=True)
            existing = list(self.rollout_save_dir.glob("*.h5"))
            if existing:
                raise RuntimeError(
                    f"Rollout dir {self.rollout_save_dir} already contains "
                    f"{len(existing)} .h5 files. Use a fresh dir."
                )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]
