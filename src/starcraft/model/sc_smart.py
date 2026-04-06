"""StarCraft SMART model (LightningModule)."""

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import TokenCls, minADE
from src.starcraft.metrics.sc_action_target_loss import SCActionTargetLoss
from src.starcraft.metrics.sc_cross_entropy import SCCrossEntropy
from src.starcraft.modules.sc_decoder import SCDecoder
from src.starcraft.tokens.sc_token_processor import SCTokenProcessor
from src.starcraft.utils.vis_starcraft import extract_scenario_data, save_scenario_gif
from src.smart.utils.finetune import set_model_for_finetuning


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
        self.TokenCls = TokenCls(max_guesses=5)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout

        self.gif_dir = (
            Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            / "gifs"
        )

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map, tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss_motion = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],
            token_traj=tokenized_agent["token_traj"],  # [n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],
        )
        if self.use_aux_loss:
            loss_aux = self.action_target_loss(
                **pred, train_mask=data["agent"]["train_mask"],
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
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],
                token_traj=tokenized_agent["token_traj"],
            )
            if self.use_aux_loss:
                self.action_target_loss(**pred)
                for k, v in self.action_target_loss.batch_components().items():
                    self.log(f"val_open/loss_{k}", v, on_epoch=True, sync_dist=True, batch_size=1)

            self.TokenCls.update(
                pred=pred["next_token_logits"],
                pred_valid=pred["next_token_valid"],
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log("val_open/acc", self.TokenCls, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/loss_motion", loss, on_epoch=True, sync_dist=True, batch_size=1)

        if self.val_closed_loop:
            pred_traj = []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_native"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]

            # Agents not alive at the current frame have no valid initial position,
            # so their predictions are garbage (centered at origin).  Exclude them
            # from metrics and visualization, mirroring the train_mask filter.
            alive_at_current = data["agent"]["valid_mask"][
                :, self.num_historical_steps - 1
            ]
            target_valid = (
                data["agent"]["valid_mask"][:, self.num_historical_steps :]
                & alive_at_current.unsqueeze(1)
            )

            self.minADE.update(
                pred=pred_traj,
                target=data["agent"]["position"][
                    :, self.num_historical_steps :, :pred_traj.shape[-1]
                ],
                target_valid=target_valid,
            )
            self.log("val_closed/ADE", self.minADE, on_epoch=True, sync_dist=True, batch_size=1)

            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                n_scenarios = min(self.n_vis_scenario, data.num_graphs)
                n_rollouts_vis = min(self.n_vis_rollout, pred_traj.shape[1])
                for i_sc in range(n_scenarios):
                    for i_roll in range(n_rollouts_vis):
                        sc_data = extract_scenario_data(
                            data, pred_traj, i_sc, i_roll,
                            num_historical_steps=self.num_historical_steps,
                        )
                        save_dir = (
                            self.gif_dir
                            / f"batch_{batch_idx:02d}"
                            / f"scenario_{i_sc:02d}"
                        )
                        save_dir.mkdir(parents=True, exist_ok=True)
                        gif_path = save_dir / f"rollout_{i_roll:02d}.gif"
                        save_scenario_gif(
                            **sc_data,
                            save_path=str(gif_path),
                            num_historical_steps=self.num_historical_steps,
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
