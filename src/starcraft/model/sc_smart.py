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
from src.starcraft.tokens.sc_token_processor import SCTokenProcessor, filter_agents_for_perspective
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
        self.minADE_observer = minADE()
        self.minADE_opponent = minADE()
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
        train_mask = data["agent"]["train_mask"]

        # Randomly choose one observer perspective per step
        observer_player = 1 if torch.rand(1).item() < 0.5 else 2
        filtered_agent, obs_mask, opp_mask, _ = filter_agents_for_perspective(
            tokenized_agent, train_mask, observer_player,
        )

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, filtered_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map, filtered_agent,
                sampling_scheme=self.training_rollout_sampling,
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
        tokenized_map, tokenized_agent = self.token_processor(data)
        train_mask = data["agent"]["train_mask"]

        # Run both perspectives for complete metrics
        for observer_player in [1, 2]:
            filtered_agent, obs_mask, opp_mask, keep_mask = filter_agents_for_perspective(
                tokenized_agent, train_mask, observer_player,
            )
            combined_mask = obs_mask | opp_mask

            if self.val_open_loop:
                pred = self.encoder(tokenized_map, filtered_agent)
                loss = self.training_loss(
                    **pred,
                    token_agent_shape=filtered_agent["token_agent_shape"],
                    token_traj=filtered_agent["token_traj"],
                    train_mask=combined_mask,
                )
                if self.use_aux_loss:
                    self.action_target_loss(**pred, train_mask=obs_mask)
                    for k, v in self.action_target_loss.batch_components().items():
                        self.log(f"val_open/loss_{k}", v, on_epoch=True, sync_dist=True, batch_size=1)

                self.TokenCls.update(
                    pred=pred["next_token_logits"],
                    pred_valid=pred["next_token_valid"] & combined_mask.unsqueeze(1),
                    target=filtered_agent["gt_idx"][:, 2:],
                    target_valid=filtered_agent["valid_mask"][:, 2:] & combined_mask.unsqueeze(1),
                )
                self.log("val_open/acc", self.TokenCls, on_epoch=True, sync_dist=True, batch_size=1)
                self.log("val_open/loss_motion", loss, on_epoch=True, sync_dist=True, batch_size=1)

            if self.val_closed_loop:
                pred_traj_list = []
                for _ in range(self.n_rollout_closed_val):
                    pred = self.encoder.inference(
                        tokenized_map, filtered_agent, self.validation_rollout_sampling
                    )
                    pred_traj_list.append(pred["pred_traj_native"])

                pred_traj = torch.stack(pred_traj_list, dim=1)  # [n_filtered, n_rollout, n_step, 2]

                # Align targets to filtered agent indices using keep_mask
                target = data["agent"]["position"][keep_mask, self.num_historical_steps:, :pred_traj.shape[-1]]
                alive_at_current = data["agent"]["valid_mask"][keep_mask, self.num_historical_steps - 1]
                target_valid = (
                    data["agent"]["valid_mask"][keep_mask, self.num_historical_steps:]
                    & alive_at_current.unsqueeze(1)
                )

                # Observer ADE
                obs_valid = target_valid & obs_mask.unsqueeze(1)
                self.minADE_observer.update(pred=pred_traj, target=target, target_valid=obs_valid)

                # Opponent ADE
                opp_valid = target_valid & opp_mask.unsqueeze(1)
                if opp_valid.any():
                    self.minADE_opponent.update(pred=pred_traj, target=target, target_valid=opp_valid)

        if self.val_closed_loop:
            self.log("val_closed/ADE_observer", self.minADE_observer, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_closed/ADE_opponent", self.minADE_opponent, on_epoch=True, sync_dist=True, batch_size=1)

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
