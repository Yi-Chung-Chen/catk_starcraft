"""StarCraft SMART model (LightningModule)."""

import math

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import TokenCls, minADE
from src.starcraft.metrics.sc_cross_entropy import SCCrossEntropy
from src.starcraft.modules.sc_decoder import SCDecoder
from src.starcraft.tokens.sc_token_processor import SCTokenProcessor
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
        self.encoder = SCDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.training_loss = SCCrossEntropy(**model_config.training_loss)
        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map, tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],
            token_traj=tokenized_agent["token_traj"],  # [n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)
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

            self.TokenCls.update(
                pred=pred["next_token_logits"],
                pred_valid=pred["next_token_valid"],
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log("val_open/acc", self.TokenCls, on_epoch=True, sync_dist=True, batch_size=1)
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        if self.val_closed_loop:
            pred_traj = []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_native"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            self.minADE.update(
                pred=pred_traj,
                target=data["agent"]["position"][
                    :, self.num_historical_steps :, :pred_traj.shape[-1]
                ],
                target_valid=data["agent"]["valid_mask"][:, self.num_historical_steps :],
            )
            self.log("val_closed/ADE", self.minADE, on_epoch=True, sync_dist=True, batch_size=1)

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
