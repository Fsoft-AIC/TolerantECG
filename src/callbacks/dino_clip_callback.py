import wandb
from typing import Any

import lightning as pl
from lightning.pytorch.callbacks import Callback

from src.utils.ecg_utils.tensor import half_image
from src.utils.ecg_utils.ecg_plot import signal_to_image


class DinoClipCallback(Callback):
    def __init__(self, sample_rate = 500):
        self.num_batch_mask_to_log = 1
        self.num_batch_noise_to_log = 1
        self.max_sample = 12
        self.sample_rate = sample_rate

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        if dataloader_idx == 0:
            if self.num_batch_mask_to_log <= 0:
                return
        else:
            if self.num_batch_noise_to_log <= 0:
                return

        ecg_image, mask_ecg_image, caption_target, ecg_paths = outputs

        ecg_image = [signal_to_image(ecg.cpu(), self.sample_rate, "", True, True, False) for ecg in ecg_image]
        global_mask_ecg = mask_ecg_image[0]
        local_mask_ecg = mask_ecg_image[-1]
        global_mask_ecg = [signal_to_image(ecg.cpu(), self.sample_rate, "", True, True, False) for ecg in global_mask_ecg]
        local_mask_ecg = [signal_to_image(ecg.cpu(), self.sample_rate, "", True, True, False) for ecg in local_mask_ecg]

        table = []

        for image, global_image, local_image, target, path in zip(ecg_image, global_mask_ecg, local_mask_ecg, caption_target, ecg_paths):
            image = half_image(image)
            global_image = half_image(global_image)
            local_image = half_image(local_image)
            table.append([path, wandb.Image(image), wandb.Image(global_image), wandb.Image(local_image), target])
            
            if len(table) >= self.max_sample:
                break
        
        if dataloader_idx == 0:
            columns = ["File path", 'ECG', "Global Mask", "Local Mask", "Caption target"]
            key = "Validation Mask"
            self.num_batch_mask_to_log -= 1
        else:
            columns = ["File path", 'ECG', "Global Noise", "Local Noise", "Caption target"]
            key = "Validation Noise"
            self.num_batch_noise_to_log -= 1

        self.logger.log_table(
            key=key,
            data=table,
            columns=columns
        )


    # def on_validation_epoch_start(self, trainer, pl_module):
    #     self.num_batch_to_log = 1