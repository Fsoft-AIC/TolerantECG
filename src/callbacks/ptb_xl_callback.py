import wandb
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
import json

import torch
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks import Callback

from src.utils.ecg_utils.tensor import denormalize_batch, half_image
from src.utils.ecg_utils.ecg_plot import signal_to_image


def list_to_string(ls):
    return "[" + ", ".join(map(str, ls)) + "]"

class PtbXlCallback(Callback):
    def __init__(self, threshold=0.5, num_test_sample=100, sample_rate=500):
        self.num_batch_to_log = 1

        self.threshold = threshold

        self.num_test_sample = num_test_sample
        self.test_table = []
        self.has_log_test = False
        self.sample_rate = sample_rate

    def setup(self, trainer, pl_module, stage):
        self.logger = trainer.logger

        with open('id_to_label.json', 'r') as json_file:
            self.id_to_label = json.load(json_file)

    def to_grid(self, image):
        image = make_grid(image.unsqueeze(1), nrow=2)   # B, F, T -> B, 1, F, T -> 3, H, W
        image = image.permute(1, 2, 0)
        return image.numpy()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:

        if self.num_batch_to_log <= 0:
            return

        preds, labels, ecgs, paths = outputs
        preds = preds.cpu()
        labels = labels.cpu()

        # if ECG is 1D
        if len(ecgs.shape) == 3:
            ecgs = [torch.tensor(signal_to_image(ecg.cpu(), self.sample_rate, "", True, True, False), dtype=torch.uint8) for ecg in ecgs]
        # if ECG id 2D
        else:
            # if RGB
            if ecgs.size(1) == 3:
                ecgs = (denormalize_batch(ecgs).permute(0, 2, 3, 1) * 255).cpu().type(torch.uint8)
            # if gray-scale
            else:
                ecgs = (ecgs.permute(0, 2, 3, 1) * 255).cpu().type(torch.uint8)

        table = []

        for image, pred, label, path in zip(ecgs, preds, labels, paths):
            probs = torch.sigmoid(pred)

            ids,  = torch.where(probs >= self.threshold)
            pred = [self.id_to_label[str(id_.item())] for id_ in ids]

            ids, = torch.where(label == 1)
            label = [self.id_to_label[str(id_.item())] for id_ in ids]

            probs = list_to_string(probs.tolist())
            pred = list_to_string(pred)
            label = list_to_string(label)

            if image.shape[-1] == 12:     # H, W, C
                image = image.permute(2, 0, 1)  # C, H, W
                lead_2 = image[1].numpy()
                # use Channel as batch size
                # image = make_grid(image.unsqueeze(1), nrow=2)   # C, H, W -> C, 1, H, W
                # image = image.numpy()
                image = self.to_grid(image)
                table.append([path, wandb.Image(image), probs, pred, label, wandb.Image(lead_2, mode="L")])

            else:
                image = half_image(image.numpy())
                table.append([path, wandb.Image(image), probs, pred, label])

        columns = ["File path", 'ECG image', "Probabilities", "Predictions", "Labels"]
        if len(table[0]) == 6:
            columns.append("Lead II")

        self.logger.log_table(
            key='Validation Prediction',
            data=table,
            columns=columns
        )

        self.num_batch_to_log -= 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.num_batch_to_log = 1

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if self.num_test_sample <= 0:
            if not self.has_log_test:
                columns = ["File path", 'ECG image', "Probabilities", "Predictions", "Labels"]
                if len(self.test_table[0]) == 6:
                    columns.append("Lead II")

                self.logger.log_table(
                    key='Test Prediction',
                    data=self.test_table,
                    columns=columns
                )

                self.has_log_test = True
            return
        
        preds, labels, ecgs, paths = outputs
        preds = preds.cpu()
        labels = labels.cpu()

        # if ECG is 1D
        if len(ecgs.shape) == 3:
            ecgs = [torch.tensor(signal_to_image(ecg.cpu(), 500, "", True, True, False), dtype=torch.uint8) for ecg in ecgs]
        # if ECG id 2D
        else:
            # if RGB
            if ecgs.size(1) == 3:
                ecgs = (denormalize_batch(ecgs).permute(0, 2, 3, 1) * 255).cpu().type(torch.uint8)
            # if gray-scale
            else:
                ecgs = (ecgs.permute(0, 2, 3, 1) * 255).cpu().type(torch.uint8)

        for image, pred, label, path in zip(ecgs, preds, labels, paths):
            if self.num_test_sample <= 0:
                break

            probs = torch.sigmoid(pred)

            ids,  = torch.where(probs >= self.threshold)
            pred = [self.id_to_label[str(id_.item())] for id_ in ids]

            ids, = torch.where(label == 1)
            label = [self.id_to_label[str(id_.item())] for id_ in ids]

            probs = list_to_string(probs.tolist())
            pred = list_to_string(pred)
            label = list_to_string(label)

            if image.shape[-1] == 12:     # H, W, C
                image = image.permute(2, 0, 1)  # C, H, W
                lead_2 = image[1].numpy()
                image = self.to_grid(image)

                self.test_table.append([path, wandb.Image(image), probs, pred, label, wandb.Image(lead_2, mode="L")])

            else:
                image = half_image(image.numpy())
                self.test_table.append([path, wandb.Image(image), probs, pred, label])

            self.num_test_sample -= 1

    def on_test_end(self, trainer, pl_module):
        if not self.has_log_test:
            columns = ["File path", 'ECG image', "Probabilities", "Predictions", "Labels"]
            if len(self.test_table[0]) == 6:
                columns.append("Lead II")

            self.logger.log_table(
                key='Test Prediction',
                data=self.test_table,
                columns=columns
            )

            self.has_log_test = True
