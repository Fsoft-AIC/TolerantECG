from typing import Any, Dict, Tuple
import io
import json
from PIL import Image

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision, MultilabelF1Score

from src.models.metric.f1_max import MultilabelF1Max
from src.utils.ecg_utils.tensor import clean_memory


class DinoClipFinetuneLitModule(LightningModule):
    def __init__(
        self,
        ecg_encoder: torch.nn.Module,
        loss_function: torch.nn.Module = None,
        encoder_ckpt_path: str = None,
        is_finetune: bool = True,
        num_classes: int = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        **kwargs
    ) -> None:
        """
        is_finetune: if False, perform Linear Probing
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, 
                                  ignore=['ecg_encoder', "loss_function", "encoder_ckpt_path"])

        self.ecg_encoder = ecg_encoder
        if encoder_ckpt_path is not None:
            state_dict = torch.load(encoder_ckpt_path, map_location=self.device)
            self.ecg_encoder.load_state_dict(state_dict)

        if not is_finetune:
            print("Linear Probing")
        else:
            print("Fine-tuning")

        for param in self.ecg_encoder.parameters():
            param.requires_grad = is_finetune

        # classifier
        self.out_layer = torch.nn.Linear(self.ecg_encoder.embed_dim, num_classes)
        # loss function
        self.multilabel_criterion = loss_function

        # for averaging loss across batches
        self.train_loss = MeanMetric(nan_strategy='error')
        self.val_loss = MeanMetric(nan_strategy='error')

        self.train_f1 = MultilabelF1Score(num_labels=num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=num_classes)

        self.train_fmax = MultilabelF1Max()
        self.val_fmax = MultilabelF1Max()

        self.train_auroc = MultilabelAUROC(num_labels=num_classes)
        self.val_auroc = MultilabelAUROC(num_labels=num_classes)

        self.train_auprc = MultilabelAveragePrecision(num_labels=num_classes)
        self.val_auprc = MultilabelAveragePrecision(num_labels=num_classes)

        # for tracking best so far validation accuracy
        self.val_best_f1 = MaxMetric(nan_strategy="error")
        self.val_best_fmax = MaxMetric(nan_strategy="error")
        self.val_best_auroc = MaxMetric(nan_strategy="error")
        self.val_best_auprc = MaxMetric(nan_strategy="error")

        self.data_types = ["clean", "mask", "noise", "mask_noise"]
        for dtype in self.data_types:
            setattr(self, f"test_loss_{dtype}", MeanMetric(nan_strategy='error'))
            setattr(self, f"test_f1_{dtype}", MultilabelF1Score(num_labels=num_classes))
            setattr(self, f"test_fmax_{dtype}", MultilabelF1Max())
            setattr(self, f"test_auroc_{dtype}", MultilabelAUROC(num_labels=num_classes))
            setattr(self, f"test_auprc_{dtype}", MultilabelAveragePrecision(num_labels=num_classes))

    def forward(self, 
                ecgs
                ) -> torch.Tensor:
        """
        :param ecg: A tensor of ecg signals/images.
        :return: A tensor of logits.
        """
        if not self.hparams.is_finetune:
            with torch.no_grad():
                ecgs = self.ecg_encoder(ecgs)
        else:
            ecgs = self.ecg_encoder(ecgs)
        return self.out_layer(ecgs)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_f1.reset()
        self.val_fmax.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()

        self.val_best_f1.reset()
        self.val_best_fmax.reset()
        self.val_best_auroc.reset()
        self.val_best_auprc.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ecgs, _, labels, paths = batch

        logits = self.forward(ecgs)
        loss = self.multilabel_criterion(logits, labels.float())

        return loss, logits, labels, ecgs, paths

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch_size = len(batch)
        loss, logits, labels, _, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_f1(logits, labels)
        self.train_fmax(torch.sigmoid(logits), labels)
        self.train_auroc(logits, labels)
        self.train_auprc(logits, labels)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/fmax", self.train_fmax, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def log_confusion(self, key, confuse):
        fig_, ax_ = confuse.plot(labels=self.labels)

        buf = io.BytesIO()
        fig_.savefig(buf, format='jpg')
        buf.seek(0)

        conf = Image.open(buf).convert("RGB")

        self.logger.log_image(key=key, images=[conf], mode=['RGB'])
        clean_memory(fig_, conf)

    def on_train_epoch_end(self) -> None:
        pass
        # if self.num_classes is None:
        #     self.log_confusion("train/confusion", self.train_confusion)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_size = len(batch[-1])
        loss, logits, labels, ecgs, paths = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_f1(logits, labels)
        self.val_fmax(torch.sigmoid(logits), labels)
        self.val_auroc(logits, labels)
        self.val_auprc(logits, labels)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/fmax", self.val_fmax, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # return loss or backpropagation will fail
        return logits, labels, ecgs, paths

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        f1 = self.val_f1.compute()
        self.val_best_f1(f1)

        fmax = self.val_fmax.compute()
        self.val_best_fmax(fmax)

        auroc = self.val_auroc.compute()
        self.val_best_auroc(auroc)

        auprc = self.val_auprc.compute()
        self.val_best_auprc(auprc)

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_best_f1.compute(), sync_dist=True, prog_bar=True)
        self.log("val/fmax_best", self.val_best_fmax.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auroc_best", self.val_best_auroc.compute(), sync_dist=True, prog_bar=True)
        self.log("val/auprc_best", self.val_best_auprc.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_size = len(batch[-1])
        loss, logits, labels, ecgs, paths = self.model_step(batch)

        # update and log metrics
        dtype = self.data_types[dataloader_idx]
        getattr(self, f"test_loss_{dtype}")(loss)
        getattr(self, f"test_f1_{dtype}")(logits, labels)
        getattr(self, f"test_fmax_{dtype}")(torch.sigmoid(logits), labels)
        getattr(self, f"test_auroc_{dtype}")(logits, labels)
        getattr(self, f"test_auprc_{dtype}")(logits, labels)

        self.log(f"test/loss_{dtype}", getattr(self, f"test_loss_{dtype}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log(f"test/f1_{dtype}", getattr(self, f"test_f1_{dtype}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log(f"test/fmax_{dtype}", getattr(self, f"test_fmax_{dtype}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log(f"test/auroc_{dtype}", getattr(self, f"test_auroc_{dtype}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log(f"test/auprc_{dtype}", getattr(self, f"test_auprc_{dtype}"), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

        # return loss or backpropagation will fail
        return logits, labels, ecgs, paths

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass
        # if self.num_classes is None:
        #     for dtype in self.data_types:
        #         self.log_confusion(f"test/confusion_{dtype}", getattr(self, f"test_confusion_{dtype}"))

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.ecg_encoder = torch.compile(self.ecg_encoder)
            self.out_layer = torch.compile(self.out_layer)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = self.hparams.optimizer(params=params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
