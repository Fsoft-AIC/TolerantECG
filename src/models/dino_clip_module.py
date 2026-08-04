import io
from typing import Any, Dict, Tuple
import os

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from src.models.loss.dino_loss import DINOLoss
from src.models.metric.cos import CosineSimilarity

from src.models.text_encoder.hugging_face_bert import HuggingFaceBert
from src.utils.ecg_utils.tokenizer import HuggingFaceTokenizer

from src.utils.ecg_utils.tensor import clean_memory

from src.models.dino.head import DINOHead
from src.models.metric.dino_cos import DinoCos
from src.models.metric.dino_kl import DinoKL

from src.data.mimic_dino_mask_noise_datamodule import num_global, num_local


class DinoClipLitModule(LightningModule):
    def __init__(
        self,
        ecg_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        loss_function: torch.nn.Module = None,
        alpha: int = 1,
        beta: int = 1,
        out_dim: int = 768,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, 
                                  ignore=['ecg_encoder', 'text_encoder', "loss_function"])

        self.global_num = kwargs.get("global_num", num_global)
        self.local_num = kwargs.get("local_num", num_local)
        print("Global", self.global_num)
        print("Local", self.local_num)

        self.student_ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder

        ecg_embed_dim = ecg_encoder.embed_dim
        text_embed_dim = text_encoder.embed_dim

        self.teacher_mask_ecg_encoder = deepcopy(self.student_ecg_encoder)
        self.teacher_noise_ecg_encoder = deepcopy(self.student_ecg_encoder)

        self.teacher_mask_dino_head = DINOHead(in_dim=ecg_embed_dim, out_dim=out_dim, use_bn=False, nlayers=1)
        self.student_mask_dino_head = DINOHead(in_dim=ecg_embed_dim, out_dim=out_dim, use_bn=False, nlayers=1)

        self.teacher_noise_dino_head = DINOHead(in_dim=ecg_embed_dim, out_dim=out_dim, use_bn=False, nlayers=1)
        self.student_noise_dino_head = DINOHead(in_dim=ecg_embed_dim, out_dim=out_dim, use_bn=False, nlayers=1)

        self.student_clip_head = nn.Linear(ecg_embed_dim, out_dim, bias=False)
        self.text_clip_head = nn.Linear(text_embed_dim, out_dim, bias=False)

        self.teacher_mask_ecg_encoder.load_state_dict(self.student_ecg_encoder.state_dict())
        self.teacher_mask_dino_head.load_state_dict(self.student_mask_dino_head.state_dict())

        self.teacher_noise_ecg_encoder.load_state_dict(self.student_ecg_encoder.state_dict())
        self.teacher_noise_dino_head.load_state_dict(self.student_noise_dino_head.state_dict())

        for p in self.teacher_mask_ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_noise_ecg_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_mask_dino_head.parameters():
            p.requires_grad = False
        for p in self.teacher_noise_dino_head.parameters():
            p.requires_grad = False
        
        self.text_tokenizer = HuggingFaceTokenizer(self.text_encoder.model_name)
        print(f"{self.text_encoder.model_name} tokenizer")

        # loss function
        self.momentum_teacher = 0.996
        self.teacher_temperature = 0.07
        self.student_temperature = 0.1
        self.momentum_center = 0.9

        self.constrastive_criterion = loss_function

        if (self.global_num is not None) or (self.local_num is not None):
            self.dino_mask_criterion = DINOLoss(out_dim=out_dim,
                                            n_local_crops=self.local_num,
                                            n_global_crops=self.global_num,
                                            teacher_temp=self.teacher_temperature,
                                            student_temp=self.student_temperature,
                                            center_momentum=self.momentum_center)
            self.dino_noise_criterion = DINOLoss(out_dim=out_dim,
                                            n_local_crops=self.local_num,
                                            n_global_crops=self.global_num,
                                            teacher_temp=self.teacher_temperature,
                                            student_temp=self.student_temperature,
                                            center_momentum=self.momentum_center)

        # for averaging loss across batches
        self.train_loss = MeanMetric(nan_strategy='error')
        self.val_loss = MeanMetric(nan_strategy='error')
        self.test_loss = MeanMetric(nan_strategy='error')

        self.train_cos_ecg_txt = CosineSimilarity()
        self.val_cos_ecg_txt = CosineSimilarity()
        self.test_cos_ecg_txt = CosineSimilarity()

        self.val_best_cos_ecg_txt = MaxMetric(nan_strategy='error')

        if (self.global_num is not None) and (self.local_num is not None):
            self.train_cos_ecg_mask = DinoCos(self.local_num, self.global_num)
            self.val_cos_ecg_mask = DinoCos(self.local_num, self.global_num)
            self.test_cos_ecg_mask = DinoCos(self.local_num, self.global_num)

            self.train_cos_ecg_noise = DinoCos(self.local_num, self.global_num)
            self.val_cos_ecg_noise = DinoCos(self.local_num, self.global_num)
            self.test_cos_ecg_noise = DinoCos(self.local_num, self.global_num)

            self.train_kl_ecg_mask = DinoKL(self.local_num, self.global_num)
            self.val_kl_ecg_mask = DinoKL(self.local_num, self.global_num)
            self.test_kl_ecg_mask = DinoKL(self.local_num, self.global_num)

            self.train_kl_ecg_noise = DinoKL(self.local_num, self.global_num)
            self.val_kl_ecg_noise = DinoKL(self.local_num, self.global_num)
            self.test_kl_ecg_noise = DinoKL(self.local_num, self.global_num)

            # for tracking best so far validation accuracy
            self.val_best_cos_ecg_mask = MaxMetric(nan_strategy='error')
            self.val_best_kl_ecg_mask = MinMetric(nan_strategy='error')
            self.val_best_cos_ecg_noise = MaxMetric(nan_strategy='error')
            self.val_best_kl_ecg_noise = MinMetric(nan_strategy='error')

    def clip_forward(self,
                ecg: torch.Tensor,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_embedding: bool = False,
                ) -> torch.Tensor:
        """
        :param ecg: A tensor of ecg signals.
        :param text: A tensor of tokenized text description.
        :return: A tensor of logits.
        """
        ecg_embed = self.student_ecg_encoder(ecg)
        _, text_embed = self.text_encoder(input_ids, token_type_ids, attention_mask)

        if return_embedding:
            return ecg_embed, text_embed

        return self.student_clip_head(ecg_embed), \
                self.text_clip_head(text_embed)
    
    def ecg_forward(self, ecg: torch.Tensor, return_embedding=True):
        ecg_embed = self.student_ecg_encoder(ecg)
        if return_embedding:
            return ecg_embed
        return self.student_clip_head(ecg_embed)

    def text_forward(self,
                    input_ids: torch.Tensor,
                    token_type_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    return_embedding: bool = False):
        
        _, text_embed = self.text_encoder(input_ids, token_type_ids, attention_mask)
        if return_embedding:
            return text_embed
        return self.text_clip_head(text_embed)

    def dino_forward(self,
                augment_ecgs: list,
                dataloader_idx: int,
                return_embedding: bool = False,
                ):
        """
        augment_ecgs: a list of augmented tensor. Each tensor has shape [B, 12, 5000]
        The 1st 2 elements should be global view
        while others are local view
        """
        # pass all global & local view for student
        augment_ecgs = list(torch.unbind(augment_ecgs, dim=0))
        student_embed = self.student_ecg_encoder(torch.cat(augment_ecgs).to(self.device))

        with torch.no_grad():
            # only pass in global view for teacher
            teacher_input = torch.cat(augment_ecgs[: self.global_num]).to(self.device)
            if dataloader_idx == 0:
                teacher_embed = self.teacher_mask_ecg_encoder(teacher_input).detach()
            else:
                teacher_embed = self.teacher_noise_ecg_encoder(teacher_input).detach()

        if return_embedding:
            return student_embed, \
                    teacher_embed
        
        if dataloader_idx == 0:
            student_embed = self.student_mask_dino_head(student_embed)
            with torch.no_grad():
                teacher_embed = self.teacher_mask_dino_head(teacher_embed).detach()
        else:
            student_embed = self.student_noise_dino_head(student_embed)
            with torch.no_grad():
                teacher_embed = self.teacher_noise_dino_head(teacher_embed).detach()

        return student_embed, teacher_embed

    def forward(self, ecg: torch.Tensor):
        return self.student_ecg_encoder(ecg)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        self.val_cos_ecg_txt.reset()
        self.val_best_cos_ecg_txt.reset()

        if hasattr(self, 'val_cos_ecg_mask'):
            self.val_cos_ecg_mask.reset()
            self.val_cos_ecg_noise.reset()
            self.val_kl_ecg_mask.reset()
            self.val_kl_ecg_noise.reset()
            self.val_best_cos_ecg_mask.reset()
            self.val_best_cos_ecg_noise.reset()
            self.val_best_kl_ecg_mask.reset()
            self.val_best_kl_ecg_noise.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], dataloader_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        model_input, file_paths = batch
        input_ids = model_input['input_ids']    # B, Seq, Dim
        token_type_ids = model_input['token_type_ids']  # B, Seq
        attention_mask = model_input['attention_mask']  # B, Seq

        augment_ecgs = model_input['augment_ecgs']  # List of [B, 12, 5000] Tensor (len = 10)
        ecgs = model_input['ecgs']        # B, 12, 5000

        student_embeddings, teacher_embeddings = self.dino_forward(augment_ecgs,
                                                                    dataloader_idx,
                                                                    return_embedding=False)
        ecg_embeddings, text_embeddings = self.clip_forward(ecgs, 
                                                            input_ids, 
                                                            token_type_ids, 
                                                            attention_mask, 
                                                            return_embedding=False)

        if dataloader_idx == 0:
            loss_dino = self.dino_mask_criterion(student_embeddings,
                                            teacher_embeddings,
                                            self.training)
        else:
            loss_dino = self.dino_noise_criterion(student_embeddings,
                                            teacher_embeddings,
                                            self.training)
        loss_contrastive = self.constrastive_criterion(ecg_embeddings, text_embeddings) * 2.0

        text = self.text_tokenizer.batch_decode(input_ids)

        # clean_memory(input_ids, token_type_ids, attention_mask)

        if self.training:
            opt = self.optimizers()
            sche = self.lr_schedulers()

            opt.zero_grad()
            loss_total = self.hparams.alpha * loss_contrastive  + self.hparams.beta * loss_dino
            self.manual_backward(loss_total)
            opt.step()

            # step scheduler only after done looking at 2 datasets
            if dataloader_idx == 1:
                sche.step()
                # self.constrastive_criterion.logit_scale.data = torch.clamp(self.constrastive_criterion.logit_scale.data, 0, 4.6052) 

            m = self.momentum_teacher
            if dataloader_idx == 0:
                with torch.no_grad():
                    for param_q, param_k in zip(self.student_ecg_encoder.parameters(), self.teacher_mask_ecg_encoder.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                    for param_q, param_k in zip(self.student_mask_dino_head.parameters(), self.teacher_mask_dino_head.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            else:
                with torch.no_grad():
                    for param_q, param_k in zip(self.student_ecg_encoder.parameters(), self.teacher_noise_ecg_encoder.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                    for param_q, param_k in zip(self.student_noise_dino_head.parameters(), self.teacher_noise_dino_head.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)     

        return loss_dino, loss_contrastive, \
                student_embeddings, teacher_embeddings, \
                ecg_embeddings, text_embeddings, \
                ecgs, augment_ecgs, \
                text, file_paths

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int, 
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        for dataloader_idx, batch_item in enumerate(batch):
            batch_size = len(batch_item)
            loss_dino, loss_contrastive, student_embeddings, teacher_embeddings, ecg_embeddings, text_embeddings, _, _, _, _ = self.model_step(batch_item, dataloader_idx)

            loss = self.hparams.alpha * loss_dino + self.hparams.beta * loss_contrastive
            self.train_loss(loss)

            # update and log metrics
            self.train_cos_ecg_txt(ecg_embeddings, text_embeddings)

            if dataloader_idx == 0:
                self.train_cos_ecg_mask(student_embeddings, teacher_embeddings)
                self.train_kl_ecg_mask(F.softmax(student_embeddings, dim=-1), 
                                    F.softmax(teacher_embeddings, dim=-1))
            else:
                self.train_cos_ecg_noise(student_embeddings, teacher_embeddings)
                self.train_kl_ecg_noise(F.softmax(student_embeddings, dim=-1), 
                                    F.softmax(teacher_embeddings, dim=-1))

            if dataloader_idx == 0:
                self.log("train/loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
                self.log("train/loss_total_mask", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
                self.log("train/loss_dino_mask", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            else:
                self.log("train/loss_total_noise", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
                self.log("train/loss_dino_noise", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

        self.log("train/cos_ecg_txt", self.train_cos_ecg_txt, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log("train/cos_ecg_noise", self.train_cos_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log("train/kl_ecg_noise", self.train_kl_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log("train/cos_ecg_mask", self.train_cos_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        self.log("train/kl_ecg_mask", self.train_kl_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

        for name, param in self.constrastive_criterion.named_parameters():
            self.log(f"train/{name}", param.data.exp().item(), on_step=True)

    def get_heat(self, matrix):
        if matrix.shape[0] > 24:
            matrix = matrix[:24, :24]
        num_rows, num_cols = matrix.shape
        base_font_size = 17  # Base font size
        font_size = min(base_font_size, base_font_size * 10 / max(num_rows, num_cols))

        plt.figure(figsize=(10, 7))
        heatmap = sns.heatmap(matrix.to(torch.float32), 
                            annot=True, 
                            fmt="0.2f", 
                            cmap="hot", 
                            annot_kws={'size': font_size}, 
                            vmin=-1.0, 
                            vmax=1.0, 
                            linewidths=0.5, 
                            linecolor='black')
        heatmap = heatmap.get_figure()

        buf = io.BytesIO()
        heatmap.savefig(buf, format='jpg')
        buf.seek(0)

        heatmap_image = Image.open(buf).convert("RGB")

        # clean_memory(heatmap)
        return heatmap_image

    def log_heatmap(self, key, cos):
        # log cosine sim heatmap
        heatmap = cos.compute_heatmap().cpu()
        heatmap_image = self.get_heat(heatmap)

        if self.logger is not None:
            self.logger.log_image(key=key, images=[heatmap_image], mode=['RGB'])

        # clean_memory(heatmap, heatmap_image)

    def on_train_epoch_end(self) -> None:
        self.log_heatmap("train/ecg_text", self.train_cos_ecg_txt)
        self.log_heatmap("train/ecg_mask", self.train_cos_ecg_mask)
        self.log_heatmap("train/ecg_noise", self.train_cos_ecg_noise)

    def validation_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int, 
            dataloader_idx: int = 0
        ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_size = len(batch)
        loss_dino, loss_contrastive, student_embeddings, teacher_embeddings, ecg_embeddings, text_embeddings, ecgs, mask_ecgs, texts, paths = self.model_step(batch, dataloader_idx)

        loss = self.hparams.alpha * loss_dino + self.hparams.beta * loss_contrastive

        # update and log metrics
        self.val_loss(loss)
        self.val_cos_ecg_txt(ecg_embeddings, text_embeddings)

        if dataloader_idx == 0:
            self.val_cos_ecg_mask(student_embeddings, teacher_embeddings)
            self.val_kl_ecg_mask(F.softmax(student_embeddings, dim=-1), 
                                F.softmax(teacher_embeddings, dim=-1))
        else:
            self.val_cos_ecg_noise(student_embeddings, teacher_embeddings)
            self.val_kl_ecg_noise(F.softmax(student_embeddings, dim=-1), 
                                F.softmax(teacher_embeddings, dim=-1))

        if dataloader_idx == 0:
            self.log("val/loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/cos_ecg_txt", self.val_cos_ecg_txt, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

            self.log("val/loss_total_mask", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/loss_dino_mask", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/cos_ecg_mask", self.val_cos_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/kl_ecg_mask", self.val_kl_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        else:
            self.log("val/loss_total_noise", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/loss_dino_noise", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/cos_ecg_noise", self.val_cos_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("val/kl_ecg_noise", self.val_kl_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

        return ecgs, mask_ecgs, texts, paths

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        cos_ecg_txt = self.val_cos_ecg_txt.compute()
        self.val_best_cos_ecg_txt(cos_ecg_txt)

        cos_ecg_mask = self.val_cos_ecg_mask.compute()
        self.val_best_cos_ecg_mask(cos_ecg_mask)

        kl_ecg_mask = self.val_kl_ecg_mask.compute()
        self.val_best_kl_ecg_mask(kl_ecg_mask)

        cos_ecg_noise = self.val_cos_ecg_noise.compute()
        self.val_best_cos_ecg_noise(cos_ecg_noise)

        kl_ecg_noise = self.val_kl_ecg_noise.compute()
        self.val_best_kl_ecg_noise(kl_ecg_noise)

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/cos_ecg_text_best", self.val_best_cos_ecg_txt.compute(), sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        self.log("val/cos_ecg_mask_best", self.val_best_cos_ecg_mask.compute(), sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        self.log("val/kl_ecg_mask_best", self.val_best_kl_ecg_mask.compute(), sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        self.log("val/cos_ecg_noise_best", self.val_best_cos_ecg_noise.compute(), sync_dist=True, prog_bar=True, add_dataloader_idx=False)
        self.log("val/kl_ecg_noise_best", self.val_best_kl_ecg_noise.compute(), sync_dist=True, prog_bar=True, add_dataloader_idx=False)

        self.log_heatmap("val/ecg_text", self.val_cos_ecg_txt)
        self.log_heatmap("val/ecg_mask", self.val_cos_ecg_mask)
        self.log_heatmap("val/ecg_noise", self.val_cos_ecg_noise)

    def test_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int, 
            dataloader_idx: int = 0
        ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_size = len(batch)
        loss_dino, loss_contrastive, student_embeddings, teacher_embeddings, ecg_embeddings, text_embeddings, ecgs, mask_ecgs, texts, paths = self.model_step(batch, dataloader_idx)

        loss = self.hparams.alpha * loss_dino + self.hparams.beta * loss_contrastive

        # update and log metrics
        self.test_loss(loss)
        self.test_cos_ecg_txt(ecg_embeddings, text_embeddings)

        if dataloader_idx == 0:
            self.test_cos_ecg_mask(student_embeddings, teacher_embeddings)
            self.test_kl_ecg_mask(F.softmax(student_embeddings, dim=-1), 
                                F.softmax(teacher_embeddings, dim=-1))
        else:
            self.test_cos_ecg_noise(student_embeddings, teacher_embeddings)
            self.test_kl_ecg_noise(F.softmax(student_embeddings, dim=-1), 
                                F.softmax(teacher_embeddings, dim=-1))

        if dataloader_idx == 0:
            self.log("test/loss_contrastive", loss_contrastive, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/cos_ecg_txt", self.test_cos_ecg_txt, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

            self.log("test/loss_total_mask", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/loss_dino_mask", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/cos_ecg_mask", self.test_cos_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/kl_ecg_mask", self.test_kl_ecg_mask, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
        else:
            self.log("test/loss_total_noise", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/loss_dino_noise", loss_dino, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/cos_ecg_noise", self.test_cos_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
            self.log("test/kl_ecg_noise", self.test_kl_ecg_noise, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

        return ecgs, mask_ecgs, texts, paths

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_heatmap("test/ecg_text", self.test_cos_ecg_txt)
        self.log_heatmap("test/ecg_mask", self.test_cos_ecg_mask)
        self.log_heatmap("test/ecg_noise", self.test_cos_ecg_noise)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.student_ecg_encoder = torch.compile(self.student_ecg_encoder)
            self.teacher_mask_ecg_encoder = torch.compile(self.teacher_mask_ecg_encoder)
            self.teacher_noise_ecg_encoder = torch.compile(self.teacher_noise_ecg_encoder)
            self.text_encoder = torch.compile(self.text_encoder)

            self.student_clip_head = torch.compile(self.student_clip_head)
            self.student_mask_dino_head = torch.compile(self.student_mask_dino_head)
            self.teacher_mask_dino_head = torch.compile(self.teacher_mask_dino_head)
            self.student_noise_dino_head = torch.compile(self.student_noise_dino_head)
            self.teacher_noise_dino_head = torch.compile(self.teacher_noise_dino_head)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        model_param = [param for param in self.parameters() if param.requires_grad]

        optimizer = self.hparams.optimizer(params=model_param)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint):
        save_path = self.trainer.checkpoint_callback.dirpath
        save_path = save_path if save_path else self.trainer.default_root_dir
        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, "TolerantECG_encoder.pth")
        encoder_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if "student_ecg_encoder" in k:
                encoder_state_dict[k] = v
        torch.save(encoder_state_dict, save_path)