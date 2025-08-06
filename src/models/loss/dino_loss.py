import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_local_crops, n_global_crops, teacher_temp, student_temp=0.1, center_momentum=0.9):

        super().__init__()
        # self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = n_local_crops + n_global_crops  # total number of crops = num global crops + local_crops_number
        self.n_global_crops = n_global_crops    # = 2 in paper
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp = teacher_temp

        self.teacher_temp = nn.Parameter(torch.Tensor([teacher_temp]))
        self.student_temp = nn.Parameter(torch.Tensor([student_temp]))

    def forward(self, student_output, teacher_output, is_traininig=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_global_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, p in enumerate(student_out):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(p, dim=-1), dim=-1)

                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        if is_traininig:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MetaDINOLoss(nn.Module):
    def __init__(self, out_dim, n_local_crops, n_global_crops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_crops = n_local_crops + n_global_crops  # total number of crops = num global crops + local_crops_number
        self.n_global_crops = n_global_crops    # = 2 in paper
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, is_training=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.n_global_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        if is_training:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

if __name__ == "__main__":
    s_emb = torch.randn((16, 768))
    t_emb = torch.randn((4, 768))

    loss = DINOLoss(768, 6, 2, 0.04)

    loss(s_emb, t_emb)