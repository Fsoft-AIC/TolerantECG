import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ClipLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, ecg_embeddings, text_embeddings):
        """
        ecg_embeddings: [B, L]
        description_embeddings: [B, L]
        """
        if ecg_embeddings.shape != text_embeddings.shape:
            raise ValueError("ecg_embeddings and text_embeddings must have the same shape")
        # Normalize the embeddings
        ecg_embeddings = F.normalize(ecg_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity scores
        logits = self.logit_scale.exp() * ecg_embeddings @ text_embeddings.t()
 
        # Create labels
        batch_size = ecg_embeddings.size(0)
        labels = torch.arange(batch_size, device=ecg_embeddings.device)

        # Compute cross-entropy loss
        loss_image = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)

        loss_total = loss_image + loss_text

        # Return the average loss
        return loss_total / 2.0
    