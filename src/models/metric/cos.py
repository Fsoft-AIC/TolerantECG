import torch
from torchmetrics import Metric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class CosineSimilarity(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sim", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("len", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("heatmap", default=[], dist_reduce_fx='cat')

    def update(self, ecg_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> None:
        """
        ecg_embeddings: [B, L]
        description_embeddings: [B, L]
        """
        if ecg_embeddings.shape != text_embeddings.shape:
            raise ValueError("ecg_embeddings and text_embeddings must have the same shape")

        similar = pairwise_cosine_similarity(ecg_embeddings, text_embeddings, zero_diagonal=False)
        # if torch.isnan(similar).any():
        #     raise ValueError(f"Nan in Cos\nECG: {torch.isnan(ecg_embeddings).any(), torch.isinf(ecg_embeddings).any(), (ecg_embeddings == 0.0).all(dim=1)}\nText: {torch.isnan(text_embeddings).any(), torch.isinf(text_embeddings).any(), (text_embeddings == 0.0).all(dim=1)}")
        
        if len(self.heatmap) == 0:
            self.heatmap.append(similar)
        elif self.heatmap[-1].shape == similar.shape:
            self.heatmap.append(similar)

        sim_pair = torch.diag(similar, 0)
        self.sim += sim_pair.sum()
        self.len += ecg_embeddings.size(0)

    def compute(self) -> torch.Tensor:
        return self.sim / self.len

    def compute_heatmap(self) -> torch.Tensor:
        heat = torch.stack(self.heatmap, dim=0)
        return heat.mean(0)
    

if __name__ == "__main__":
    x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
    y = torch.tensor([[1, 0], [2, 1], [3, 2]], dtype=torch.float32)

    metric = CosineSimilarity()
    metric.update(x, y)
    print(metric.compute_heatmap())

    # metric.reset()
    x = torch.tensor([[2, 3], [3, 5], [5, 8], [9, 10]], dtype=torch.float32)
    y = torch.tensor([[1, 0], [2, 1], [3, 2], [0, 2]], dtype=torch.float32)
    metric.update(x, y)
    print(metric.compute_heatmap())