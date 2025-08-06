import torch
from torchmetrics import Metric
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

class DinoCos(Metric):
    def __init__(self, n_local_crops, n_global_crops):
        super().__init__()
        self.n_crops = n_local_crops + n_global_crops  # total number of crops = num global crops + local_crops_number
        self.n_global_crops = n_global_crops    # = 2 in paper

        self.add_state("sim", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("len", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("heatmap", default=[], dist_reduce_fx='cat')

    def update(self, student_out, teacher_out):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_out.chunk(self.n_crops)

        # teacher centering and sharpening
        teacher_out = teacher_out.chunk(self.n_global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                cosine = pairwise_cosine_similarity(q, student_out[v], zero_diagonal=False)

                if len(self.heatmap) == 0:
                    self.heatmap.append(cosine)
                elif self.heatmap[-1].shape == cosine.shape:
                    self.heatmap.append(cosine)

                cosine = torch.diag(cosine, 0)

                self.sim += cosine.mean()
                self.len += 1
    
    def compute(self):
        return self.sim / self.len

    def compute_heatmap(self) -> torch.Tensor:
        heat = torch.stack(self.heatmap, dim=0)
        return heat.mean(0)

if __name__ == "__main__":
    s_emb = torch.randn((20, 768))
    t_emb = torch.randn((4, 768))

    cos = DinoCos(8, 2)

    cos(s_emb, t_emb)

    print(cos.compute())
    print(cos.compute_heatmap().shape)