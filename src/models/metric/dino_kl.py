import torch
import torch.nn.functional as F

from torchmetrics import Metric

class DinoKL(Metric):
    def __init__(self, n_local_crops, n_global_crops):
        super().__init__()
        self.n_crops = n_local_crops + n_global_crops  # total number of crops = num global crops + local_crops_number
        self.n_global_crops = n_global_crops    # = 2 in paper

        self.add_state("kl", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("len", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, student_out, teacher_out):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_out.chunk(self.n_crops)
        teacher_out = teacher_out.chunk(self.n_global_crops)

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                kl_div = torch.sum(q * torch.log(q / student_out[v]), dim=-1)
                self.kl += kl_div.mean()
                self.len += 1

    def compute(self):
        return self.kl / self.len

if __name__ == "__main__":
    s_emb = torch.randn((8, 768))
    t_emb = torch.randn((4, 768))

    KLDiv = DinoKL(1, 1)

    KLDiv(F.softmax(s_emb, dim=-1), F.softmax(t_emb, dim=-1))

    print(KLDiv.compute())
