import warnings

import torch
from torchmetrics import Metric
from sklearn.metrics import precision_recall_curve
import numpy as np


warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="torchmetrics")

class MultilabelF1Max(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Add state variables
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_probs", default=[], dist_reduce_fx="cat")

    def update(self, y_probs: torch.Tensor, y_true: torch.Tensor):
        """
        Update state with new batch of data.
        
        Args:
            y_true: Ground truth binary labels
            y_probs: Predicted probabilities
        """
        self.y_true.append(y_true)
        self.y_probs.append(y_probs.to(torch.float32))
    
    def compute(self):
        """
        Compute the maximum F1 score across thresholds.
        
        Returns:
            max_f1: Maximum F1 score
            best_threshold: Threshold corresponding to max F1
        """
        # Concatenate all batches
        if isinstance(self.y_true, list):
            y_true = torch.cat(self.y_true).detach().cpu().numpy()
        else:
            y_true = self.y_true.detach().detach().cpu().numpy()
        
        if isinstance(self.y_probs, list):
            y_probs = torch.cat(self.y_probs).detach().cpu().numpy()
        else:
            y_probs = self.y_probs.detach().cpu().numpy()

        # Calculate precision, recall, and thresholds
        # calculate f1 for all classes
        max_f1s = []
        for i in range(y_true.shape[1]):
            gt_np = y_true[:, i]
            pred_np = y_probs[:, i]
            precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1 = np.max(f1_scores)
            max_f1s.append(max_f1)

        return torch.tensor(max_f1s, dtype=torch.float32).mean()

if __name__ == "__main__":
    # Initialize the metric
    max_f1_metric = MultilabelF1Max()

    # Simulate a batch update
    y_true_batch = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    y_probs_batch = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5], dtype=torch.float32)
    max_f1_metric.update(y_probs_batch.unsqueeze(0), y_true_batch.unsqueeze(0))

    # Compute the result
    max_f1 = max_f1_metric.compute()
    print(f"Maximum F1 Score: {max_f1:.4f}")