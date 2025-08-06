import numpy as np
from torch.utils.data import Dataset

from src.data.components.ptb_xl import PtbXlDataset
from src.utils.ecg_utils.ecg_process import mask_ecg, random_lead_keep, keep_lead


class PtbXlMaskDataset(Dataset):
    def __init__(
            self, 
            dataset: PtbXlDataset,
            mask_rate: float        = 0.5,
            keep_num_lead: int      = None,
            keep_lead_idx: list     = None,
            seed: int               = None    
        ):
        super().__init__()
        assert not (mask_rate is None and keep_num_lead is None and keep_lead_idx is None)
        """
        if mask_rate is None, keep_num_lead is used
        if mask_rate is not None, use mask_rate
        """

        self.dataset = dataset
        # self.mask_rate = mask_rate
        # self.keep_num_lead = keep_num_lead

        if hasattr(dataset, "text_prompts"):
            self.text_prompts = dataset.text_prompts

        if mask_rate is not None:
            self.mask_func = mask_ecg
            self.p = mask_rate
        if keep_num_lead is not None:
            self.mask_func = random_lead_keep
            self.p = keep_num_lead
        if keep_lead_idx is not None:
            self.mask_func = keep_lead
            self.p = keep_lead_idx

        if seed is not None:
            np.random.seed(seed)
            self.seeds = np.random.randint(0, len(dataset), size=len(dataset)).tolist()
        else:
            self.seeds = [None] * len(dataset)

    @property
    def num_classes(self) -> int:
        return len(self.dataset.num_classes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        signal = sample[0]
        label = sample[-2]
        path = sample[-1]

        signal, lead_indicator = self.mask_func(signal, p=self.p, seed=self.seeds[idx])
        return signal, lead_indicator, label, path 