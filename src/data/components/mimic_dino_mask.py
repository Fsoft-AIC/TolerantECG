import random
import numpy as np
from torch.utils.data import Dataset

from src.data.components.mimic import MimicDataset
from src.utils.ecg_utils.ecg_process import random_lead_keep


class DinoMaskAugmentation(object):
    def __init__(self, 
                data_len,
                global_scale    = (6, 12), 
                local_scale     = (1, 6),
                global_num      = 2,
                local_num       = 8,
                ):

        self.data_len = data_len

        self.global_scale = global_scale
        self.global_num = global_num
        self.local_scale = local_scale
        self.local_num = local_num

    def __call__(self, ecg, seed=None):
        augment_len = self.global_num + self.local_num

        random.seed(seed)
        seeds = random.choices(range(0, self.data_len), k=augment_len)
        masked_ecgs = []

        for i in range(self.global_num):
            masked_ecg, _ = random_lead_keep(ecg, self.global_scale, seed=seeds[i])
            masked_ecgs.append(masked_ecg)

        for i in range(self.local_num):
            masked_ecg, _ = random_lead_keep(ecg, self.local_scale, seed=seeds[i + self.global_num])
            masked_ecgs.append(masked_ecg)

        return masked_ecgs


class MimicDinoMaskDataset(Dataset):
    def __init__(
                self, 
                dataset: MimicDataset, 
                global_scale: tuple     = (6, 12), 
                local_scale: tuple      = (1, 6),
                global_num: int         = 2,
                local_num: int          = 8, 
                seed: int               = None
                ):

        self.dataset = dataset
        self.transform = DinoMaskAugmentation(
                            len(dataset),
                            global_scale=global_scale, 
                            local_scale=local_scale,
                            global_num=global_num,
                            local_num=local_num
                        )

        if seed is not None:
            np.random.seed(seed)
            self.seeds = np.random.randint(0, len(dataset), size=len(dataset)).tolist()
        else:
            self.seeds = [None] * len(dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ecg_signal, ecg_description, file_path = self.dataset[idx]

        return self.transform(ecg_signal, seed=self.seeds[idx]), ecg_signal, ecg_description, file_path