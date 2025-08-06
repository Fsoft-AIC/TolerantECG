from typing import Union

import wfdb
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from src.data.components.ptb_xl import PtbXlDataset
from src.utils.ecg_utils.ecg_process import mask_ecg
from src.utils.ecg_utils.ecg_process import resample_signal
from src.utils.ecg_utils.noise_transforms import crop_and_mask_noise, add_combined_noise


class PtbXlMaskNoiseDataset(Dataset):
    def __init__(
            self, 
            dataset: PtbXlDataset,

            mask_rate: float        = 0.5,
            keep_num_lead: int      = None,

            noise_probs: float      = [0.7, 0.7, 0.7],
            noise_rate: float       = 0.5,
            snr_db: Union[float, tuple]   = (-6, 0),

            seed: int               = None    
        ):
        super().__init__()
        self.dataset = dataset
        self.sample_rate = dataset.sample_rate

        if mask_rate is not None:
            self.mask_func = mask_ecg
            self.p = mask_rate
        else:
            self.mask_func = random_lead_keep
            self.p = keep_num_lead

        baseline_wander, _ = wfdb.rdsamp("data/mit-noise/bw")
        muscle_artifact, _ = wfdb.rdsamp("data/mit-noise/ma")
        electrode_motion, _ = wfdb.rdsamp("data/mit-noise/em")
        
        baseline_wander = resample_signal(baseline_wander.T, new_sr=self.sample_rate)    # 2, N
        muscle_artifact = resample_signal(muscle_artifact.T, new_sr=self.sample_rate)    # 2, N
        electrode_motion = resample_signal(electrode_motion.T, new_sr=self.sample_rate)  # 2, N

        self.noise_list = [(baseline_wander, noise_probs[0]),
                           (muscle_artifact, noise_probs[1]),
                           (electrode_motion, noise_probs[2])]
        self.noise_rate = noise_rate
        self.snr_db = snr_db

        if seed is not None:
            np.random.seed(seed)
            self.seeds = np.random.randint(0, len(dataset), size=len(dataset)).tolist()
        else:
            self.seeds = [None] * len(dataset)

    @property
    def num_classes(self) -> int:
        return len(self.dataset.num_classes)

    def compose_transform(self, signal, seed=None):
        """
        noise_list: a tuple (noise, probability)
        """
        np.random.seed(seed)

        length = int(self.sample_rate * 10)
        combined_noise = np.zeros((12, length))
        lead_indicator = np.ones(12).astype(bool)

        rand_seeds = np.random.randint(len(self.dataset), size=len(self.noise_list)).tolist()
        for (noise, p), rand_seed in zip(self.noise_list, rand_seeds):
            random.seed(rand_seed)
            if random.random() < p:
                noise, clean_lead = crop_and_mask_noise(noise, p=self.noise_rate, seed=rand_seed, target_length=length)

                combined_noise += noise
                lead_indicator = np.bitwise_and(lead_indicator, clean_lead)

        return add_combined_noise(signal, combined_noise, self.snr_db), \
                torch.from_numpy(lead_indicator)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        signal, _, label, path = self.dataset[idx]
        current_seed = self.seeds[idx]

        signal, lead_indicator_noise = self.compose_transform(signal, seed=current_seed)
        signal, lead_indicator_mask = self.mask_func(signal, p=self.p, seed=current_seed)
        return signal, \
                torch.bitwise_and(lead_indicator_noise.to(bool), lead_indicator_mask.to(bool)).to(torch.float32), \
                label, \
                path 