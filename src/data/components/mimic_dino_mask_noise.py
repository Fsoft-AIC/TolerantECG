import random
import wfdb
from scipy.signal import resample

import numpy as np
import torch

from src.data.components.mimic import MimicDataset
from src.utils.ecg_utils.ecg_process import resample_signal, random_lead_keep
from src.utils.ecg_utils.noise_transforms import lowpass_filter, \
                                                highpass_filter, \
                                                add_nst, \
                                                crop_and_mask_noise, \
                                                add_combined_noise


class DinoMaskNoiseAugmentation(object):
    def __init__(self, 
                data_len,
                snr_db          = (-10, 0),
                noise_probs     = (0.7, 0.7, 0.7),
                noise_rate      = 0.25,
                global_scale    = (6, 12), 
                local_scale     = (1, 6),
                global_num      = 2,
                local_num       = 8,
                sample_rate     = 500, 
                ):

        self.data_len = data_len

        baseline_wander, _ = wfdb.rdsamp("data/mit-noise/bw")
        muscle_artifact, _ = wfdb.rdsamp("data/mit-noise/ma")
        electrode_motion, _ = wfdb.rdsamp("data/mit-noise/em")
        
        baseline_wander = resample_signal(baseline_wander.T, 360, sample_rate)    # 2, N
        muscle_artifact = resample_signal(muscle_artifact.T, 360, sample_rate)    # 2, N
        electrode_motion = resample_signal(electrode_motion.T, 360, sample_rate)  # 2, N

        self.noise_list = []
        for noise, p in zip([baseline_wander, muscle_artifact, electrode_motion], noise_probs):
            self.noise_list.append([noise, p])

        self.noise_rate = noise_rate
        self.global_num = global_num
        self.local_num = local_num
        self.snr_db = snr_db

        self.global_scale = global_scale
        self.local_scale = local_scale

    def compose_transform(self, signal, seed=None):
        """
        noise_list: a tuple (noise, probability)
        """
        np.random.seed(seed)

        combined_noise = np.zeros_like(signal)
        seq_len = signal.shape[-1]
        rand_seeds = np.random.randint(self.data_len, size=len(self.noise_list)).tolist()
        for (noise, p), rand_seed in zip(self.noise_list, rand_seeds):
            random.seed(rand_seed)
            if random.random() < p:
                combined_noise += crop_and_mask_noise(noise, p=self.noise_rate, target_length=seq_len, seed=rand_seed)[0]

        # if no noise is apply, use a random noise
        if (combined_noise == 0).all():
            idx = np.random.randint(0, len(self.noise_list))
            noise = self.noise_list[idx][0]
            combined_noise = crop_and_mask_noise(noise, p=self.noise_rate, target_length=seq_len, seed=rand_seeds[idx])[0]
        return add_combined_noise(signal, combined_noise, self.snr_db)

    def __call__(self, ecg, seed=None):
        random.seed(seed)
        teacher_ecgs = []
        student_ecgs = []

        augment_len = self.global_num + self.local_num
        seeds = random.choices(range(0, self.data_len), k=augment_len)

        # add 2 clean ecgs for teachers
        clean_ecg = lowpass_filter(highpass_filter(ecg))
        mask_clean_ecg, _ = random_lead_keep(clean_ecg, self.global_scale, seed=seeds[0])
        teacher_ecgs.append(mask_clean_ecg)
        if self.global_num == 2:
            mask_ecg, _ = random_lead_keep(ecg, self.global_scale, seed=seeds[1])
            teacher_ecgs.append(mask_ecg)

        # if number of local samples are less than number of listed noises
        # just add combined noise
        # else save slot for uni-noise
        if self.local_num < len(self.noise_list):
            for i in range(self.local_num):
                noise_ecg = self.compose_transform(clean_ecg, seed=seeds[i + self.global_num])
                noise_ecg, _ = random_lead_keep(noise_ecg, self.local_scale, seed=seeds[i + self.global_num])
                student_ecgs.append(noise_ecg)
        else:
            # combined noises for student
            for i in range(self.local_num - len(self.noise_list)):
                noise_ecg = self.compose_transform(clean_ecg, seed=seeds[i + self.global_num])
                noise_ecg, _ = random_lead_keep(noise_ecg, self.local_scale, seed=seeds[i + self.global_num])
                student_ecgs.append(noise_ecg)

            combined_noise_num = len(student_ecgs)
            for i, (noise, _) in enumerate(self.noise_list):
                noise_ecg = add_nst(clean_ecg, noise, snr_db=self.snr_db, p=self.noise_rate, seed=seeds[i + combined_noise_num + self.global_num])
                noise_ecg, _ = random_lead_keep(noise_ecg, self.local_scale, seed=seeds[i + self.global_num + combined_noise_num])
                student_ecgs.append(noise_ecg)

        return teacher_ecgs + student_ecgs


class MimicDinoMaskNoiseDataset(torch.utils.data.Dataset):
    def __init__(
                self, 
                dataset: MimicDataset, 
                snr_db: tuple           = (-10, 0),
                noise_probs: tuple      = (0.7, 0.7, 0.7),
                noise_rate: float       = 0.25,
                global_scale            = (6, 12), 
                local_scale             = (1, 6),
                global_num: int         = 2,
                local_num: int          = 8,
                seed: int               = None
                ):
        self.dataset = dataset
        
        try:
            sample_rate = dataset.sample_rate
        except:
            sample_rate = dataset.dataset.sample_rate

        self.transform = DinoMaskNoiseAugmentation(
                            len(dataset),
                            snr_db=snr_db,
                            noise_probs=noise_probs,
                            noise_rate=noise_rate,
                            global_scale=global_scale, 
                            local_scale=local_scale,
                            global_num=global_num,
                            local_num=local_num,
                            sample_rate=sample_rate
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