import os
import ast
import json
from glob import glob
import os
import wfdb
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.ecg_utils.ecg_process import mean_ecg, resample_signal
from src.utils.ecg_utils.noise_transforms import highpass_filter, lowpass_filter


class MitBihDataset(Dataset):
    def __init__(self,
                 data_dir: str      = "data/mit-bih",
                 data_type: str     = "train",
                 multilabel: bool   = True,
                 sample_rate: str   = 500,
                 denoising: bool    = False
                 ):
        super().__init__()
        assert data_type in ["train", "val", "test"]

        self.classes = ['N', "L", "R", "A", "V"]
        self.label_to_index = {c : i for i, c in enumerate(self.classes)}
        self.multilabel = multilabel

        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.data_type = data_type

        self.denoising = denoising
        print("Denoising:", denoising)

        folder = "multilabel" if multilabel else "multiclass"
        processed_dir = os.path.join(data_dir, "processed", folder)
        os.makedirs(processed_dir, exist_ok=True)

        if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
            print("Processing data")
            record_ids = glob(os.path.join(data_dir, "*.hea"))
            record_ids = [record_id[:-4] for record_id in record_ids]

            X, y = self.build_multilabel_dataset(record_ids)

            from skmultilearn.model_selection import iterative_train_test_split
            # Ratio 7, 1, 2
            X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.3)
            X_val, y_val, X_test, y_test = iterative_train_test_split(X_temp, y_temp, test_size=0.2 / (0.1 + 0.2))

            print("Saving processed data...")
            np.save(os.path.join(processed_dir, "train_data.npy"), X_train)
            np.save(os.path.join(processed_dir, "train_label.npy"), y_train)
            np.save(os.path.join(processed_dir, "val_data.npy"), X_val)
            np.save(os.path.join(processed_dir, "val_label.npy"), y_val)
            np.save(os.path.join(processed_dir, "test_data.npy"), X_test)
            np.save(os.path.join(processed_dir, "test_label.npy"), y_test)

        self.data = np.load(os.path.join(processed_dir, f"{data_type}_data.npy"))
        self.label = np.load(os.path.join(processed_dir, f"{data_type}_label.npy"))

    def extract_10s_segments_multilabel(self, record_id):
        try:
            record = wfdb.rdrecord(record_id)
        except:
            record = wfdb.rdrecord(os.path.basename(record_id), pn_dir='mitdb')

        try:
            ann = wfdb.rdann(record_id, 'atr')
        except:
            ann = wfdb.rdann(os.path.basename(record_id), 'atr', pn_dir='mitdb')
    
        signal = record.p_signal
        sig_name = record.sig_name

        signal_12 = np.zeros((len(signal), 12))

        for i, lead in enumerate(sig_name):
            if lead == "MLII":
                signal_12[:, 1] = signal[:, i]
            else:
                v_lead = int(lead[-1])
                signal_12[:, v_lead + 5] = signal[:, i]

        r_peaks = ann.sample
        labels = ann.symbol
    
        segments, segment_labels = [], []
        segment_len = int(record.fs * 10)

        if not self.multilabel:
            from collections import Counter
            for start in range(0, len(signal_12) - segment_len, segment_len):
                end = start + segment_len
                seg = signal_12[start:end]
        
                # Find beat labels in this segment
                beats_in_segment = [
                    labels[i] for i, r in enumerate(r_peaks)
                    if r >= start and r < end and labels[i] in self.label_to_index
                ]

                if not beats_in_segment:
                    continue

                # Use majority label (you can refine this)
                most_common = Counter(beats_in_segment).most_common(1)[0][0]
                segments.append(seg.T)
                segment_labels.append(self.label_to_index[most_common])
        
            return segments, segment_labels
        else:
            for start in range(0, len(signal_12) - segment_len, segment_len):
                end = start + segment_len
                segment = signal_12[start : end]    # 5000 x 12
        
                # Find all beats in this segment that match the 5 classes
                beats_in_segment = [
                    labels[i] for i, r in enumerate(r_peaks)
                    if start <= r < end and labels[i] in self.label_to_index
                ]
        
                if not beats_in_segment:
                    continue

                # Create a binary vector for this segment
                label_vec = [0] * len(self.classes)
                for beat in beats_in_segment:
                    label_vec[self.label_to_index[beat]] = 1
        
                segments.append(segment.T)      # 12 x 5000
                segment_labels.append(label_vec)
        
            return segments, segment_labels

    def build_multilabel_dataset(self, record_ids):
        X, y = [], []
        for rec_id in tqdm(record_ids):
            segments, labels = self.extract_10s_segments_multilabel(rec_id)
            X.extend(segments)
            y.extend(labels)
        return np.array(X), np.array(y)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        signal = resample_signal(signal, 360, self.sample_rate)

        binary_label = self.label[idx]
        
        lead_indicator = np.zeros(12, dtype=np.float32)
        lead_indicator[np.where(np.any(signal != 0, axis=1))[0]] = 1

        if self.denoising:
            signal = highpass_filter(signal)    # remove baseline wander
            signal = lowpass_filter(signal)     # remove powerline interference
        signal = torch.from_numpy(signal.copy()).to(torch.float32)
        # signal = mean_ecg(signal)

        binary_label = torch.IntTensor(binary_label)

        return signal, torch.from_numpy(lead_indicator), binary_label, f"{self.data_type}_{idx}"
