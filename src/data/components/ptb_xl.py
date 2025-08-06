import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


import os
import ast
import json
import wfdb

import pandas as pd

import torch
from torch.utils.data import Dataset

from src.utils.ecg_utils.ecg_process import mean_ecg, resample_signal
from src.utils.ecg_utils.noise_transforms import highpass_filter, lowpass_filter


class PtbXlDataset(Dataset):
    def __init__(self,
                 data_dir: str      = "data/ptb-xl",
                 sample_rate: str   = 500,
                 data_type: str     = "train",
                 level: str         = 'diagnostic_super',
                 denoising: bool    = False):
        super().__init__()
        assert data_type in ["train", "val", "test"]
        assert level in ["all", "diagnostic", "diagnostic_sub", "diagnostic_super", "rhythm", "form"]

        self.data_dir = data_dir
        self.sample_rate = sample_rate

        self.denoising = denoising
        print("Denoising:", denoising)

        # load and convert annotation data
        self.data = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col='ecg_id')
        # 1-8 as training set, fold 9 as validation set and fold 10 as test set.
        if data_type == "train":
            self.data = self.data[self.data.strat_fold.isin(range(1, 9))]
        elif data_type == "val":
            self.data = self.data[self.data.strat_fold == 9]
        else:
            self.data = self.data[self.data.strat_fold == 10]

        self.data.scp_codes = self.data.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        if level != "all":
            if level not in ["diagnostic_sub", "diagnostic_super"]:
                self.agg_df = self.agg_df[self.agg_df[level] == 1]
                self.classes = self.agg_df.index
            else:
                self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
                column = 'diagnostic_subclass' if level == "diagnostic_sub" else 'diagnostic_class'
                self.classes = self.agg_df[column].unique()
        else:
            self.classes = self.agg_df.index

        # global label_to_id, id_to_label
        self.label_to_id, self.id_to_label = {}, {}
        for i, label in enumerate(self.classes):
            self.label_to_id[label] = i
            self.id_to_label[i] = label

        def aggregate_diagnostic(y_dic):
            classes = []
            for key in y_dic.keys():
                if key in self.agg_df.index:
                    if level not in ["diagnostic_sub", "diagnostic_super"]:
                        classes.append(key)
                    else:
                        classes.append(self.agg_df.loc[key][column])
            return list(set(classes))   # remove duplicates

        self.data['label'] = self.data.scp_codes.apply(aggregate_diagnostic)
        # Remove instances without any labels
        # self.data = self.data[self.data['label'].apply(lambda x: len(x) > 0)]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        path = row.filename_lr if self.sample_rate == 100 else row.filename_hr
        signal, meta = wfdb.rdsamp(os.path.join(self.data_dir, path))
        
        signal = signal.T                   # [12, 5000]

        if self.sample_rate not in [100, 500]:
            signal = resample_signal(signal, old_sr=meta['fs'], new_sr=self.sample_rate)    # 2, N

        if self.denoising:
            signal = highpass_filter(signal)    # remove baseline wander
            signal = lowpass_filter(signal)     # remove powerline interference
        signal = torch.from_numpy(signal.copy()).to(torch.float32)
        # signal = mean_ecg(signal)

        binary_label = [0.0] * self.num_classes

        for cls in row.label:
            index = self.label_to_id[cls]
            binary_label[index] = 1.0
        binary_label = torch.IntTensor(binary_label)
        
        return signal, torch.ones(12, dtype=torch.float32), binary_label, path


if __name__ == "__main__":
    data = PtbXlDataset()
    print(data.classes)

    sample = data[0]
    ecg, _, label, _ = sample
    print(label)