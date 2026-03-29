import re
import os
import json
import wfdb
import tqdm
import pandas as pd

import numpy as np
from tqdm import tqdm

import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.text.cleaners import ecg_text_cleaner, english_cleaner
from src.utils.ecg_utils.tensor import clean_memory
from src.utils.ecg_utils.ecg_process import mean_ecg, resample_signal
from src.utils.ecg_utils.noise_transforms import highpass_filter, lowpass_filter


class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir='data/mimic-iv-ecg', 
                 meta_dir='data/mimic-iv-ecg-ext-id',
                 chroma_dir='data/chroma_db',

                 is_rag: bool = True,
                 subset_percent=1.0,
                 denoising = True,
                 sample_rate = 500,
                 seed=None,
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.is_rag = is_rag
        self.denoising = denoising
        print("Denoising:", denoising)
        self.sample_rate = sample_rate

        self.seperator = ";"

        if is_rag:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                            model_kwargs={"device": "cpu"},
                                            encode_kwargs={'normalize_embeddings': True})

            # Read data from the JSON file
            with open('data/litfl.json', 'r', encoding='utf-8') as json_file:
                litfl_data = json.load(json_file)

            if not os.path.exists(chroma_dir):
                print("Createing ChromaDB")
                self.documents = [Document(page_content=key, metadata={"source": i}) 
                                    for i, key in enumerate(litfl_data.keys())]
                self.documents = Chroma.from_documents(self.documents, 
                                                    embeddings, 
                                                    persist_directory=chroma_dir, 
                                                    collection_name='LITFL')
            else:
                print("Using predefined ChromaDB")
                self.documents = Chroma(
                    collection_name="LITFL",
                    embedding_function=embeddings,
                    persist_directory=chroma_dir,
                )
            self.ecg_features = list(litfl_data.values())

        delimeters = [' with ', ' and ', " or ", ", ", " - "]
        self.split_pattern = "|".join(map(re.escape, delimeters))

        processed_data_path = os.path.join(data_dir, "processed_data.csv")
        if os.path.exists(processed_data_path):
            self.processed_data = pd.read_csv(processed_data_path, index_col=0)
            clean_memory(processed_data_path)

            # get subset
            if subset_percent < 1.0:
                self.processed_data = self.processed_data.sample(frac=subset_percent, random_state=seed)

            return

        record_list = pd.read_csv(os.path.join(data_dir, "record_list.csv"))
        measurements = pd.read_csv(os.path.join(data_dir, "machine_measurements.csv"))
        measurements = measurements[measurements['report_3'] != 'Analysis error']

        metadata = pd.read_csv(os.path.join(meta_dir, "records_w_diag_icd10.csv"), index_col=0)
        metadata = metadata[~metadata['age'].isna()].drop("file_name", axis=1) # drop NA values & file_name column

        dataset = pd.merge(record_list, measurements, how='inner', on=['subject_id', 'study_id', 'ecg_time'])
        dataset = pd.merge(dataset, metadata, how='inner', on=['subject_id', 'study_id', 'ecg_time'])
        
        clean_memory(record_list, measurements, metadata)

        print("Processing data...")
        # remove all files that has nan in value

        self.processed_data = []
        for i in tqdm(range(len(dataset))):
            file_path = dataset.iloc[i]['path']

            # if it is already deleted, ignore
            ecg_path = os.path.join(self.data_dir, file_path)
            if not os.path.exists(ecg_path + ".hea"):
                continue

            ecg_signal = wfdb.rdrecord(ecg_path).p_signal # (5000, 12)

            # remove error ECGs
            if np.isnan(ecg_signal).any() or np.isinf(ecg_signal).any():
                continue
            if (ecg_signal == 0.0).all():
                continue

            value_range = np.max(ecg_signal, axis=0) - np.min(ecg_signal, axis=0)
            if (value_range > 10.0).any():
                continue

            self.processed_data.append(dataset.iloc[i])

        self.processed_data = pd.DataFrame(self.processed_data)
        self.processed_data.to_csv(processed_data_path)

        if subset_percent < 1.0:
            self.processed_data = self.processed_data.sample(frac=subset_percent, random_state=seed)

        clean_memory(processed_data_path, dataset)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx]
        # get ecg signal information
        file_path = row['path']
        
        ecg_signal = wfdb.rdrecord(
            os.path.join(self.data_dir, file_path)
            )
        ecg_signal = ecg_signal.p_signal    # (5000, 12)
        # Swap avL and avF
        # I-III, avR, avF, avL, V1-6 -> I-III, avR, avL, avF, V1-6
        ecg_signal[:, [4, 5]] = ecg_signal[:, [5, 4]]

        text_file = file_path + ".txt"
        text_file = text_file.split("/")
        text_file[0] = "text_rag" if self.is_rag else "text"
        text_file = "/".join(text_file)
        text_file = os.path.join(self.data_dir, text_file)

        if os.path.exists(text_file):
            with open(text_file, "r") as f:
                ecg_description = f.read().strip()
        else:
            # get report information
            subject_id = row['subject_id'].item()
            study_id = row['study_id'].item()
            
            gender = row['gender']
            if gender == 'F':
                gender = 'Female'
            if gender == 'M':
                gender = 'Male'
            age = int(row['age'].item())

            ecg_description = f"This person is a {gender} of age {age}. "
            ecg_description += "He " if gender == 'Male' else "She "
            ecg_description += f"is having "

            for i in range(0, 18): # there are 17 reports (0 -> 17)
                report = row[f'report_{i}']
                if pd.isna(report): # remove nan values
                    continue

                if "Warning" in report:
                    continue
                
                if "Age not entered" in report:
                    continue

                if "**" in report or "--" in report:
                    continue

                if "Summary" in report:
                    continue

                if report.isupper():
                    report = report.capitalize()

                if report.endswith("."):
                    report = report[:-1]    # remove . punctuation

                for rep in re.split(self.split_pattern, report):
                    rep = rep.strip()
                    rep = ecg_text_cleaner(rep)

                    ecg_description += rep

                    if self.is_rag:
                        rag_result, score = self.documents._similarity_search_with_relevance_scores(rep, k=1)[0]
                        if score > 0.7:
                            idx = rag_result.metadata['source']

                            description = self.ecg_features[idx]
                            if description != "":
                                ecg_description += f": {description}"

                    ecg_description += f"{self.seperator} "

            ecg_description = ecg_description[:-2] + "."  # remove ending "; " and add end punc "."
            ecg_description = english_cleaner(ecg_description)

            clean_memory(subject_id, study_id, gender, age)

            os.makedirs(os.path.dirname(text_file), exist_ok=True)
            with open(text_file, "w") as f:
                f.write(ecg_description)

        ecg_signal = ecg_signal.T       # 12, 5000
        if self.sample_rate != 500:
            ecg_signal = resample_signal(ecg_signal, 500, self.sample_rate)
        if self.denoising:
            ecg_signal = highpass_filter(ecg_signal)    # remove baseline wander
            ecg_signal = lowpass_filter(ecg_signal)     # remove powerline interference

        ecg_signal = torch.from_numpy(ecg_signal.copy()).to(torch.float32)
        # ecg_signal = mean_ecg(ecg_signal)  # [12, 5000]

        return ecg_signal, ecg_description, file_path

    def list_to_string(self, lst):
        # Check if the list is empty
        if not lst:
            return ""
        
        # pattern = r'[^a-zA-Z0-9\s]'
        # lst = [re.sub(pattern, '', s) for s in lst]
        
        # Check if the list has only one element
        if len(lst) == 1:
            return lst[0]
        
        # Join all elements except the last one with a comma and a space
        all_but_last = ", ".join(lst[:-1])
        
        # Append the last element with the word "and" before it
        result = f"{all_but_last} and {lst[-1]}."
        
        return result
    
if __name__ == "__main__":
    _ = MimicDataset()
