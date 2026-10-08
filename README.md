# [ACMMM 2025] TolerantECG: A Foundation Model for Imperfect Electrocardiogram

This repository contains the source code of paper **[TolerantECG: A Foundation Model for Imperfect Electrocardiogram](https://arxiv.org/abs/2507.09887)**
![model architecture](figures/tolerantECG.png)

## Data preprocessing
The data used in this work can be downloaded using the links below:
- MIMIC-IV-IV: https://physionet.org/content/mimic-iv-ecg/1.0/
- MIMIC-IV-ECG-Ext-ICD: https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/
- MIT-BIH: https://www.physionet.org/content/mitdb/1.0.0/
- MIT-BIH Noise Stress Test: https://www.physionet.org/content/nstdb/1.0.0/
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/

After downloading, please put them in the `data` folder as such:

```
data
└───mimic-iv-ecg
|   |   machine_measurements.csv
│   |   record_list.csv
│   └───files
│       ...
└───mimic-iv-ecg-ext-id
│   |   records_w_diag_icd10.csv
└───mit-bih
|   │   100.atr
|   │   100.dat
│       ...
└───mit-noise
|   │   bw.dat
|   │   em.dat
|   │   ma.dat
│       ...
└───ptb-xl
|   │   pltxl_database.csv
|   │   scp_statements.csv
│   └───records500
│       ...
```

## Installation requirement

The necessary requirements can be installed using pip:
```bash
pip install -r requirements.txt
```

## Create CFR database
```bash
python -m src.init_cfr
```

## Pretrained Checkpoint

The pre-trained weights for the TolerantECG encoder (`TolerantECG_encoder.pth`) are available on Hugging Face:
🤗 **[ndhuynh02/TolerantECG](https://huggingface.co/ndhuynh02/TolerantECG)**

You can download the checkpoint manually or programmatically via Python:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ndhuynh02/TolerantECG", 
    filename="TolerantECG_encoder.pth", 
    local_dir="checkpoints"
)
```

## Training & finetuning
The training scripts are provided in the `script` folder. Or it can be simply run as followed:
- Pretraining with MIMIC-IV-ECG:
```bash
./script/train.sh
```
- Finetune with PTB-XL Super-diagnosis:
```bash
./script/finetune.sh
```
Make sure to set `model.encoder_ckpt_path` in `script/finetune.sh` to the path of your downloaded checkpoint (e.g., `checkpoints/TolerantECG_encoder.pth`).
Please note to change some of the arguments in the `.sh` files for desired modification.

## Citation
```bibtex
@inproceedings{10.1145/3746027.3755287,
    author = {Nguyen, Huynh Dang and Pham, Trong-Thang and Le, Ngan and Nguyen, Van},
    title = {TolerantECG: A Foundation Model for Imperfect Electrocardiogram},
    year = {2025},
    isbn = {9798400720352},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3746027.3755287},
    doi = {10.1145/3746027.3755287},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
    pages = {8097–8105},
    numpages = {9},
    keywords = {contrastive learning, electrocardiogram (ecg), foundation model, imperfect signal, knowledge retrieval, self-supervised learning},
    location = {Dublin, Ireland},
    series = {MM '25}
}
```
