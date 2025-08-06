from typing import Any, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.mimic import MimicDataset
from src.data.components.mimic_dino_mask import MimicDinoMaskDataset
from src.data.components.mimic_dino_noise import MimicDinoNoiseDataset
from src.utils.ecg_utils.tokenizer import HuggingFaceTokenizer


num_global = None
num_local = None

class MimicDinoMaskNoiseDataModule(LightningDataModule):
    def __init__(
        self,

        data_dir='data/mimic-iv-ecg', 
        meta_dir='data/mimic-iv-ecg-ext-icd',
        chroma_dir='data/chroma_db',
        model_name="michiyasunaga/BioLinkBERT-base",
        is_rag= True,
        denoising=False,
        subset_percent: float = 1.0,
        sample_rate: int = 500,
        seed=42,

        global_mask_scale=(6, 12), 
        local_mask_scale=(1, 6),

        snr_db=(-10, 0),
        noise_probs=(0.7, 0.7, 0.7),
        noise_rate=0.25,

        global_num=2,
        local_num=8, 

        train_val_test_seed = (None, 7, 123),

        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_mask_train: Optional[Dataset] = None
        self.data_mask_val: Optional[Dataset] = None
        self.data_mask_test: Optional[Dataset] = None

        self.data_noise_train: Optional[Dataset] = None
        self.data_noise_val: Optional[Dataset] = None
        self.data_noise_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        global num_global, num_local
        num_global = global_num
        num_local = local_num

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_mask_train and not self.data_mask_val and not self.data_mask_test \
        and not self.data_noise_train and not self.data_noise_val and not self.data_noise_test:
            dataset = MimicDataset(data_dir=self.hparams.data_dir, 
                                    meta_dir=self.hparams.meta_dir,
                                    chroma_dir=self.hparams.chroma_dir,
                                    is_rag=self.hparams.is_rag,
                                    subset_percent=self.hparams.subset_percent,
                                    denoising=self.hparams.denoising,
                                    sample_rate=self.hparams.sample_rate
                 )
            print("Data length:", len(dataset))
            sample_ecg = dataset[0][0]
            print("ECG shape:", sample_ecg.shape)
            
            train_len = int(self.hparams.train_val_test_split[0] * len(dataset))
            test_len = int(self.hparams.train_val_test_split[1] * len(dataset))
            val_len = len(dataset) - train_len - test_len
            
            data_train, data_val, data_test = random_split(
                dataset=dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )

            self.data_mask_train = MimicDinoMaskDataset(
                                            data_train, 
                                            global_scale=self.hparams.global_mask_scale, 
                                            local_scale=self.hparams.local_mask_scale,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[0])
            self.data_mask_val = MimicDinoMaskDataset(
                                            data_val, 
                                            global_scale=self.hparams.global_mask_scale, 
                                            local_scale=self.hparams.local_mask_scale,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[1])
            self.data_mask_test = MimicDinoMaskDataset(
                                            data_test, 
                                            global_scale=self.hparams.global_mask_scale, 
                                            local_scale=self.hparams.local_mask_scale,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[2])
            
            self.data_noise_train = MimicDinoNoiseDataset(
                                            data_train, 
                                            snr_db=self.hparams.snr_db,
                                            noise_probs=self.hparams.noise_probs,
                                            noise_rate=self.hparams.noise_rate,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[0])
            self.data_noise_val = MimicDinoNoiseDataset(
                                            data_val, 
                                            snr_db=self.hparams.snr_db,
                                            noise_probs=self.hparams.noise_probs,
                                            noise_rate=self.hparams.noise_rate,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[1])
            self.data_noise_test = MimicDinoNoiseDataset(
                                            data_test, 
                                            snr_db=self.hparams.snr_db,
                                            noise_probs=self.hparams.noise_probs,
                                            noise_rate=self.hparams.noise_rate,
                                            global_num=self.hparams.global_num,
                                            local_num=self.hparams.local_num, 
                                            seed=self.hparams.train_val_test_seed[2])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return [
            DataLoader(
                dataset=self.data_mask_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=True,
            ),
            DataLoader(
                dataset=self.data_noise_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=True,
            ),
        ]

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return [
            DataLoader(
                dataset=self.data_mask_val,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=False,
            ),
            DataLoader(
                dataset=self.data_noise_val,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=False,
            ),
        ]

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return [
            DataLoader(
                dataset=self.data_mask_test,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=False,
            ),
            DataLoader(
                dataset=self.data_noise_test,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=MaskNoiseECGBatchCollage(self.hparams.model_name),
                shuffle=False,
            ),
        ]


class MaskNoiseECGBatchCollage:
    def __init__(self, model_name) -> None:
        self.text_tokenizer = HuggingFaceTokenizer(model_name)

    def __call__(self, batch):
        augment_ecgs, ecgs, texts, paths = [], [], [], []

        for i, item in enumerate(batch):
            augment_ecg, ecg, text, path = item
            augment_ecgs.append(augment_ecg)
            ecgs.append(ecg)
            texts.append(text)
            paths.append(path)

        input_ids = self.text_tokenizer.encode(texts)
        ecgs = torch.stack(ecgs)

        augment_ecgs = list(zip(*augment_ecgs))
        augment_ecgs = [torch.stack(tensors, dim=0) for tensors in augment_ecgs]
        augment_ecgs = torch.stack(augment_ecgs, dim=0)

        input_ids['ecgs'] = ecgs
        input_ids['augment_ecgs'] = augment_ecgs

        return input_ids, paths
