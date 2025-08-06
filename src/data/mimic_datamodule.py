from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.mimic import MimicDataset
from src.utils.ecg_utils.tokenizer import HuggingFaceTokenizer


class MimicDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir='data/mimic-iv-ecg', 
        meta_dir='data/mimic-iv-ecg-ext-icd',
        chroma_dir='data/chroma_db',
        is_rag= True,
        denoising=False,
        subset_percent: float = 1.0,
        sample_rate: int = 500,
        seed=42,

        model_name="michiyasunaga/BioLinkBERT-base",

        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size


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
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = MimicDataset(data_dir=self.hparams.data_dir, 
                                    meta_dir=self.hparams.meta_dir,
                                    chroma_dir=self.hparams.chroma_dir,
                                    is_rag=self.hparams.is_rag,
                                    subset_percent=self.hparams.subset_percent,
                                    denoising=self.hparams.denoising,
                                    seed=self.hparams.seed,
                                    sample_rate=self.hparams.sample_rate
                 )
            print("Data length:", len(dataset))
            sample_ecg = dataset[0][0]
            print("ECG shape:", sample_ecg.shape)

            train_len = int(self.hparams.train_val_test_split[0] * len(dataset))
            val_len = int(self.hparams.train_val_test_split[1] * len(dataset))
            test_len = len(dataset) - train_len - val_len

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=[train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42),
            )


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=ECGBatchCollage(self.hparams.model_name),
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=ECGBatchCollage(self.hparams.model_name),
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=ECGBatchCollage(self.hparams.model_name),
            shuffle=False,
        )
    
class ECGBatchCollage:
    def __init__(self, model_name) -> None:
        self.text_tokenizer = HuggingFaceTokenizer(model_name)

    def __call__(self, batch):
        ecg_image, texts, paths = [], [], []

        for i, item in enumerate(batch):
            image, text, path = item
            ecg_image.append(image)
            texts.append(text)
            paths.append(path)

        input_ids = self.text_tokenizer.encode(texts)
        ecg_image = torch.stack(ecg_image)

        input_ids['ecgs'] = ecg_image

        return input_ids, paths
        

if __name__ == "__main__":
    data = MimicDataModule(ecg_mimic_dir="data/mimic-iv-ecg", batch_size=2, num_dim=2)
    data.setup()

    val = data.val_dataloader()
    batch = next(iter(val))
    inputs, paths = batch
    print(inputs['image'].shape)