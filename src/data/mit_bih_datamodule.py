from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.mit_bih import MitBihDataset


class MitBihDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str       = "data/mit-bih",
        multilabel: bool    = True,
        sample_rate: str    = 500,
        denoising: bool     = False,

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
            self.data_train = MitBihDataset(data_type='train',
                                            data_dir=self.hparams.data_dir,
                                            multilabel=self.hparams.multilabel,
                                            sample_rate=self.hparams.sample_rate,
                                            denoising=self.hparams.denoising,
                                            )

            self.data_val = MitBihDataset(data_type="val",
                                            data_dir=self.hparams.data_dir,
                                            multilabel=self.hparams.multilabel,
                                            sample_rate=self.hparams.sample_rate,
                                            denoising=self.hparams.denoising,)

            self.data_test = MitBihDataset(data_type="test",
                                            data_dir=self.hparams.data_dir,
                                            multilabel=self.hparams.multilabel,
                                            sample_rate=self.hparams.sample_rate,
                                            denoising=self.hparams.denoising,)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
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
            shuffle=False,
        )