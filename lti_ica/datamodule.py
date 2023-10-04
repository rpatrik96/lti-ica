"""A pytorch lightning datamodule that uses the dataset class from `dataset.py`"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lti_ica.dataset import NonstationaryLTIDataset


class NonstationaryLTIDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        num_comp,
        num_data_per_segment,
        num_segment,
        dt,
        triangular=False,
        use_B=True,
        zero_means=True,
        max_variability=False,
        use_C=True,
        system_type="lti",
        ar_order=1,
        batch_size=64,
    ):
        super().__init__()
        self.save_hyperparameters()

        if type(self.hparams.num_segment) == int:
            self.hparams.num_segment = (
                self.hparams.num_segment * self.hparams.num_comp + 1
            )
            print(
                f"num_segment was an int, so it was changed to {self.hparams.num_segment}"
            )

        self.hparams.num_data = (
            self.hparams.num_data_per_segment * self.hparams.num_segment
        )

    def setup(self, stage=None):
        self.dataset = NonstationaryLTIDataset(
            num_comp=self.hparams.num_comp,
            num_data=self.hparams.num_data,
            num_segment=self.hparams.num_segment,
            dt=self.hparams.dt,
            triangular=self.hparams.triangular,
            use_B=self.hparams.use_B,
            zero_means=self.hparams.zero_means,
            use_C=self.hparams.use_C,
            max_variability=self.hparams.max_variability,
            system_type=self.hparams.system_type,
            ar_order=self.hparams.ar_order,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=False
        )
