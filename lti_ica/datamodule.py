"""A pytorch lightning datamodule that uses the dataset class from `dataset.py`"""


from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

from lti_ica.dataset import NonstationaryLTIDataset


class NonstationaryLTIDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        num_comp,
        num_data,
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
        self.num_comp = num_comp
        self.num_data = num_data
        self.num_segment = num_segment
        self.dt = dt
        self.triangular = triangular
        self.use_B = use_B
        self.zero_means = zero_means
        self.use_C = use_C
        self.max_variability = max_variability
        self.system_type = system_type
        self.ar_order = ar_order
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = NonstationaryLTIDataset(
            num_comp=self.num_comp,
            num_data=self.num_data,
            num_segment=self.num_segment,
            dt=self.dt,
            triangular=self.triangular,
            use_B=self.use_B,
            zero_means=self.zero_means,
            use_C=self.use_C,
            max_variability=self.max_variability,
            system_type=self.system_type,
            ar_order=self.ar_order,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
