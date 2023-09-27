"""pytorch lightning module for training the ICA model"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from lti_ica.dataset import NonstationaryLTIDataset
from lti_ica.models import LTINetMLP, LTINet


class LTILightning(pl.LightningModule):
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
        lr=1e-3,
        max_norm=1.0,
        model="mlp",
    ):
        super().__init__()

        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        if self.hparams.model == "lti":
            self.model = LTINet(
                num_dim=self.hparams.num_comp,
                num_class=self.hparams.num_segment,
                C=False,
                triangular=self.hparams.triangular,
                B=self.hparams.use_B,
            )
        elif self.hparams.model == "mlp":
            self.model = LTINetMLP(num_dim=self.hparams.num_comp)

    def training_step(self, batch, batch_idx):
        segment, segment_mean, segment_var = batch
        segment_var = segment_var.squeeze().diag()

        latent = self.model(segment)

        log_likelihood = (
            torch.distributions.MultivariateNormal(segment_mean, segment_var)
            .log_prob(latent)
            .mean()
        )

        self.log("train_log_likelihood", log_likelihood)

        return -log_likelihood

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.hparams.lr)
