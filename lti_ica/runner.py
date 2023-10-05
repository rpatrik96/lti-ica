"""pytorch lightning module for training the ICA model"""

import subprocess
from os.path import dirname

import pytorch_lightning as pl
import pytorch_lightning.loggers.wandb
import torch
import wandb
from torch.optim import SGD

from lti_ica.mcc import calc_mcc
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
        ar_order=1,
        lr=1e-3,
        model="mlp",
        offline=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self._setup_model()

    def _setup_model(self):
        if self.hparams.model == "lti":
            self.model = LTINet(
                num_dim=self.hparams.num_comp,
                num_class=self.hparams.num_segment,
                C=self.hparams.use_C,
                triangular=self.hparams.triangular,
                B=self.hparams.use_B,
            )
        elif self.hparams.model == "mlp":
            self.model = LTINetMLP(num_dim=self.hparams.num_comp)

    def training_step(self, batch, batch_idx):
        (
            log_likelihood,
            observations,
            states,
            controls,
            predicted_control,
        ) = self._forward(batch)

        self.log("train_log_likelihood", log_likelihood)

        mcc = calc_mcc(s=controls, s_hat=predicted_control)

        self.log("train_mcc", mcc)

        return -log_likelihood

    def _forward(self, batch):
        (
            observations,
            states,
            controls,
            segment_indices,
            segment_means,
            segment_variances,
        ) = batch

        # convert segment_var from size (batch, num_segment, dim) to (batch, num_segment, dim, dim)
        # such that it is a batch of diagonal matrices
        segment_variances = segment_variances.diag_embed()

        predicted_control = self.model(observations)
        log_likelihood = (
            torch.distributions.MultivariateNormal(segment_means, segment_variances)
            .log_prob(predicted_control)
            .mean()
        )
        return log_likelihood, observations, states, controls, predicted_control

    def validation_step(self, batch, batch_idx):
        (
            log_likelihood,
            observations,
            states,
            controls,
            predicted_control,
        ) = self._forward(batch)

        mcc = calc_mcc(s=controls, s_hat=predicted_control)

        self.log("val_log_likelihood", log_likelihood)
        self.log("val_mcc", mcc)

        return -log_likelihood

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.hparams.lr)

    def on_fit_end(self) -> None:
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger) is True:
            logger: pl.loggers.wandb.WandbLogger = self.logger
            if self.hparams.offline is True:  # type: ignore [attr-defined]
                # Syncing W&B at the end
                # 1. save sync dir (after marking a run finished, the W&B object changes (is teared down?)
                sync_dir = dirname(logger.experiment.dir)
                # 2. mark run complete
                wandb.finish()  # type: ignore
                # 3. call the sync command for the run directory
                subprocess.check_call(["wandb", "sync", sync_dir])
