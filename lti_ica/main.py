""" Training
    Main script for training the model
"""
import pytorch_lightning as pl

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_comp = 2  # number of components (dimension)
ar_order = 1
random_seed = 568  # random seed
triangular = False
num_segment = 3
data_per_segment = 2**11
num_data = num_segment * (data_per_segment * 2)
zero_means = True

use_B = True
use_C = True
max_variability = False
system_type = "lti"  # "lti" or "spring_mass_damper"

# Training ----------------------------------------------------
num_epoch = 30
num_epoch_mse = 1000
model = "mlp"

dt = 0.003
lr = 3e-3
max_norm = 0.25
num_experiment = 1
save = True

import numpy as np
import torch

if __name__ == "__main__":
    # Generate sensor signal --------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    """setup the datamodule"""
    from datamodule import NonstationaryLTIDatamodule

    datamodule = NonstationaryLTIDatamodule(
        num_comp,
        num_data,
        num_segment,
        dt,
        triangular,
        use_B,
        zero_means,
        max_variability,
        use_C,
        system_type,
        ar_order,
        batch_size=1,
    )
    datamodule.setup()
    dataset = datamodule.train_dataloader().dataset

    """use the lightning module to train the model"""
    from runner import LTILightning

    runner = LTILightning(
        num_comp,
        num_data,
        num_segment,
        dt,
        triangular,
        use_B,
        zero_means,
        max_variability,
        use_C,
        system_type,
        ar_order,
        batch_size=1,
        lr=lr,
        max_norm=max_norm,
        model=model,
    )

    """run the training with the lightning module"""
    trainer = pl.Trainer(
        max_epochs=num_epoch, gradient_clip_val=max_norm, val_check_interval=1
    )
    trainer.fit(runner, datamodule)
