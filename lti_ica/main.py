""" Training
    Main script for training the model
"""
from lti_ica.mcc import calc_mcc

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_comp = 1  # number of components (dimension)
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
system_type = "spring_mass_damper"

# Training ----------------------------------------------------
num_epoch = 3000
num_epoch_mse = 1000
model = "mlp"

dt = 0.003
lr = 3e-3
max_norm = 0.25
num_experiment = 1
save = True

import numpy as np
import pandas as pd
import torch

import lti_ica.models
from lti_ica.data import data_gen
from lti_ica.training import regularized_log_likelihood
from state_space_models.state_space_models.lti import LTISystem

if __name__ == "__main__":
    # Generate sensor signal --------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    segment_means, segment_variances, x, s, lti = data_gen(
        num_comp,
        num_data,
        num_segment,
        dt,
        triangular,
        use_B,
        zero_means,
        max_variability,
        system_type,
    )

    mccs = []

    # run experiments
    for i in range(num_experiment):
        model = regularized_log_likelihood(
            x.T,
            num_segment,
            segment_means,
            segment_variances,
            num_epoch=num_epoch,
            lr=lr,
            model=model,
        )
        mccs.append(calc_mcc(model, x, s, ar_order, diff_dims=(system_type != "lti")))

        print(f"mcc: {mccs[-1]}")

    filename = f"seed_{random_seed}_segment_{num_segment}_comp_{num_comp}_triangular_{triangular}_use_B_{use_B}_use_C_{use_C}_max_variability_{max_variability}_{system_type}.csv"

    # convert mccs list to numpy and calculate mean and std
    mccs: np.ndarray = np.array(mccs)  # type: ignore
    mcc_mean = np.mean(mccs)
    mcc_std = np.std(mccs)

    print("------------------------------------")
    print(f"mcc_mean: {mcc_mean}, mcc_std: {mcc_std}")
    print("------------------------------------")

    if save is True:
        # Define your data as a dictionary or a list of dictionaries
        data = [
            {
                "mcc_mean": mcc_mean,
                "mcc_std": mcc_std,
                "random_seed": random_seed,
                "dt": dt,
                "num_segment": num_segment,
                "num_comp": num_comp,
                "num_data": num_data,
                "num_epoch": num_epoch,
                "lr": lr,
                "use_B": use_B,
                "use_C": use_C,
                "triangular": triangular,
            }
        ]

        # Create a DataFrame from the data
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file with column names
        df.to_csv(filename, index=False, header=True, mode="a")
