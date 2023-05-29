""" Training
    Main script for training the model
"""

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_comp = 3  # number of components (dimension)
num_data = 2 ** 14  # number of data points
ar_order = 1
random_seed = 42  # random seed
triangular = True
num_segment = 16  # learn by IIA-TCL

# Training ----------------------------------------------------
num_epoch = 350
dt = 0.001
lr = 3e-3
batch_size = 512  # mini-batch size
max_norm = 0.5
num_experiment = 5

import numpy as np
import pandas as pd
import torch

import lti_ica.mcc
from lti_ica.data import generate_nonstationary_data, generate_segment_stats
from lti_ica.training import regularized_log_likelihood
from state_space_models.state_space_models.lti import LTISystem


def data_gen(num_comp, dt, triangular):
    lti = LTISystem.controllable_system(num_comp, num_comp, dt=dt, triangular=triangular)
    segment_means, segment_variances = generate_segment_stats(num_comp, num_segment)

    # Remake label for TCL learning
    num_segmentdata = int(np.ceil(num_data / num_segment))

    x, s = generate_nonstationary_data(lti, segment_means, segment_variances, num_comp, num_segmentdata, dt)
    return segment_means, segment_variances, x, s


def calc_mcc(model, x, s, ar_order=1):
    estimated_factors = model(torch.from_numpy(x.T.astype(np.float32).reshape([-1, ar_order + 1, x.T.shape[1]])))
    mat, _, _ = lti_ica.mcc.correlation(
        s[:, 0::2],  # since we use xt, xtplusone, we only have half the preds
        estimated_factors.detach().numpy().T,
        method="Pearson",
    )
    mcc = np.mean(np.abs(np.diag(mat)))

    return mcc


if __name__ == '__main__':
    # Generate sensor signal --------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    segment_means, segment_variances, x, s = data_gen(num_comp, dt, triangular)

    mccs = []

    # run experiments
    for i in range(num_experiment):
        model = regularized_log_likelihood(x.T, num_segment, segment_means, segment_variances, num_epoch=num_epoch,
                                           lr=lr)
        mccs.append(calc_mcc(model, x, s, ar_order))

    filename = f"seed_{random_seed}_segment_{num_segment}_comp_{num_comp}_triangular_{triangular}.csv"

    # convert mccs list to numpy and calculate mean and std
    mccs = np.array(mccs)
    mcc_mean = np.mean(mccs)
    mcc_std = np.std(mccs)

    # Define your data as a dictionary or a list of dictionaries
    data = [{'mcc_mean': mcc_mean, "mcc_std": mcc_std, 'random_seed': random_seed, 'dt': dt, 'num_segment': num_segment,
             'num_comp': num_comp, 'num_data': num_data, 'num_epoch': num_epoch, 'lr': lr,
             'batch_size': batch_size, 'triangular': triangular}]

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file with column names
    df.to_csv(filename, index=False, header=True, mode='a')
