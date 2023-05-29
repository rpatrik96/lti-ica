""" Training
    Main script for training the model
"""

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_layer = 3  # number of layers of mixing-MLP
num_comp = 3  # number of components (dimension)
num_data = 2 ** 14  # number of data points
num_basis = 64  # number of frequencies of fourier bases
modulate_range = [-2, 2]
modulate_range2 = [-2, 2]
ar_order = 1
random_seed = 42  # random seed
triangular = True
num_segment = 16  # learn by IIA-TCL

# MLP ---------------------------------------------------------
list_hidden_nodes = [4 * num_comp] * (num_layer - 1) + [num_comp]
list_hidden_nodes_z = None
# list of the number of nodes of each hidden layer of feature-MLP
# [layer1, layer2, ..., layer(num_layer)]


# Training ----------------------------------------------------
initial_learning_rate = 0.001  # initial learning rate (default:0.1)
num_epoch = 350
dt = 0.001
lr = 3e-3
momentum = 0.9  # momentum parameter of SGD
max_steps = int(15e4)  # number of iterations (mini-batches)
decay_steps = int(1e6)  # decay steps (tf.train.exponential_decay)
decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
batch_size = 512  # mini-batch size
moving_average_decay = 0.999  # moving average decay of variables to be saved
checkpoint_steps = int(1e7)  # interval to save checkpoint
summary_steps = int(1e4)  # interval to save summary
apply_pca = True  # apply PCA for preprocessing or not
# todo: turn PCA off
weight_decay = 1e-5  # weight decay

import numpy as np
import torch

from lti_ica.data import generate_nonstationary_data, generate_segment_stats
from lti_ica.training import regularized_log_likelihood

if __name__ == '__main__':
    # Generate sensor signal --------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Remake label for TCL learning
    num_segmentdata = int(np.ceil(num_data / num_segment))
    y = np.tile(np.arange(num_segment), [num_segmentdata, 1]).T.reshape(-1)[:num_data]

    from state_space_models.state_space_models.lti import LTISystem

    lti = LTISystem.controllable_system(num_comp, num_comp, dt=dt, triangular=triangular)

    segment_means, segment_variances = generate_segment_stats(num_comp, num_segment)

    x, s = generate_nonstationary_data(lti, segment_means, segment_variances, num_comp, num_segmentdata, dt)

    model = regularized_log_likelihood(x.T, num_segment, segment_means, segment_variances, num_epoch=num_epoch, lr=lr)

    # calculate MCC
    _, estimated_factors, _ = model(torch.from_numpy(x.T.astype(np.float32).reshape([-1, ar_order + 1, x.T.shape[1]])))
    import mcc

    mat, _, _ = mcc.correlation(
        s[:, 0::2],  # since we use xt, xtplusone, we oonly have half the preds
        estimated_factors.detach().numpy().T,
        method="Pearson",
    )
    mcc = np.mean(np.abs(np.diag(mat)))

    print(f"{mcc=}")
