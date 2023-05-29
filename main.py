""" Training
    Main script for training the model
"""

import os
import shutil

from subfunc.preprocessing import pca
from subfunc.showdata import *

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

import torch
import torch.nn as nn


def regularized_log_likelihood(data, num_segment, segment_means, segment_variances, num_epoch=1000, lr=1e-3):
    """
    A function that takes data stratified into segments.
    Assuming each segment is distributed according to a multivariate Gaussian,
    it calculates the log likelihood and maximizes it with torch
    """

    # Initialize random weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make shuffled batch
    ar_order = 1

    data = data.reshape([-1, ar_order + 1, data.shape[1]])

    data = torch.from_numpy(data.astype(np.float32)).to(device)
    segment_variances = torch.from_numpy(segment_variances.astype(np.float32)).to(device)
    segment_means = torch.from_numpy(segment_means.astype(np.float32)).to(device)

    model = model.LTINet(num_dim=data.shape[-1],
                         num_class=num_segment, C=False, triangular=triangular)

    model = model.to(device)
    model.train()

    # split data into segments
    segments = torch.split(data, num_segment * [data.shape[0] // num_segment], dim=0)
    segments = [s for s in segments if s.shape[0] > 0]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # loop over the optimizer
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        # learn the latents from the segments
        latents = []
        for segment, segment_mean, segment_var in zip(segments, segment_means, segment_variances):
            segment_var = segment_var.diag()

            _, latent, _ = model(segment)

            log_likelihood = torch.distributions.MultivariateNormal(segment_mean, segment_var).log_prob(latent).mean()

            (-log_likelihood).backward()
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, log_likelihood: {log_likelihood}")

    return model


def generate_segment_stats():
    while rank < num_comp:
        segment_variances = np.abs(np.random.randn(num_segment, num_comp)) / 2

        print(f"{segment_variances=}")
        # segment_means = np.random.randn(num_segment, num_comp)
        segment_means = np.zeros((num_segment, num_comp))

        # check sufficient variability of the variances
        base_prec = 1. / segment_variances[0, :]
        delta_prec = 1. / segment_variances[1:, :] - base_prec
        if num_segment == 1:
            break
        rank = np.linalg.matrix_rank(delta_prec)

        print(f"rank: {rank}")
        print(f"Condition number: {np.linalg.cond(delta_prec)}")

        num_attempt += 1
        if num_attempt > 100:
            raise ValueError("Could not find sufficiently variable system!")

    return segment_means, segment_variances


def generate_nonstationary_data(segment_means, segment_variances):

    # iterate over the segment variances,
    # generate multivariate normal with each variance,
    # and simulate it with the LTI system
    obs = []
    states = []
    for i in range(num_segment):
        segment_var = segment_variances[i, :]
        segment_cov = np.diag(segment_var)

        # todo: change the means to be non-zero
        segment_mean = np.zeros(num_comp)

        segment_U = np.random.multivariate_normal(segment_mean, segment_cov, num_segmentdata)

        _, segment_obs, segment_state = lti.simulate(segment_U, dt=dt)

        obs.append(segment_obs)
        states.append(segment_state)
    obs = np.concatenate(obs, axis=0)
    states = np.concatenate(states, axis=0)
    x = obs.T
    s = states.T

    return x, s


if __name__ == '__main__':

    # Generate sensor signal --------------------------------------
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Remake label for TCL learning
    num_segmentdata = int(np.ceil(num_data / num_segment))
    y = np.tile(np.arange(num_segment), [num_segmentdata, 1]).T.reshape(-1)[:num_data]

    from state_space_models.lti import LTISystem

    lti = LTISystem.controllable_system(num_comp, num_comp, dt=dt, triangular=triangular)

    rank = 0
    num_attempt = 0

    segment_means, segment_variances = generate_segment_stats()

    x, s = generate_nonstationary_data(segment_means, segment_variances)

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
