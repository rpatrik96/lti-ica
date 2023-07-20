""" Training
    Main script for training the model
"""
from lti_ica.mcc import calc_mcc

# Parameters ==================================================
# =============================================================

# Data generation ---------------------------------------------
num_comp = 10  # number of components (dimension)
ar_order = 1
random_seed = 568  # random seed
triangular = False
num_segment = 11
data_per_segment = 2**11
num_data = num_segment * (data_per_segment * 2)
zero_means = True

use_B = True
use_C = True
max_variability = True

# Training ----------------------------------------------------
num_epoch = 10000
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
        mccs.append(calc_mcc(model, x, s, ar_order))

        print(f"mcc: {mccs[-1]}")

    filename = f"seed_{random_seed}_segment_{num_segment}_comp_{num_comp}_triangular_{triangular}_use_B_{use_B}_use_C_{use_C}_max_variability_{max_variability}.csv"

    # convert mccs list to numpy and calculate mean and std
    mccs = np.array(mccs)
    mcc_mean = np.mean(mccs)
    mcc_std = np.std(mccs)

    print("------------------------------------")
    print(f"mcc_mean: {mcc_mean}, mcc_std: {mcc_std}")
    print("------------------------------------")

    if False and isinstance(model, lti_ica.models.LTINetMLP):
        # parametrize A, B_inv, C_inv and learn them to match model.net.weight.data in the MSE sense

        A: torch.nn.Parameter = torch.nn.Parameter(torch.randn(num_comp, num_comp))
        B_inv: torch.nn.Parameter = torch.nn.Parameter(torch.randn(num_comp, num_comp))
        # C_inv = torch.nn.Parameter(torch.randn(num_comp, num_comp))
        C_inv: torch.nn.Parameter = torch.eye(num_comp, dtype=torch.float32)
        eye = torch.eye(num_comp, dtype=torch.float32)

        optimizer = torch.optim.Adam([A, B_inv], lr=3e-3)
        target = model.net.weight.data.detach()

        # todo: the problem is that the estimate can be unstable (eigval>1)

        for i in range(num_epoch_mse):
            optimizer.zero_grad()
            # calculate loss

            est = (
                B_inv
                @ torch.cat((eye, eye), dim=1)
                @ torch.block_diag(C_inv, -A @ C_inv)
            )

            loss = torch.mean((target - est) ** 2)
            # backprop
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"epoch: {i}, loss: {loss}")

    # evaluate
    eval = False
    # generate new data and calculate MSE between predicted and true output
    # segment_means, segment_variances, x, s = data_gen(num_comp, dt, triangular, use_B)

    if eval is True:
        # extract the A,B,C matrices from teh model
        if isinstance(model, lti_ica.models.LTINet):
            A_est = model.A.detach().numpy()
            B_est = model.B_inv.inverse().detach().numpy()
            C_est = model.C_inv.inverse().detach().numpy()
        elif isinstance(model, lti_ica.models.LTINetMLP):
            A_est = A.detach().numpy()
            B_est = B_inv.inverse().detach().numpy()
            C_est = C_inv.inverse().detach().numpy()

        # create a scipy LTI object from the matrices
        lti_est = LTISystem(A_est, B_est, C_est, dt=dt)

        # generate new data from a multivariate normal
        cov = np.diag(np.random.uniform(0.1, 1, size=num_comp))
        u = np.random.multivariate_normal(
            np.zeros(num_comp), cov, size=data_per_segment
        )

        # simulate x from s with lti
        t, out, state = lti.simulate(u)
        t, out_est, state_est = lti_est.simulate(u)

        # MSE between x and out_est
        mse = np.mean((out - out_est) ** 2)
        print("------------------------------------")
        print(f"MSE: {mse}")
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
