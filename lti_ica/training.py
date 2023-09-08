import numpy as np
import torch

import lti_ica.models


def regularized_log_likelihood(
    dataset,
    num_epoch=1000,
    lr=1e-3,
    triangular=False,
    max_norm=0.5,
    use_B=True,
    model="mlp",
):
    """
    A function that takes data stratified into segments.
    Assuming each segment is distributed according to a multivariate Gaussian,
    it calculates the log likelihood and maximizes it with torch
    """

    # Initialize random weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model == "lti":
        model = lti_ica.models.LTINet(
            num_dim=dataset.observations.shape[-1],
            num_class=dataset.num_segment,
            C=False,
            triangular=triangular,
            B=use_B,
        )
    elif model == "mlp":
        model = lti_ica.models.LTINetMLP(num_dim=dataset.observations.shape[-1])

    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # loop over the optimizer
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        # learn the latents from the segments
        latents = []
        log_likelihood = 0

        for (
            segment,
            segment_mean,
            segment_var,
            _,
        ) in dataset:
            segment_var = segment_var.diag()

            latent = model(segment)

            log_likelihood += (
                torch.distributions.MultivariateNormal(segment_mean, segment_var)
                .log_prob(latent)
                .mean()
            )

        (-log_likelihood).backward()
        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, log_likelihood: {log_likelihood}")

    return model
