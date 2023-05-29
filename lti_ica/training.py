import torch
import numpy as np
import lti_ica.models



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

    model = lti_ica.models.LTINet(num_dim=data.shape[-1],
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

