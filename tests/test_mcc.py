from lti_ica.mcc import calc_mcc
import numpy as np
import torch
from lti_ica.data import generate_segment_stats, generate_nonstationary_data
from state_space_models.state_space_models.lti import LTISystem
import lti_ica.models
import pytest

@pytest.mark.parametrize("model", ["lti", "mlp"])
def test_calc_mcc(model):
    num_comp = 3
    num_segment = 4
    num_segmentdata = 3000
    dt = 0.01
    ar_order = 1
    segment_means, segment_variances = generate_segment_stats(num_comp, num_segment, zero_means=False,
                                                              max_variability=False)
    lti = LTISystem.controllable_system(num_comp, num_comp, dt=dt)
    x, s = generate_nonstationary_data(lti, segment_means, segment_variances, num_segmentdata, dt)

    # Initialize random weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make shuffled batch
    ar_order = 1

    data = x.T.reshape([-1, ar_order + 1, x.T.shape[1]])

    data = torch.from_numpy(data.astype(np.float32)).to(device)
    segment_variances = torch.from_numpy(segment_variances.astype(np.float32)).to(device)
    segment_means = torch.from_numpy(segment_means.astype(np.float32)).to(device)

    if model == "lti":
        model = lti_ica.models.LTINet(num_dim=data.shape[-1],
                                  num_class=num_segment, C=False, triangular=False, B=True)
    elif model == "mlp":
        model = lti_ica.models.LTINetMLP(num_dim=data.shape[-1])


    model = model.to(device)
    model.train()


    mcc = calc_mcc(model, x, s, ar_order)
    assert isinstance(mcc, float)


