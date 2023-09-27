from lti_ica.mcc import calc_mcc
import numpy as np
import torch
from lti_ica.data import generate_segment_stats, generate_nonstationary_data
from state_space_models.state_space_models.lti import LTISystem
import lti_ica.models
import pytest
from lti_ica.dataset import NonstationaryLTIDataset


@pytest.mark.parametrize("model", ["lti", "mlp"])
def test_calc_mcc(model, num_comp, num_segment, num_segmentdata, dt, ar_order, device):
    num_data = num_segment * num_segmentdata

    triangular = False
    use_B = True

    dataset = NonstationaryLTIDataset(
        num_comp,
        num_data,
        num_segment,
        dt=dt,
        triangular=triangular,
        use_B=use_B,
        zero_means=False,
        max_variability=False,
        use_C=True,
        ar_order=ar_order,
    )

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

    mcc = calc_mcc(dataset.sources, model(dataset.observations), ar_order)
    assert isinstance(mcc, float)
