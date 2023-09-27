import pytest
from lti_ica.models import LTINet, LTINetMLP
from lti_ica.datamodule import NonstationaryLTIDatamodule
import torch


@pytest.fixture
def num_comp():
    return 3


@pytest.fixture
def num_segment():
    return 4


@pytest.fixture
def num_segmentdata():
    return 3000


@pytest.fixture
def dt():
    return 0.01


@pytest.fixture
def ar_order():
    return 1


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @pytest.fixture(use_B=[True, False]])
# def lti_model(use_B):
#     model = lti_ica.models.LTINet(num_dim=data.shape[-1],
#                                   num_class=num_segment, C=False, triangular=triangular, B=use_B)
#
# @pytest.fixture
# def lti_mlp_model():
#     model = lti_ica.models.LTINetMLP(num_dim=data.shape[-1])


@pytest.fixture
def datamodule(num_comp, num_segmentdata, num_segment, dt):
    num_data = num_segmentdata * num_segment
    datamodule = NonstationaryLTIDatamodule(num_comp, num_data, num_segment, dt)
    datamodule.setup()

    return datamodule
