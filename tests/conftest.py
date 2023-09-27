import pytest
import torch

from lti_ica.datamodule import NonstationaryLTIDatamodule
from lti_ica.runner import LTILightning


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


@pytest.fixture
def datamodule(num_comp, num_segmentdata, num_segment, dt):
    num_data = num_segmentdata * num_segment
    datamodule = NonstationaryLTIDatamodule(num_comp, num_data, num_segment, dt)
    datamodule.setup()

    return datamodule


@pytest.fixture
def runner(num_comp, num_segmentdata, num_segment, dt):
    num_data = num_segmentdata * num_segment
    return LTILightning(num_comp, num_data, num_segment, dt)
