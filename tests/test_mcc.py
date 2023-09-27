import torch

from lti_ica.mcc import calc_mcc


def test_calc_mcc(num_comp, num_segment):
    s = torch.randn(num_segment, num_comp)
    s_hat = torch.randn(num_segment, num_comp)

    mcc = calc_mcc(s, s_hat)
    assert isinstance(mcc, float)
