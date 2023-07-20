import pytest

from lti_ica.data import data_gen, generate_segment_stats
from state_space_models.state_space_models.lti import LTISystem
import numpy as np

@pytest.mark.parametrize("zero_means", [True, False])
def test_generate_segment_stats_zero_means(zero_means):
    num_comp = 3
    num_segment = 4
    segment_means, segment_variances = generate_segment_stats(num_comp, num_segment, zero_means=zero_means,
                                                              max_variability=False)
    assert segment_means.shape == (num_segment, num_comp)
    assert segment_variances.shape == (num_segment, num_comp)
    assert np.all(segment_variances >= 0)

    if zero_means is True:
        assert np.all(segment_means == 0)

@pytest.mark.parametrize("max_variability", [True, False])
def test_generate_segment_stats_max_variability(max_variability):

    num_comp = 3
    num_segment = 4
    segment_means, segment_variances = generate_segment_stats(num_comp, num_segment, zero_means=False,
                                                              max_variability=max_variability)
    assert segment_means.shape == (num_segment, num_comp)
    assert segment_variances.shape == (num_segment, num_comp)
    assert np.all(segment_variances >= 0)

    if max_variability is True:

        # check for the 1's
        rows = list(range(1, num_segment))
        cols = list(range(num_comp))
        assert np.all(segment_variances[rows, cols] == 1.)

        # check for the other values (masking out the 1's to make comparison easier)
        segment_variances_copy = segment_variances.copy()
        segment_variances_copy[rows, cols] = 0.0001
        assert np.all(segment_variances_copy == 0.0001)


@pytest.mark.parametrize("triangular", [True, False])
def test_data_gen_default_params(triangular):
    num_segment = 4
    num_data = 3000
    num_comp = 3
    segment_means, segment_variances, x, s, lti = data_gen(num_comp, num_data, num_segment, dt=0.01, triangular=triangular)
    assert len(segment_means) == num_segment
    assert len(segment_variances) == num_segment
    assert x.shape == (num_comp, num_data)
    assert s.shape == (num_comp, num_data)
    assert isinstance(lti, LTISystem)

@pytest.mark.parametrize("use_B", [True, False])
def test_data_gen_custom_params_B(use_B):
    num_segment = 4
    num_data = 3000
    num_comp = 3
    triangular = False
    segment_means, segment_variances, x, s, lti = data_gen(num_comp, num_data, num_segment, dt=0.01, triangular=triangular, use_B=use_B, zero_means=False,
                                                           max_variability=False, use_C=True)
    assert len(segment_means) == num_segment
    assert len(segment_variances) == num_segment
    assert x.shape == (num_comp, num_data)
    assert s.shape == (num_comp, num_data)
    assert isinstance(lti, LTISystem)

@pytest.mark.parametrize("use_C", [True, False])
def test_data_gen_custom_params_C(use_C):
    num_segment = 4
    num_data = 3000
    num_comp = 3
    triangular = False
    segment_means, segment_variances, x, s, lti = data_gen(num_comp, num_data, num_segment, dt=0.01, triangular=triangular, use_B=True, zero_means=True,
                                                           max_variability=False, use_C=use_C)
    assert len(segment_means) == num_segment
    assert len(segment_variances) == num_segment
    assert x.shape == (num_comp, num_data)
    assert s.shape == (num_comp, num_data)
    assert isinstance(lti, LTISystem)

