from lti_ica.dataset import NonstationaryLTIDataset
from state_space_models.state_space_models.lti import LTISystem

import pytest


@pytest.mark.parametrize("triangular", [True, False])
def test_dataset_default_params(
    triangular, num_comp, num_segment, num_data_per_segment, dt, ar_order
):
    num_data = num_data_per_segment * num_segment
    dataset = NonstationaryLTIDataset(
        num_comp, num_data, num_segment, dt=dt, triangular=triangular, ar_order=ar_order
    )
    assert dataset.segment_means.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.segment_variances.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.observations.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.states.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.controls.shape == (num_data, num_comp)
    assert dataset.segment_indices.shape == (num_data,)
    assert isinstance(dataset.lti, LTISystem)


@pytest.mark.parametrize("use_B", [True, False])
def test_dataset_custom_params_B(
    use_B, num_comp, num_segment, num_data_per_segment, dt, ar_order
):
    num_data = num_segment * num_data_per_segment
    triangular = False
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
    assert dataset.segment_means.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.segment_variances.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.observations.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.states.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.controls.shape == (num_data, num_comp)
    assert isinstance(dataset.lti, LTISystem)


@pytest.mark.parametrize("use_C", [True, False])
def test_dataset_custom_params_C(
    use_C, num_comp, num_segment, num_data_per_segment, dt, ar_order
):
    num_data = num_segment * num_data_per_segment
    triangular = False
    dataset = NonstationaryLTIDataset(
        num_comp,
        num_data,
        num_segment,
        dt=dt,
        triangular=triangular,
        use_B=True,
        zero_means=True,
        max_variability=False,
        use_C=use_C,
        ar_order=ar_order,
    )
    assert dataset.segment_means.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.segment_variances.shape == (
        dataset.num_segment,
        dataset.num_comp,
    )
    assert dataset.observations.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.states.shape == (
        num_data // (ar_order + 1),
        ar_order + 1,
        num_comp,
    )
    assert dataset.controls.shape == (num_data, num_comp)
    assert isinstance(dataset.lti, LTISystem)


from torch.utils.data import DataLoader


@pytest.mark.parametrize("batch_size", [1, 64])
def test_return_correct_tensor_shapes(
    batch_size, num_comp, num_segment, num_data_per_segment, dt, ar_order
):
    num_data = num_data_per_segment * num_segment
    dataset = NonstationaryLTIDataset(
        num_comp, num_data, num_segment, dt=dt, ar_order=ar_order, triangular=False
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    observations, states, controls, _, _, _ = next(iter(dataloader))

    assert observations.shape == (batch_size, dataset.ar_order + 1, dataset.num_comp)
    assert states.shape == (batch_size, dataset.ar_order + 1, dataset.num_comp)
    assert controls.shape == (batch_size, dataset.num_comp)
