from lti_ica.models import LTINet
import torch


def test_rescaling_A(num_comp, num_segment):
    # create an instance of the model with num_dim=num_comp and
    model = LTINet(
        num_dim=num_comp,
        num_class=num_segment,
    )
    # get the eigenvalues of A
    eigvals = torch.linalg.eig(model.A.weight.data)[0]
    # check that all eigenvalues lie within the unit circle
    assert torch.all(eigvals.abs() < 1)


def test_input_shape(num_comp, num_segment):
    # create an instance of the model with num_dim=num_comp and
    model = LTINet(
        num_dim=num_comp,
        num_class=num_segment,
    )
    # create a dummy input tensor with shape [batch, time(t:t-p), dim]
    x = torch.randn(5, 10, 3)
    # pass the input through the model
    output = model(x)
    # check that the output shape is [batch, dim]
    assert output.shape == (5, 3)


def test_C_inv(num_comp, num_segment):
    # create an instance of the model with num_dim=num_comp and , and C_inv set to False
    model = LTINet(num_dim=num_comp, num_class=num_segment, C=False)
    # create a dummy input tensor with shape [batch, time(t:t-p), dim]
    x = torch.randn(5, 10, 3)
    # pass the input through the model
    output = model(x)
    # check that the output shape is [batch, dim]
    assert output.shape == (5, 3)


def test_identity_matrix_B_inv(num_comp, num_segment):
    # create an instance of the model with num_dim=num_comp and
    model = LTINet(num_dim=num_comp, num_class=num_segment, B=False)

    assert torch.allclose(torch.eye(model.num_dim), model.B_inv)
    assert model.B_inv.requires_grad == False
