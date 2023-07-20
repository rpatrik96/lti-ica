def test_rescaling_A():
    # create an instance of the model with num_dim=3 and
    model = LTINet(
        num_dim=3,
    )
    # get the eigenvalues of A
    eigvals = torch.linalg.eig(model.A.weight.data)[0]
    # check that all eigenvalues are less than 1
    assert torch.all(eigvals < 1)


def test_input_shape():
    # create an instance of the model with num_dim=3 and
    model = LTINet(
        num_dim=3,
    )
    # create a dummy input tensor with shape [batch, time(t:t-p), dim]
    x = torch.randn(5, 10, 3)
    # pass the input through the model
    output = model(x)
    # check that the output shape is [batch, dim]
    assert output.shape == (5, 3)


def test_C_inv():
    # create an instance of the model with num_dim=3 and , and C_inv set to False
    model = LTINet(num_dim=3, C=False)
    # create a dummy input tensor with shape [batch, time(t:t-p), dim]
    x = torch.randn(5, 10, 3)
    # pass the input through the model
    output = model(x)
    # check that the output shape is [batch, dim]
    assert output.shape == (5, 3)


def test_identity_matrix_B_inv():
    # create an instance of the model with num_dim=3 and
    model = LTINet(num_dim=3)
    # check that B_inv is set to identity matrix
    assert torch.allclose(torch.eye(model.num_dim), model.B_inv.weight)
    # check that B_inv requires_grad is False
    assert model.B_inv.weight.requires_grad == False


def test_triangular():
    # create an instance of the model with num_dim=3 and , and triangular set to True
    model = LTINet(num_dim=3, triangular=True)
    # create a dummy input tensor with shape [batch, time(t:t-p), dim]
    x = torch.randn(5, 10, 3)
    # pass the input through the model
    output = model(x)
    # check that the output shape is [batch, dim]
    assert output.shape == (5, 3)

    # check that all elements above the main diagonal are 0
    A = model.A.weight.data
    assert torch.triu(A, diagonal=1).sum() == 0
