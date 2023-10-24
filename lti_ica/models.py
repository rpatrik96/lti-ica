import torch
from torch import nn


# =============================================================
# =============================================================
class LTINet(nn.Module):
    def __init__(self, num_dim, num_class, B=True, C=True, triangular=False):
        """Network model for segment-wise stationary model for LTI systems
        Args:
            num_dim: number of dimension
            num_class: number of classes
        """
        super().__init__()

        self.A = nn.Linear(num_dim, num_dim, bias=False)
        self.B_inv = (
            nn.Linear(num_dim, num_dim, bias=False) if B is True else torch.eye(num_dim)
        )
        self.C_inv = (
            nn.Linear(num_dim, num_dim, bias=False) if C is True else torch.eye(num_dim)
        )
        self.I = torch.eye(num_dim)
        self.triangular = triangular

        self.num_dim = num_dim

        # initialize
        for k in [self.B_inv, self.A, self.C_inv]:
            try:
                torch.nn.init.orthogonal_(k.weight)
            except:
                pass

        # rescale A such that all eigenvalues are < 1
        self.A.weight.data = (
            self.A.weight.data
            / torch.max(torch.abs(torch.linalg.eig(self.A.weight.data)[0]))
            * 0.9
        )

    def forward(self, x):
        """forward
        Args:
            x: input [batch, time(t:t-p), dim]
        """

        if self.C_inv is nn.Linear:
            cinvxt = self.C_inv(x[:, 0, :])
            cinvxtplusone = self.C_inv(x[:, 1, :])
        else:
            cinvxt = x[:, 0, :]
            cinvxtplusone = x[:, 1, :]

        if self.triangular:
            hout = cinvxtplusone - cinvxt @ self.A.weight.tril()
        else:
            hout = cinvxtplusone - self.A(cinvxt)

        if self.B_inv is nn.Linear:
            hout = self.B_inv(hout)

        return hout


class LTINetMLP(nn.Module):
    def __init__(self, state_dim, control_dim=None):
        """Network model for segment-wise stationary model for LTI systems
        Args:
            state_dim: dimensionality of the (observed) state
            control_dim: dimensionality of the control signal
        """
        super().__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim if control_dim is not None else state_dim
        self.net = nn.Linear(2 * state_dim, control_dim, bias=False)

        torch.nn.init.orthogonal_(self.net.weight)

    def forward(self, x):
        """forward
        Args:
            x: input [batch, time(t:t-p), dim]
        """

        return self.net(x.reshape(-1, 2 * self.state_dim))
