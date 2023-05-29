
# =============================================================
# =============================================================
class LTINet(nn.Module):
    def __init__(self,  num_dim, num_class, C=True, triangular=False):
        """ Network model for segment-wise stationary model for LTI systems
         Args:
             num_dim: number of dimension
             num_class: number of classes
         """
        super().__init__()

        self.B_inv = nn.Linear(num_dim, num_dim, bias=False)
        self.A = nn.Linear(num_dim, num_dim, bias=False)
        self.C_inv = nn.Linear(num_dim, num_dim, bias=False) if C is True else torch.eye(num_dim)
        self.I = torch.eye(num_dim)
        self.triangular = triangular


        self.num_dim = num_dim

        # initialize
        for k in [self.B_inv, self.A, self.C_inv]:
            try:
                torch.nn.init.orthogonal_(k.weight)
            except:
                pass

        torch.nn.init.orthogonal_(self.mlr.weight)

        # rescale A such that all eigenvalues are < 1
        self.A.weight.data = self.A.weight.data / torch.max(torch.abs(torch.linalg.eig(self.A.weight.data)[0])) * 0.9
        self.B_inv.weight.data = torch.eye(num_dim)
        self.B_inv.requires_grad = False

    def forward(self, x):
        """ forward
         Args:
             x: input [batch, time(t:t-p), dim]
         """


        if self.C_inv is nn.Linear:
            cinvx = self.C_inv(x)
        else:
            cinvx = x

        cinvxt = cinvx[:, 0, :]
        cinvxtplusone = cinvx[:, 1, :]
        if self.triangular:
            hout = self.B_inv(cinvxtplusone - cinvxt@self.A.weight.tril())
        else:
            hout = self.B_inv(cinvxtplusone - self.A(cinvxt))


        return hout
