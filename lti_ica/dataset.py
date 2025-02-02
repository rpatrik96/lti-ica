import torch
from torch.utils.data import Dataset
import numpy as np

from lti_ica.data import generate_nonstationary_data, generate_segment_stats


from state_space_models.state_space_models.lti import (
    LTISystem,
    SpringMassDamper,
    DCMotor,
)


"""Pytorch dataset for the data generated from nonstationary segments"""


class NonstationaryLTIDataset(Dataset):
    """
    A pytorch dataset that uses the data generation methods from `data.py` to generate nonstationary data.
     __getitem__ returns the ith segment, its mean and variance
    """

    def __init__(
        self,
        num_comp,
        num_data,
        num_segment,
        dt,
        triangular,
        use_B=True,
        zero_means=True,
        max_variability=False,
        use_C=True,
        system_type="lti",
        ar_order=1,
        obs_noise_var=0,
    ):
        super().__init__()
        self.num_comp = num_comp
        self.num_data = num_data
        self.dt = dt
        self.triangular = triangular
        self.use_B = use_B
        self.zero_means = zero_means
        self.use_C = use_C
        self.max_variability = max_variability
        self.obs_noise_var = np.array(obs_noise_var)

        if self.max_variability is True and num_segment != num_comp + 1:
            print("Overwriting num_segment to construct a maximally variable system")
            num_segment = num_comp + 1

        self.num_segment = num_segment

        self.system_type = system_type
        self.ar_order = ar_order

        # Remake label for TCL learning
        self.num_data_per_segment = self.num_data // self.num_segment

        # this ensures that after reshaping there is no overlap between segments in an observation tuple
        assert self.num_data_per_segment % (self.ar_order + 1) == 0

        if system_type == "lti":
            self.lti = LTISystem.controllable_system(
                self.num_comp,
                self.num_comp,
                dt=self.dt,
                triangular=self.triangular,
                use_B=self.use_B,
                use_C=self.use_C,
            )
        elif system_type == "spring_mass_damper":
            self.lti = SpringMassDamper.from_params(dt=self.dt)
        elif system_type == "dc_motor":
            self.lti = DCMotor.from_params(dt=self.dt)
        else:
            raise ValueError(f"Unknown system type {system_type=}")

        self.segment_means, self.segment_variances = generate_segment_stats(
            self.num_comp,
            self.num_segment,
            zero_means=self.zero_means,
            max_variability=self.max_variability,
            control_dim=None if system_type == "lti" else self.lti.B.shape[1],
        )

        observations, states, controls = generate_nonstationary_data(
            self.lti,
            self.segment_means,
            self.segment_variances,
            self.num_data_per_segment,
            self.dt,
            self.obs_noise_var,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        segment_indices = np.repeat(
            np.arange(self.num_segment), self.num_data_per_segment
        )

        self.segment_indices = torch.from_numpy(segment_indices.astype(int)).to(
            self.device
        )

        observations = observations.reshape([-1, self.ar_order + 1, self.num_comp])

        self.observations = torch.from_numpy(observations.astype(np.float32)).to(
            self.device
        )
        self.segment_variances = torch.from_numpy(
            self.segment_variances.astype(np.float32)
        ).to(self.device)
        self.segment_means = torch.from_numpy(self.segment_means.astype(np.float32)).to(
            self.device
        )

        states = states.reshape([-1, self.ar_order + 1, self.num_comp])
        self.states = torch.from_numpy(states.astype(np.float32)).to(self.device)
        self.controls = torch.from_numpy(controls.astype(np.float32)).to(self.device)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        # observations contains (y_t, y_{t+1},...,y_{t+ar_order-1}) which can be used to predict u_t
        segment_idx = self.segment_indices[(self.ar_order + 1) * idx]
        return (
            self.observations[idx],
            self.states[idx],
            self.controls[(self.ar_order + 1) * idx],
            segment_idx,
            self.segment_means[segment_idx],
            self.segment_variances[segment_idx],
        )
