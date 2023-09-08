import torch
from torch.utils.data import Dataset
import numpy as np

from lti_ica.data import generate_nonstationary_data, data_gen, generate_segment_stats


from state_space_models.state_space_models.lti import LTISystem, SpringMassDamper


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
    ):
        super().__init__()
        self.num_comp = num_comp
        self.num_data = num_data
        self.num_segment = num_segment
        self.dt = dt
        self.triangular = triangular
        self.use_B = use_B
        self.zero_means = zero_means
        self.use_C = use_C
        self.max_variability = max_variability
        self.system_type = system_type
        self.ar_order = ar_order

        # Remake label for TCL learning
        self.num_segmentdata = int(np.ceil(self.num_data / self.num_segment))

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
        else:
            raise ValueError(f"Unknown system type {system_type=}")

        self.segment_means, self.segment_variances = generate_segment_stats(
            self.num_comp,
            self.num_segment,
            zero_means=self.zero_means,
            max_variability=self.max_variability,
        )

        observations, sources = generate_nonstationary_data(
            self.lti,
            self.segment_means,
            self.segment_variances,
            self.num_segmentdata,
            self.dt,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        observations = observations.T.reshape(
            [-1, self.ar_order + 1, observations.shape[1]]
        )

        self.observations = torch.from_numpy(observations.astype(np.float32)).to(
            self.device
        )
        self.segment_variances = torch.from_numpy(
            self.segment_variances.astype(np.float32)
        ).to(self.device)
        self.segment_means = torch.from_numpy(self.segment_means.astype(np.float32)).to(
            self.device
        )

        self.sources = torch.from_numpy(sources.astype(np.float32)).to(self.device)

    @property
    def labels(self):
        return self.num_segmentdata

    @labels.setter
    def labels(self, num_segmentdata):
        self.num_segmentdata = num_segmentdata

    def __len__(self):
        return self.num_segment

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.segment_means[idx],
            self.segment_variances[idx],
            self.sources,
            self.labels[idx],
        )
