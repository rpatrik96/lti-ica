import numpy as np


def generate_segment_stats(
    num_comp, num_segment, zero_means=False, max_variability=False, control_dim=None
):
    rank = 0
    num_attempt = 0

    if control_dim is None:
        control_dim = num_comp

    while rank < control_dim:
        segment_variances = np.abs(np.random.randn(num_segment, control_dim)) / 2

        if max_variability is True:
            if num_segment == num_comp + 1:
                print("Constructing maximally variable system")
                segment_variances = np.ones((num_segment, control_dim)) * 0.0001
                segment_variances[1:, :] = (
                    segment_variances[1:, :] + np.eye(control_dim) * 0.9999
                )
            else:
                raise ValueError(
                    f"Cannot construct maximally variable system for this number of segments, num_segment == num_comp+1 should hold, got {num_segment=}, {num_comp=}"
                )

        print(f"{segment_variances=}")
        segment_means = (
            np.random.randn(num_segment, control_dim)
            if zero_means is False
            else np.zeros((num_segment, control_dim))
        )

        # check sufficient variability of the variances
        base_prec = 1.0 / segment_variances[0, :]
        delta_prec = 1.0 / segment_variances[1:, :] - base_prec
        if num_segment == 1:
            break
        rank = np.linalg.matrix_rank(delta_prec)

        print(f"rank: {rank}")
        print(f"Condition number: {np.linalg.cond(delta_prec)}")

        num_attempt += 1
        if num_attempt > 100:
            raise ValueError("Could not find sufficiently variable system!")

    return segment_means, segment_variances


def generate_nonstationary_data(
    lti, segment_means, segment_variances, num_data_per_segment, dt, obs_noise_var=0
):
    # iterate over the segment variances,
    # generate multivariate normal with each variance,
    # and simulate it with the LTI system
    obs = []
    states = []
    controls = []
    for i, (segment_mean, segment_var) in enumerate(
        zip(segment_means, segment_variances)
    ):
        segment_U = np.random.multivariate_normal(
            segment_mean, np.diag(segment_var), num_data_per_segment
        )

        _, segment_obs, segment_state = lti.simulate(segment_U, dt=dt)

        assert np.all(np.isfinite(segment_obs))
        assert np.all(np.isfinite(segment_state))

        if obs_noise_var != 0:
            segment_obs += np.random.multivariate_normal(
                np.zeros(obs_dim := segment_obs.shape[1]),
                np.sqrt(obs_noise_var) * np.eye(obs_dim),
                len(segment_obs),
            )

        obs.append(segment_obs)
        states.append(segment_state)
        controls.append(segment_U)

    obs = np.concatenate(obs, axis=0)
    states = np.concatenate(states, axis=0)
    controls = np.concatenate(controls, axis=0)

    return obs, states, controls
