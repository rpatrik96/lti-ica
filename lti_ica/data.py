import numpy as np


def generate_segment_stats(num_comp, num_segment):
    rank = 0
    num_attempt = 0

    while rank < num_comp:
        segment_variances = np.abs(np.random.randn(num_segment, num_comp)) / 2

        print(f"{segment_variances=}")
        # segment_means = np.random.randn(num_segment, num_comp)
        segment_means = np.zeros((num_segment, num_comp))

        # check sufficient variability of the variances
        base_prec = 1. / segment_variances[0, :]
        delta_prec = 1. / segment_variances[1:, :] - base_prec
        if num_segment == 1:
            break
        rank = np.linalg.matrix_rank(delta_prec)

        print(f"rank: {rank}")
        print(f"Condition number: {np.linalg.cond(delta_prec)}")

        num_attempt += 1
        if num_attempt > 100:
            raise ValueError("Could not find sufficiently variable system!")

    return segment_means, segment_variances


def generate_nonstationary_data(lti, segment_means, segment_variances, num_comp, num_segmentdata, dt):
    # iterate over the segment variances,
    # generate multivariate normal with each variance,
    # and simulate it with the LTI system
    obs = []
    states = []
    for i in enumerate(segment_variances):
        segment_var = segment_variances[i, :]
        segment_cov = np.diag(segment_var)

        # todo: change the means to be non-zero
        segment_mean = np.zeros(num_comp)

        segment_U = np.random.multivariate_normal(segment_mean, segment_cov, num_segmentdata)

        _, segment_obs, segment_state = lti.simulate(segment_U, dt=dt)

        obs.append(segment_obs)
        states.append(segment_state)
    obs = np.concatenate(obs, axis=0)
    states = np.concatenate(states, axis=0)
    x = obs.T
    s = states.T

    return x, s
