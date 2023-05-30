import numpy as np


def generate_segment_stats(num_comp, num_segment, zero_means=False,  max_variability=False):
    rank = 0
    num_attempt = 0

    while rank < num_comp:

        segment_variances = np.abs(np.random.randn(num_segment, num_comp)) / 2

        if max_variability is True:
            if num_segment == num_comp+1:
                print("Constructing maximally variable system")
                segment_variances = np.ones((num_segment, num_comp))*0.0001
                segment_variances[1:, :] = segment_variances[1:, :] +np.eye(num_comp)*0.9999
            else:
                raise ValueError(f"Cannot construct maximally variable system for this number of segments, num_segment == num_comp+1 should hold, got {num_segment=}, {num_comp=}")


        print(f"{segment_variances=}")
        segment_means = np.random.randn(num_segment, num_comp) if zero_means is True else np.zeros((num_segment, num_comp))

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
    for i, (segment_mean, segment_var) in enumerate(zip(segment_means,segment_variances)):

        segment_U = np.random.multivariate_normal(segment_mean, np.diag(segment_var), num_segmentdata)

        _, segment_obs, segment_state = lti.simulate(segment_U, dt=dt)

        obs.append(segment_obs)
        states.append(segment_state)
    obs = np.concatenate(obs, axis=0)
    states = np.concatenate(states, axis=0)
    x = obs.T
    s = states.T

    return x, s
