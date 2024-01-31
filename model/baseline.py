import numpy as np


def run(
    transition_velocity_atvv,
    transition_position_pvp,
    timestep,
    position,
    velocity,
    n_sample_run=20,
):
    n_timestep, n_velocity, n_position = timestep.size, velocity.size, position.size

    runs = []

    for policy in "all-one", "all-zero", "random", "max-expected-velocity":
        hist_pos = np.zeros((n_sample_run, n_timestep - 1))
        hist_vel = np.zeros_like(hist_pos)

        for sample in range(n_sample_run):
            pos_idx = np.absolute(position).argmin()  # Something close to 0
            v_idx = np.absolute(velocity).argmin()  # Something close to 0

            np.random.seed(123 + sample * 123)

            for t_idx in range(timestep.size - 1):
                if policy == "all-one":
                    a = 1
                elif policy == "all-zero":
                    a = 0
                elif policy == "random":
                    a = np.random.choice([0, 1])
                elif policy == "max-expected-velocity":
                    e_v = np.zeros(2)
                    for a in range(2):
                        e_v[a] = np.average(
                            velocity,
                            weights=transition_velocity_atvv[a, t_idx, v_idx, :],
                        )
                    a = e_v.argmax()
                else:
                    raise ValueError

                v_idx = np.random.choice(
                    np.arange(n_velocity),
                    p=transition_velocity_atvv[a, t_idx, v_idx, :],
                )  # = np.average(velocity, weights=p_)
                pos_idx = np.random.choice(
                    np.arange(n_position), p=transition_position_pvp[pos_idx, v_idx, :]
                )

                hist_pos[sample, t_idx] = position[pos_idx]
                hist_vel[sample, t_idx] = velocity[v_idx]

        runs.append({"policy": policy, "position": hist_pos, "velocity": hist_vel})
    return runs
