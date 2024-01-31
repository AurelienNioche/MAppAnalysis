import numpy as np
from scipy.stats import norm
from scipy.special import softmax
from .helpers import square_exponential_kernel, normalize_last_dim

np.random.seed(123)

n_timestep = 10
n_velocity = 20
n_action = 2
n_position = 50
min_position, max_position = 0.0, 4.0
min_velocity, max_velocity = -2.0, 4.0
min_timestep, max_timestep = 0.0, 1.0

timestep = np.linspace(min_timestep, max_timestep, n_timestep)

velocity = np.linspace(min_velocity, max_velocity, n_velocity)
action = np.arange(n_action)
position = np.linspace(min_position, max_position, n_position)

friction_factor = 0.5

mu = 0.5 + 0.5 * np.cos(6 * (timestep + 5))
sigma = square_exponential_kernel(timestep, 0.05, 0.1)
own_force = np.random.multivariate_normal(mu, sigma, size=300)

mu = 0.4 + 2 * np.cos(3 * (timestep - 2))
sigma = square_exponential_kernel(timestep, 0.05, 0.1)
push_effect = np.random.multivariate_normal(mu, sigma, size=300)

sigma_transition_position = 0.05

# Compute preferences ------------------------------------------------------------------------------------

log_prior = np.log(softmax(np.arange(n_position)))

# Compute velocity transitions --------------------------------------------------------------------------


def build_transition_velocity_atvv():
    tr = np.zeros((n_action, n_timestep, n_velocity, n_velocity))

    bins = list(velocity) + [velocity[-1] + (velocity[-1] - velocity[-2])]

    after_friction = velocity - friction_factor * velocity  # shape = (n_velocity,)
    after_friction = np.expand_dims(
        after_friction, (0, 1, 2)
    )  # shape = (1, 1, 1, n_velocity) as we broadcast over t, a, p

    push__ext = np.expand_dims(
        push_effect, (0, -1)
    )  # shape = (1, n_timestep, n_velocity, 1)
    action_effect = np.vstack(
        (np.zeros_like(push__ext), push__ext)
    )  # shape = (n_action, n_timestep, n_velocity, 1)

    own_force__ext = np.expand_dims(
        own_force, (0, -1)
    )  # shape = (1, n_timestep, n_velocity, 1)

    new_v = (
        after_friction + action_effect + own_force__ext
    )  # shape = (n_action, n_timestep, n_velocity, n_velocity)
    new_v = np.clip(new_v, bins[0], bins[-1])

    for v_idx, v in enumerate(velocity):
        for a_idx, a in enumerate(action):
            for t_idx, t in enumerate(timestep):
                tr[a, t_idx, v_idx, :], _ = np.histogram(
                    new_v[a, :, t_idx, v_idx], bins=bins
                )
    return normalize_last_dim(tr)


transition_velocity_atvv = build_transition_velocity_atvv()

# Compute position transitions --------------------------------------------------------------------------


def build_transition_position_pvp():
    tr = np.zeros((n_position, n_velocity, n_position))
    for p_idx, p in enumerate(position):
        for v_idx, v in enumerate(velocity):
            tr[p_idx, v_idx, :] = norm.pdf(
                position, loc=p + (1 / n_timestep) * v, scale=sigma_transition_position
            )
    return normalize_last_dim(tr)


transition_position_pvp = build_transition_position_pvp()
