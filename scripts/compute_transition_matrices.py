import os

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from datetime import datetime, time
from tqdm import tqdm

from data.subjects_to_keep import subjects_to_keep as users
from data.data_path import data_path


def plot_transition_matrix(timestep, transition_matrix, fig_name):
    fig, axes = plt.subplots(
        nrows=timestep.size - 1, figsize=(4, 3 * timestep.size)  # Exclude last timestep
    )
    for t_idx in range(timestep.size - 1):  # Exclude last timestep
        ax = axes[t_idx]
        img = transition_matrix[t_idx, :, :]
        ax.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()


def normalize_last_dim(alpha):
    sum_col = np.sum(alpha, axis=-1)
    sum_col[sum_col <= 0.0] = 1
    return alpha / np.expand_dims(sum_col, axis=-1)


def load_data(user):
    file = glob(f"{data_path}/dump_latest/{user}_activity*.csv")[0]

    df = pd.read_csv(file, index_col=0)
    df.dt = pd.to_datetime(df.dt, utc=False, format="ISO8601")
    df.dt = df.dt.dt.tz_convert("Europe/London")

    all_pos = df.step_midnight.values

    min_date = df.dt.min().date()
    days = np.asarray([(dt.date() - min_date).days for dt in df.dt])
    uniq_days = np.unique(days)
    all_timestamp = (
        np.asarray(
            [
                (dt - datetime.combine(dt, time.min, dt.tz)).total_seconds()
                for dt in df.dt
            ]
        )
        / 86400
    )  # in fraction of day (between 0 and 1)

    # List of step events for each day, the event itself being the timestamp of the step
    step_events = [[] for _ in range(uniq_days.size)]

    for idx_day, day in enumerate(uniq_days):
        is_day = days == day
        obs_timestamp, obs_pos = all_timestamp[is_day], all_pos[is_day]

        # Sort the data by timestamp
        idx = np.argsort(obs_timestamp)
        obs_timestamp, obs_pos = obs_timestamp[idx], obs_pos[idx]

        # Compute the number of steps between each observed timestamp
        diff_obs_pos = np.diff(obs_pos)

        for ts, diff in zip(obs_timestamp, diff_obs_pos):
            # TODO: In the future, we probably want to spread that
            #  over a period assuming something like 6000 steps per hour
            step_events[idx_day] += [ts for _ in range(diff)]

    return step_events


def run_analysis(step_events, n_timesteps, n_velocity, check_sum_deriv=False):
    n_days = len(step_events)

    # ------------------------------------------------------------------------
    # Compute cumulative step number and its derivative
    # ------------------------------------------------------------------------

    timestep = np.linspace(0, 1, n_timesteps)
    deriv_cum_steps = np.zeros((n_days, timestep.size))
    for idx_day in range(n_days):
        cum_steps_day = np.sum(step_events[idx_day] <= timestep[:, None], axis=1)
        deriv_cum_steps[idx_day] = np.gradient(cum_steps_day, timestep) / timestep.size

    # --------------------------------------------------------------------------
    # Check the sum of the derivative is (roughly) equal to the number of steps
    # --------------------------------------------------------------------------

    if check_sum_deriv:
        for idx_day in range(n_days):
            print(
                f"sum deriv {deriv_cum_steps[idx_day].sum():.2f}, "
                f"actual step number {len(step_events[idx_day])}"
            )

    # -------------------------------------------------------------
    # Figure 4 - Transition matrix
    # -------------------------------------------------------------

    velocity = np.concatenate((np.zeros(1), np.logspace(-4, 4, n_velocity - 1)))
    alpha_tvv = np.zeros((timestep.size, velocity.size, velocity.size))

    bins = np.concatenate((velocity, np.full(1, np.inf)))

    for idx_day in range(n_days):
        drv = np.clip(deriv_cum_steps[idx_day], bins[0], bins[-1])
        v_idx = np.digitize(drv, bins, right=False) - 1
        for t in range(v_idx.size - 1):
            alpha_tvv[t, v_idx[t], v_idx[t + 1]] += 1

    p_tvv = normalize_last_dim(alpha_tvv)

    return (
        timestep,
        velocity,
        deriv_cum_steps,
        alpha_tvv[:-1, :, :],  # remove last timestamp
        p_tvv[:-1, :, :],  # remove last timestamp
    )


def main():
    n_timestep = 20
    n_velocity = 20

    bkp_folder = "../bkp/compute_transition_matrices"
    fig_folder = "../figures/compute_transition_matrices"

    # For each user
    i = 0
    for u in tqdm(users):
        print(f"user {u} ID={i}" + "-" * 10)
        # Load data
        step_events = load_data(u)

        # Run analysis
        timestep, velocity, deriv_cum_steps, alpha_tvv, p_tvv = run_analysis(
            step_events=step_events,
            n_timesteps=n_timestep,
            n_velocity=n_velocity,
            check_sum_deriv=False,
        )

        sum_steps = np.sum(deriv_cum_steps, axis=1)
        print(f"{np.max(sum_steps):.02f} +/= {np.std(sum_steps):.02f}")

        # Save results -------------------------------------------------
        # Create folders
        user_bkp_folder = f"{bkp_folder}/{u}__{timestep.size}t_{velocity.size}v"
        os.makedirs(user_bkp_folder, exist_ok=True)
        # Save files
        np.save(f"{user_bkp_folder}/timestep.npy", timestep)
        np.save(f"{user_bkp_folder}/velocity.npy", velocity)
        np.save(f"{user_bkp_folder}/deriv_cum_steps.npy", deriv_cum_steps)
        np.save(f"{user_bkp_folder}/alpha_tvv.npy", alpha_tvv)
        np.save(f"{user_bkp_folder}/p_tvv.npy", alpha_tvv)

        # ---------------------------------------------------------------
        # Plot transition matrix
        user_fig_folder = f"{fig_folder}/{u}__{timestep.size}t_{velocity.size}v"
        os.makedirs(user_fig_folder, exist_ok=True)
        fig_name = f"{user_fig_folder}/transition_matrix.png"
        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        plot_transition_matrix(
            timestep=timestep, transition_matrix=p_tvv, fig_name=fig_name
        )

        i += 1


if __name__ == "__main__":
    main()
