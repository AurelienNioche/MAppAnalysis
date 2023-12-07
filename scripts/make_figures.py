import os

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from datetime import datetime, time
import seaborn as sns
from tqdm import tqdm

from data.subjects_to_keep import subjects_to_keep as users
from data.data_path import data_path


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


def run_analysis(
    username,
    step_events,
    n_timesteps=20,
    n_velocity=20,
    check_sum_deriv=False,
    fig_folder="figures",
    density_plot_use_log_scale=True,
    plot_step_events_kde=False,
    plot_counts=True,
):
    fig_folder = f"{fig_folder}/make_figures/{n_timesteps}t_{n_velocity}v"
    os.makedirs(fig_folder, exist_ok=True)

    fig_id = 0

    n_days = len(step_events)

    # ------------------------------------------------------------------------
    # Figure 0 - Step events as vertical lines
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=n_days)
    for idx_day in range(n_days):
        ax = axes[idx_day]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if idx_day != n_days - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.vlines(step_events[idx_day], 0, 1, color="k", lw=0.1, alpha=0.02)

    plt.savefig(f"{fig_folder}/{username}_{fig_id:02d}_step_events.png", dpi=300)
    plt.close()
    fig_id += 1

    # -----------------------------------------------------------------------
    # Figure 0B - KDE of step events
    # -----------------------------------------------------------------------

    if plot_step_events_kde:
        fig, axes = plt.subplots(nrows=n_days, figsize=(6, n_days * 0.5))
        for idx_day in range(n_days):
            ax = axes[idx_day]
            sns.kdeplot(step_events[idx_day], ax=ax)
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 28)
            ax.set_ylabel("")
            ax.grid(False)
            if idx_day != n_days - 1:
                ax.set_xticks([])
        plt.savefig(
            f"{fig_folder}/{username}_{fig_id-1:02d}B_step_events_KDE.png", dpi=300
        )
        plt.close()

    # ------------------------------------------------------------------------
    # Compute cumulative step number and its derivative
    # ------------------------------------------------------------------------

    timestep = np.linspace(0, 1, n_timesteps)
    cum_steps = np.zeros((n_days, timestep.size))
    deriv_cum_steps = np.zeros((n_days, timestep.size))
    for idx_day in range(n_days):
        cum_steps[idx_day] = np.sum(step_events[idx_day] <= timestep[:, None], axis=1)
        deriv_cum_steps[idx_day] = (
            np.gradient(cum_steps[idx_day], timestep) / timestep.size
        )

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
    # Figure 1 - Cumulative step number
    # -------------------------------------------------------------
    fig, axes = plt.subplots(nrows=n_days, ncols=2, figsize=(6, n_days * 0.5))
    for idx_day in range(n_days):
        ax = axes[idx_day, 0]
        ax.plot(timestep, cum_steps[idx_day])
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])

        ax = axes[idx_day, 1]
        ax.plot(timestep, deriv_cum_steps[idx_day])
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])

    plt.savefig(f"{fig_folder}/{username}_{fig_id}_cumulative_step_number.png", dpi=300)
    plt.close()
    fig_id += 1

    # -------------------------------------------------------------
    # Figure 2 - Derivative of cumulative step number
    # -------------------------------------------------------------
    fig, axes = plt.subplots(nrows=n_days, figsize=(6, n_days * 0.5))
    for idx_day in range(n_days):
        ax = axes[idx_day]
        ax.plot(timestep, deriv_cum_steps[idx_day])
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])

    plt.savefig(
        f"{fig_folder}/{username}_{fig_id:02d}_derivative_cumulative_step_number.png",
        dpi=300,
    )
    plt.close()
    fig_id += 1

    # -------------------------------------------------------------
    # Figure 3 - KDE of the distribution of the timestamps and
    # derivative of cumulative step number
    # -------------------------------------------------------------

    fig, axes = plt.subplots(nrows=2)
    ax = axes[0]
    sns.kdeplot(np.concatenate(step_events), ax=ax)

    ax = axes[1]
    sns.kdeplot(deriv_cum_steps.flatten(), ax=ax)
    if density_plot_use_log_scale is True:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(
        f"{fig_folder}/{username}_{fig_id:02d}_timestamp_KDE_and_deriv_cum_step_number_KDE.png",
        dpi=300,
    )
    plt.close()
    fig_id += 1

    # -------------------------------------------------------------
    # Figure 4 - Transition matrix
    # -------------------------------------------------------------

    # max_vel = max([deriv.max() for deriv in deriv_list])
    # print("max velocity", max_vel)
    # min_vel = min([deriv.min() for deriv in deriv_list])
    # range_vel = max_vel - min_vel

    velocity = np.concatenate((np.zeros(1), np.logspace(-4, 4, n_velocity - 1)))
    alpha_tvv = np.zeros((timestep.size, velocity.size, velocity.size))

    bins = np.concatenate((velocity, np.full(1, np.inf)))

    for idx_day in range(n_days):
        drv = np.clip(deriv_cum_steps[idx_day], bins[0], bins[-1])
        v_idx = np.digitize(drv, bins, right=False) - 1
        for t in range(v_idx.size - 1):
            alpha_tvv[t, v_idx[t], v_idx[t + 1]] += 1

    # ---------------------------------------------------------------

    if plot_counts:
        fig, axes = plt.subplots(
            nrows=timestep.size - 1,
            figsize=(4, 3 * timestep.size),  # Exclude last timestep
        )
        for t_idx in range(timestep.size - 1):  # Exclude last timestep
            ax = axes[t_idx]
            img = alpha_tvv[t_idx, :, :]
            ax.imshow(img, aspect="auto", cmap="viridis")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        plt.savefig(
            f"{fig_folder}/{username}_{fig_id:02d}_transition_counts.png", dpi=300
        )
        plt.close()
        fig_id += 1

    # ----------------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=timestep.size - 1, figsize=(4, 3 * timestep.size)  # Exclude last timestep
    )
    p_tvv = normalize_last_dim(alpha_tvv)
    for t_idx in range(timestep.size - 1):  # Exclude last timestep
        ax = axes[t_idx]
        img = p_tvv[t_idx, :, :]
        ax.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    plt.savefig(f"{fig_folder}/{username}_{fig_id:02d}_transition_prob.png", dpi=300)
    plt.close()
    fig_id += 1


def main():
    for u in tqdm(users):
        step_events = load_data(u)
        run_analysis(u, step_events)


if __name__ == "__main__":
    main()
