import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from datetime import datetime, time
import seaborn as sns
from data.subjects_to_keep import subjects_to_keep as users
from data.data_path import data_path


def load_data(user):

    file = glob(f"{data_path}/dump_latest/{user}_activity*.csv")[0]

    df = pd.read_csv(file, index_col=0)
    df.dt = pd.to_datetime(df.dt, utc=False, format='ISO8601')
    df.dt = df.dt.dt.tz_convert('Europe/London')

    all_pos = df.step_midnight.values

    min_date = df.dt.min().date()
    days = np.asarray([(dt.date() - min_date).days for dt in df.dt])
    uniq_days = np.unique(days)
    all_timestep = (np.asarray([(dt - datetime.combine(dt, time.min, dt.tz)).total_seconds() for dt in df.dt]) / 86400)  # in fraction of day (between 0 and 1)

    # List of step events for each day, the event itself being the timestamp of the step
    step_events = [[] for _ in range(uniq_days.size)]

    for idx_day, day in enumerate(uniq_days):

        is_day = days == day
        obs_timestep, obs_pos = all_timestep[is_day], all_pos[is_day]

        diff_obs_pos = np.diff(obs_pos)

        # Make sure the data is sorted
        assert np.all(np.arange(obs_timestep.size) == np.argsort(obs_timestep)), "Not sorted"

        for ts, diff in zip(obs_timestep, diff_obs_pos):

            # TODO: In the future, we probably want to spread that
            #  over a period assuming something like 6000 steps per hour
            step_events[idx_day] += [ts for _ in range(diff)]

    return step_events


def run_analysis(username, step_events):

    n_days = len(step_events)

    # Figure 1 - Vlines of step events
    fig, axes = plt.subplots(nrows=n_days)
    for idx_day in range(n_days):
        ax = axes[idx_day]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if idx_day != n_days - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        # for step in step_events[idx_day]:
        ax.vlines(step_events[idx_day], 0, 1, color="k", lw=0.1, alpha=0.02)

    plt.savefig(f"{username}_01_step_events_vlines.png", dpi=300)
    plt.close()

    # Figure 2 - KDE of step events
    fig, axes = plt.subplots(nrows=n_days, figsize=(6, n_days*0.5))
    for idx_day in range(n_days):
        ax = axes[idx_day]
        sns.kdeplot(step_events[idx_day], ax=ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 28)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])
    plt.savefig(f"{username}_02_step_events_KDE.png", dpi=300)
    plt.close()

    # Figure 3 - Cumulative step number
    timestep = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(nrows=n_days, figsize=(6, n_days * 0.5))
    for idx_day in range(n_days):
        ax = axes[idx_day]
        a = [idx_day]
        cum_x = np.sum(a <= timestep[:, None], axis=1)
        ax.plot(timestep, cum_x)
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])

    plt.savefig(f"{username}_03_cumulative_step_number.png", dpi=300)
    plt.close()

    # Figure 4 - Derivative of cumulative step number
    timestep = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(nrows=n_days, figsize=(6, n_days * 0.5))
    for idx_day in range(n_days):
        ax = axes[idx_day]
        step_day = step_events[idx_day]
        cum_x = np.sum(step_day <= timestep[:, None], axis=1)
        deriv = np.gradient(cum_x, timestep) / timestep.size

        ax.plot(timestep, deriv)
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        ax.grid(False)
        if idx_day != n_days - 1:
            ax.set_xticks([])

    # Check the sum of the derivative is (roughly) equal to the number of steps
    timestep = np.linspace(0, 1, 100)
    for idx_day in range(n_days):
        step_day = step_events[idx_day]
        cum_x = np.sum(step_day <= timestep[:, None], axis=1)
        deriv = np.gradient(cum_x, timestep) / timestep.size
        print(f"sum deriv {deriv.sum():.2f}, actual step number {len(a)}")

    timestep = np.linspace(0, 1, 10)

    deriv_list = []
    n_days = len(all_x)

    for idx_day in range(n_days):
        x = all_x[idx_day]
        cum_x = np.sum(x <= timestep[:, None], axis=1)
        deriv = np.gradient(cum_x, timestep) / timestep.size
        deriv_list.append(deriv)
    #%%
    fig, ax = plt.subplots()
    sns.kdeplot(np.concatenate(all_x), ax=ax)
    #%%
    fig, ax = plt.subplots()
    sns.kdeplot(np.concatenate(deriv_list), ax=ax)
    #ax.set_xscale("log")
    #%%
    all_deriv = np.concatenate(deriv_list)
    fig, ax = plt.subplots()
    sns.kdeplot(all_deriv, ax=ax)
    ax.set_xscale("log")
    #%%
    #max_vel = max([deriv.max() for deriv in deriv_list])
    # print("max velocity", max_vel)
    # min_vel = min([deriv.min() for deriv in deriv_list])
    # range_vel = max_vel - min_vel

    velocity = np.concatenate((np.zeros(1), np.logspace(-4, 4, 10)))
    alpha_tvv = np.zeros((timestep.size, velocity.size, velocity.size))

    bins = np.concatenate((velocity, np.full(1, np.inf)))

    # print("bins", bins)

    x = np.linspace(0, 1, 100)
    for idx_day in range(n_days):

        deriv = deriv_list[idx_day]
        deriv = np.clip(deriv, bins[0], bins[-1])
        v_indexes = np.digitize(deriv, bins, right=False) - 1

        n = len(v_indexes) - 1

        # density = hist / np.sum(hist)
        for t in range(n):
            # p = p_indexes[i]
            v = v_indexes[t]
            v_prime = v_indexes[t + 1]
            alpha_tvv[t, v, v_prime] += 1
    #%%
    fig, ax = plt.subplots()
    img = alpha_tvv[0, :, :]
    ax.imshow(img, aspect="auto", cmap="viridis")
    ax.grid(False)
    #%%
    fig, axes = plt.subplots(
                    nrows=timestep.size-1,  # Exclude last timestep
                    figsize=(4, 3*timestep.size))
    for t_idx in range(timestep.size-1):    # Exclude last timestep
        # for p_idx in range(n_position):
        ax = axes[t_idx]
        img = alpha_tvv[t_idx, :, :]
        ax.imshow(img, aspect="auto", cmap="viridis")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.show()

def main():

    for u in users:



if __name__ == "__main__":
    main()