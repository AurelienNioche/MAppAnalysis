import matplotlib.pyplot as plt
import numpy as np


def runs(*args, figsize=(4, 3)):
    fig, axes = plt.subplots(nrows=2, figsize=figsize)
    for i, r in enumerate(args):
        policy = r["policy"]
        hist_pos = r["position"]
        hist_vel = r["velocity"]

        if len(hist_pos.shape) == 3:
            hist_pos = hist_pos[:, -1, :]   # Take the last learning episode only
            hist_vel = hist_vel[:, -1, :]   # Take the last learning episode only

        label = policy.replace("-", " ").capitalize()
        pos = hist_pos.mean(axis=0)
        pos_disp = hist_pos.std(axis=0)
        vel = hist_vel.mean(axis=0)
        vel_disp = hist_vel.std(axis=0)
        x = np.linspace(0, 1, len(pos))
        if label.startswith("Af"):
            label = label.replace("Af", "Active inference -")
            linewidth = 2
            if label.endswith("epistemic"):
                linestyle = ":"
                linewidth = 4
            elif label.endswith("pragmatic"):
                linestyle = "-."
            else:
                label = label.replace(" -", "")
                linestyle = "--"
        else:
            linestyle, linewidth = "-", 1
        axes[0].plot(x, pos, color=f"C{i}", label=label, linestyle=linestyle, linewidth=linewidth)
        axes[0].fill_between(x, pos - pos_disp, pos + pos_disp, alpha=0.1, color=f"C{i}")
        axes[1].plot(x, vel, color=f"C{i}", linestyle=linestyle, linewidth=linewidth)
        axes[1].fill_between(x, vel - vel_disp, vel + vel_disp, alpha=0.1, color=f"C{i}")
        axes[0].set_ylabel("position")
        axes[1].set_ylabel("velocity")
        axes[1].set_xlabel("time")

    fig.legend(loc=[0.05, 0.05], fontsize=5)
    fig.tight_layout()

    plt.show()


def error(*args):

    fig, ax = plt.subplots(figsize=(3, 3))
    for i, r in enumerate(args):
        policy = r["policy"]
        hist_err = r["error"]
        label = policy.replace("-", " ").capitalize()
        hist_err_mean = hist_err.mean(axis=0)
        x = np.arange(len(hist_err_mean))
        hist_err_std = hist_err.std(axis=0)

        if label.startswith("Af"):
            label = label.replace("Af", "Active inference -")
            linestyle, linewidth = "-", 2
        else:
            linestyle, linewidth = "-", 1
        ax.plot(x, hist_err_mean, color=f"C{i}", label=label, linestyle=linestyle, linewidth=linewidth)
        ax.fill_between(x, hist_err_mean - hist_err_std, hist_err_mean + hist_err_std, alpha=0.1, color=f"C{i}")
        ax.set_ylabel("error")
        ax.set_xlabel("epoch")

    fig.legend(loc="center")
    plt.tight_layout()
    plt.show()


def q(alpha, title=r"$\alpha$", figsize=(6, 2)):
    plt.rcParams.update({'text.usetex': True})
    if len(alpha.shape) == 4:
        n_timestep, n_action, n_velocity, _ = alpha.shape
        fig, axes = plt.subplots(
            ncols=n_timestep,
            nrows=n_action,
            figsize=(6, 2))
        fig.suptitle(title)
        for a_idx in range(n_action):
            for t_idx in range(n_timestep):
                ax = axes[a_idx, t_idx]
                img = alpha[t_idx, a_idx, :, :]
                ax.imshow(img, aspect="auto")
                ax.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])
        plt.tight_layout()
    elif len(alpha.shape) == 5:
        n_action, n_timestep, n_position, n_velocity, _ = alpha.shape
        for a_idx in range(n_action):
            fig, axes = plt.subplots(
                nrows=n_position,
                ncols=n_timestep,
                figsize=figsize)
            fig.suptitle(f"{title} - Action {a_idx} (row: pos., col: timestep)", fontsize=8)
            for t_idx in range(n_timestep):
                for p_idx in range(n_position):
                    ax = axes[p_idx, t_idx]
                    img = alpha[a_idx, t_idx, p_idx, :, :]
                    ax.imshow(img, aspect="auto")
                    ax.get_xaxis().set_ticks([])
                    ax.axes.get_yaxis().set_ticks([])
            plt.show()
    else:
        raise ValueError
