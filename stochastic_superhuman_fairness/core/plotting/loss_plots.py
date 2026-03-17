import numpy as np
import matplotlib.pyplot as plt


def plot_loss_and_subdom(
    logs,
    *,
    title="Training Curves",
    xlabel="Epoch",
    loss_label="Loss",
    subdom_label="Mean Subdominance",
    loss_color="tab:blue",
    subdom_color="tab:orange",
    figsize=(8, 5),
):
    """
    Plot training loss and mean subdominance (+/- std) on dual y-axes.

    Args:
        logs: list of log dicts
        title: plot title
        xlabel: x-axis label
        loss_label: left y-axis label
        subdom_label: right y-axis label
        loss_color: color for loss curve
        subdom_color: color for subdominance curve
        figsize: figure size
    """

    # -----------------------------
    # Filter train logs
    # -----------------------------
    train_logs = [l for l in logs if l.get("stage") == "train"]

    if len(train_logs) == 0:
        raise ValueError("No train logs found.")

    epochs = np.array([l["epoch"] for l in train_logs])
    loss = np.array([l["train/loss"] for l in train_logs])
    mean_subdom = np.array([l["train/mean_subdom"] for l in train_logs])
    std_subdom = np.array([l["train/std_subdom"] for l in train_logs])

    # -----------------------------
    # Create figure
    # -----------------------------
    fig, ax1 = plt.subplots(figsize=figsize)

    # ----- Loss (left axis)
    ax1.plot(epochs, loss, color=loss_color, label=loss_label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(loss_label, color=loss_color)
    ax1.tick_params(axis="y", labelcolor=loss_color)

    # ----- Subdominance (right axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, mean_subdom, color=subdom_color, label=subdom_label)

    # Shaded std band
    ax2.fill_between(
        epochs,
        mean_subdom - std_subdom,
        mean_subdom + std_subdom,
        color=subdom_color,
        alpha=0.2,
    )

    ax2.set_ylabel(subdom_label, color=subdom_color)
    ax2.tick_params(axis="y", labelcolor=subdom_color)

    # -----------------------------
    # Title
    # -----------------------------
    plt.title(title)

    # Optional combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    return fig, ax1
