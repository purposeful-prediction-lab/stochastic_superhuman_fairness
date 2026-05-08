import numpy as np
import matplotlib.pyplot as plt

def _get_nested_mixed(d, path, default=None):
        if not isinstance(d, dict):
            return default
        if path in d:
            return d[path]

        parts = path.split("/")
        cur = d
        i = 0
        while i < len(parts):
            if not isinstance(cur, dict):
                return default

            found = False
            for j in range(len(parts), i, -1):
                key = "/".join(parts[i:j])
                if key in cur:
                    cur = cur[key]
                    i = j
                    found = True
                    break

            if not found:
                return default

        return cur



def plot_loss_and_subdom(
    logs,
    *,
    ax=None,
    title="Training Curves",
    xlabel="Epoch",
    loss_label="Loss",
    subdom_label="Mean Subdominance",
    intrademo_label="Intrademo loss",
    intrademo_scale=1.0,
    plot_freq=1,
    loss_color="tab:blue",
    intrademo_color="tab:green",
    subdom_color="tab:red",
    figsize=(8, 5),
    plot_loss_terms=True,
    loss_terms_path="train/l_terms/loss_terms",
    loss_terms_alpha=0.35,
    log_loss_scale=False,
    log_loss_base=10,
    log_loss_eps=1e-12,
):
    train_logs = [l for l in logs if l.get("stage") == "train"]
    if len(train_logs) == 0:
        raise ValueError("No train logs found.")

    epochs_all = np.array([l["epoch"] for l in train_logs])
    loss_all = np.array([l["train/loss"] for l in train_logs])
    mean_subdom_all = np.array([l["train/mean_subdom"] for l in train_logs])
    std_subdom_all = np.array([l["train/std_subdom"] for l in train_logs])

    idx = np.arange(0, len(epochs_all), plot_freq)

    epochs = epochs_all[idx]
    loss = loss_all[idx]
    mean_subdom = mean_subdom_all[idx]
    std_subdom = std_subdom_all[idx]
    train_logs_ds = [train_logs[i] for i in idx]

    # ---- intrademo
    intrademo_epochs, intrademo_loss = [], []

    for l in train_logs:
        baseline = l.get("train/demo_baseline_dict", None)
        if isinstance(baseline, dict) and baseline.get("loss", None) is not None:
            intrademo_epochs.append(l["epoch"])
            intrademo_loss.append(baseline["loss"] * intrademo_scale)

    if len(intrademo_epochs) > 0:
        intrademo_epochs = np.asarray(intrademo_epochs)
        intrademo_loss = np.asarray(intrademo_loss)

        if plot_freq > 1:
            intr_idx = np.arange(0, len(intrademo_epochs), plot_freq)
            intrademo_epochs = intrademo_epochs[intr_idx]
            intrademo_loss = intrademo_loss[intr_idx]

    has_intrademo = len(intrademo_loss) > 0

    # ---- loss terms
    loss_term_names = []
    loss_term_values = None

    if plot_loss_terms:
        term_lists = [
            _get_nested_mixed(l, loss_terms_path, default=None)
            for l in train_logs_ds
        ]
        # expected: list/tuple of scalar terms per epoch
        if any(isinstance(x, (list, tuple, np.ndarray)) for x in term_lists):
            max_terms = max(
                len(x) for x in term_lists
                if isinstance(x, (list, tuple, np.ndarray))
            )
            
            vals = []
            for x in term_lists:
                row = np.zeros(max_terms, dtype=float)
                if isinstance(x, (list, tuple, np.ndarray)):
                    arr = np.asarray(x, dtype=float).reshape(-1)
                    row[: len(arr)] = arr
                vals.append(row)

            loss_term_values = np.asarray(vals).T  # [num_terms, num_epochs]
            if max_terms == 2:
                loss_term_values = loss_term_values[::-1] # plot demos first so they are assgined blue color
                loss_term_names = ['demos_term_loss', 'rollout_term_loss']
            else:
                loss_term_names = [f"loss_term_{i}" for i in range(max_terms)]


    has_loss_terms = (
        plot_loss_terms
        and loss_term_values is not None
        and loss_term_values.size > 0
    )
    
    # TODO: pick a few demos  and one policy train on that see what happens
    # Why isnt the demo logprobs mathcing the matched demos still?
    # Do a few folds for each epoch with the same rollouts i.e mini epochs
    # Perhaps reduce the size of the sample size N
    # Does each demo match with the same policy's samples consistently? otheriwsie thats what we fail
    # Perhaps offpolicy additions with a replay buffer
    # Try to visualize the coupling over time. Somehow. Perhaps start with less policiies / demos? 
    # Email Jon Fyke for computation  
    # 2nd order  subdominance -> check for dominance over demonstration mixtures
    # Talk to Sayed for 2nd order stuff plus notes or maybe try this mathcing instead of our OT pairing



    # ---- plotting
    if ax is None:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    else:
        ax1 = ax
        fig = ax1.figure
    #  import ipdb;ipdb.set_trace()
    if has_loss_terms:
        ax1.stackplot(
            epochs,
            loss_term_values,
            labels=loss_term_names,
            alpha=loss_terms_alpha,
        )

    ax1.plot(
        epochs,
        loss,
        color=loss_color,
        linewidth=2.0,
        label=loss_label,
    )

    if has_intrademo:
        label = intrademo_label
        if intrademo_scale != 1.0:
            label = f"{intrademo_label} × {intrademo_scale:g}"

        ax1.plot(
            intrademo_epochs,
            intrademo_loss,
            color=intrademo_color,
            linestyle="--",
            label=label,
        )
    # ---- log scale (left axis)
    if log_loss_scale:
        # avoid log(0)
        if np.any(loss <= 0):
            loss_plot = np.clip(loss, log_loss_eps, None)
            # redraw safely if needed
            ax1.lines[-1].set_ydata(loss_plot)

    ax1.set_yscale("log", base=log_loss_base)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(loss_label, color=loss_color)
    ax1.tick_params(axis="y", labelcolor=loss_color)

    ax2 = ax1.twinx()
    ax2.plot(epochs, mean_subdom, color=subdom_color, label=subdom_label)

    ax2.fill_between(
        epochs,
        mean_subdom - std_subdom,
        mean_subdom + std_subdom,
        color=subdom_color,
        alpha=0.2,
    )

    ax2.set_ylabel(subdom_label, color=subdom_color)
    ax2.tick_params(axis="y", labelcolor=subdom_color)

    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    try:
        fig.tight_layout()
    except:
        import ipdb;ipdb.set_trace()
    return fig, ax1
