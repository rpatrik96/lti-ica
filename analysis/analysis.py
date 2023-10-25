from os.path import isfile

import numpy as np
import pandas as pd

BLUE = "#1A85FF"
RED = "#D0021B"
METRIC_EPS = 1e-6

from matplotlib import rc


def plot_typography(
    usetex: bool = False, small: int = 16, medium: int = 20, big: int = 22
):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).
    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc("text", usetex=usetex)
    rc("font", family="serif")

    # font sizes
    rc("font", size=small)  # controls default text sizes
    rc("axes", titlesize=big)  # fontsize of the axes title
    rc("axes", labelsize=medium)  # fontsize of the x and y labels
    rc("xtick", labelsize=small)  # fontsize of the tick labels
    rc("ytick", labelsize=small)  # fontsize of the tick labels
    rc("legend", fontsize=small)  # legend fontsize
    rc("figure", titlesize=big)  # fontsize of the figure title


def sweep2df(
    sweep_runs,
    filename,
    save=False,
    load=False,
):
    csv_name = f"{filename}.csv"
    npy_name = f"{filename}.npz"

    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        npy_data = np.load(npy_name)
        train_log_likelihood_histories = npy_data["train_log_likelihood_history"]
        train_mcc_histories = npy_data["train_mcc_history"]

        val_log_likelihood_histories = npy_data["val_log_likelihood_history"]
        val_mcc_histories = npy_data["val_mcc_history"]

        return (
            pd.read_csv(csv_name),
            train_log_likelihood_histories,
            train_mcc_histories,
            val_log_likelihood_histories,
            val_mcc_histories,
        )

    data = []
    val_log_likelihood_histories = []
    val_mcc_histories = []
    train_log_likelihood_histories = []
    train_mcc_histories = []
    for run in sweep_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            # print(f"\t Processing {run.name}...")
            # try:
            if True:
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                num_comp = config["data.num_comp"]
                num_segment = config["data.num_comp"]
                use_B = config["data.use_B"]
                use_C = config["data.use_C"]
                try:
                    max_variability = config["data.max_variability"]
                except:
                    max_variability = False

                zero_means = config["data.zero_means"]
                seed_everything = config["seed_everything"]

                try:
                    train_log_likelihood = summary["train_log_likelihood"]
                    train_mcc = summary["train_mcc"]

                    val_log_likelihood = summary["val_log_likelihood"]
                    val_mcc = summary["val_mcc"]
                except:
                    print(f"Encountered a faulty run with ID {run.name}")
                    continue

                train_log_likelihood_history = run.history(
                    keys=[f"train_log_likelihood"]
                )
                max_train_log_likelihood_step, max_train_log_likelihood = (
                    train_log_likelihood_history.idxmax()[1],
                    train_log_likelihood_history.max()[1],
                )
                train_log_likelihood_histories.append(
                    train_log_likelihood_history["train_log_likelihood"]
                )

                train_mcc_history = run.history(keys=[f"train_mcc"])
                max_train_mcc_step, max_train_mcc = (
                    train_mcc_history.idxmax()[1],
                    train_mcc_history.max()[1],
                )
                train_mcc_histories.append(train_mcc_history["train_mcc"])

                val_log_likelihood_history = run.history(keys=[f"val_log_likelihood"])
                max_val_log_likelihood_step, max_val_log_likelihood = (
                    val_log_likelihood_history.idxmax()[1],
                    val_log_likelihood_history.max()[1],
                )
                val_log_likelihood_histories.append(
                    val_log_likelihood_history["val_log_likelihood"]
                )

                val_mcc_history = run.history(keys=[f"val_mcc"])
                max_val_mcc_step, max_val_mcc = (
                    val_mcc_history.idxmax()[1],
                    val_mcc_history.max()[1],
                )
                val_mcc_histories.append(val_mcc_history["val_mcc"])

                data.append(
                    [
                        run.name,
                        seed_everything,
                        train_log_likelihood,
                        max_train_log_likelihood,
                        train_mcc,
                        max_train_mcc,
                        val_log_likelihood,
                        max_val_log_likelihood,
                        val_mcc,
                        max_val_mcc,
                        num_comp,
                        num_segment,
                        use_B,
                        use_C,
                        max_variability,
                        zero_means,
                    ]
                )

            # except:
            #     print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "seed_everything",
            "train_log_likelihood",
            "max_train_log_likelihood",
            "train_mcc",
            "max_train_mcc",
            "val_log_likelihood",
            "max_val_log_likelihood",
            "val_mcc",
            "max_val_mcc",
            "num_comp",
            "num_segment",
            "use_B",
            "use_C",
            "max_variability",
            "zero_means",
        ],
    ).fillna(0)

    # Prune histories to the minimum length
    # min_len = np.array([len(v) for v in val_log_likelihood_histories]).min()

    # val_log_likelihood_histories = np.array([v[:min_len] for v in val_log_likelihood_histories])

    if save is True:
        runs_df.to_csv(csv_name)
        np.savez_compressed(
            npy_name,
            val_log_likelihood_history=val_log_likelihood_histories,
            val_mcc_history=val_mcc_histories,
            train_log_likelihood_history=train_log_likelihood_histories,
            train_mcc_history=train_mcc_histories,
        )

    return (
        runs_df,
        train_log_likelihood_histories,
        train_mcc_histories,
        val_log_likelihood_histories,
        val_mcc_histories,
    )


def stats2string(df):
    s = [
        f"${m:.3f}\scriptscriptstyle\pm {s:.3f}$ & "
        for m, s in zip(df.mean().train_mcc, df.std().train_mcc)
    ]
    return "".join(s)
