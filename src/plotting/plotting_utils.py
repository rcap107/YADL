import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm, NoNorm


def get_labels_from_cat_codes(df: pd.DataFrame):
    labels_dict = dict(
        sorted(
            dict(
                zip(
                    df["x"].cat.codes,
                    df["x"],
                )
            ).items()
        )
    )
    return labels_dict


def get_numerical_coordinates(df: pd.DataFrame):
    df.columns = ["x", "y", "count"]
    df[["x", "y"]] = df[["x", "y"]].astype("category")
    df[["x_int", "y_int"]] = df[["x", "y"]].apply(lambda x: x.cat.codes)

    return df


def plot_pairwise_heatmap(df_src: pd.DataFrame, lognorm=True):
    df = get_numerical_coordinates(df_src.copy())

    max_category = df[["x_int", "y_int"]].max().max()

    # Set the proper coordinates
    zz = np.zeros((max_category + 1, max_category + 1))
    zz[
        df["x_int"],
        df["y_int"],
    ] = df["count"]
    # Set the proper labels
    dd = get_labels_from_cat_codes(df)

    if lognorm == True:
        norm = LogNorm()
    else:
        norm = NoNorm()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        zz,
        robust=True,
        cmap="rocket",
        norm=norm,
        xticklabels=list(dd.values()),
        yticklabels=list(dd.values()),
        square=True,
        ax=ax,
    )
    ax.set_title("Frequency of co-occurrence of predicate pairs in frequent values")

    return ax


def plot_predicate_frequency(df_cooc_predicates: pd.DataFrame, ax=None, label=None):
    if ax is not None:
        sns.lineplot(
            data=df_cooc_predicates,
            x=range(len(df_cooc_predicates)),
            y="count",
            label=label,
            ax=ax,
        )
    else:
        ax = sns.lineplot(
            data=df_cooc_predicates,
            x=range(len(df_cooc_predicates)),
            y="count",
            label=label,
        )

    ax.set_yscale("log")
    ax.set_title("Frequency of each pair of predicates. ")
    ax.set_xlabel("Rank of the predicate pair.")
    ax.set_ylabel("Number of occurrences of the predicate.")
    return ax


def plot_pairwise_relplot(df_count_cooccurring_predicates):
    df_count_cooccurring_predicates = get_numerical_coordinates(
        df_count_cooccurring_predicates
    )

    dd = get_labels_from_cat_codes(df_count_cooccurring_predicates)

    norm = LogNorm()
    g = sns.relplot(
        data=df_count_cooccurring_predicates,
        x="pred_left_int",
        y="pred_right_int",
        size="count",
        hue="count",
        hue_norm=norm,
        sizes=(10, 250),
        size_norm=norm,
        palette="vlag",
        height=7,
    )
    g.set(xticks=np.arange(len(dd)), yticks=np.arange(len(dd)))
    _ = g.set_xticklabels(
        list(dd.values()), rotation=45, horizontalalignment="right", fontsize=6
    )
    _ = g.set_yticklabels(list(dd.values()), horizontalalignment="right", fontsize=6)
    sns.despine(left=True, bottom=True)
