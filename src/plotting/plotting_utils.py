import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def get_labels_from_cat_codes(df: pd.DataFrame):
    labels_dict = dict(
    sorted(
        dict(
            zip(
                df["pred_left"].cat.codes,
                df["pred_left"],
            )
        ).items()
    )
    )
    return labels_dict
    
    
def get_numerical_coordinates(df:pd.DataFrame):
    df.columns = [
        "pred_left",
        "pred_right",
        "count"
    ]
    df[
        ["pred_left", "pred_right"]
    ] = df[["pred_left", "pred_right"]].astype(
        "category"
    )
    df[
        ["pred_left_int", "pred_right_int"]
    ] = df[["pred_left", "pred_right"]].apply(
        lambda x: x.cat.codes
    )
    
    return df


def plot_pairwise_heatmap(df_count_cooccurring_predicates: pd.DataFrame):
    df_count_cooccurring_predicates = get_numerical_coordinates(df_count_cooccurring_predicates)
    
    max_category = (
        df_count_cooccurring_predicates[["pred_left_int", "pred_right_int"]]
        .max()
        .max()
    )
    
    # Set the proper coordinates
    zz = np.zeros((max_category + 1, max_category + 1))
    zz[
        df_count_cooccurring_predicates["pred_left_int"],
        df_count_cooccurring_predicates["pred_right_int"],
    ] = df_count_cooccurring_predicates["count"]
    # Set the proper labels
    dd = get_labels_from_cat_codes(df_count_cooccurring_predicates)
    
    norm = LogNorm()
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(
        zz,
        robust=True,
        cmap="rocket",
        norm=norm,
        xticklabels=list(dd.values()),
        yticklabels=list(dd.values()),
        square=True,
        ax=ax
    )
    ax.set_title("Frequency of co-occurrence of predicate pairs in frequent values")

    return ax

def plot_predicate_frequency(df_cooc_predicates: pd.DataFrame, ax=None, label=None):
    if ax is not None:
        sns.lineplot(data=df_cooc_predicates, x=range(len(df_cooc_predicates)),y="count", label=label, ax=ax)
    else:
        ax = sns.lineplot(data=df_cooc_predicates, x=range(len(df_cooc_predicates)),y="count", label=label)
        
    ax.set_yscale("log")
    ax.set_title("Frequency of each pair of predicates. ")
    ax.set_xlabel("Rank of the predicate pair.")
    ax.set_ylabel("Number of occurrences of the predicate.")
    return ax
    
def plot_pairwise_relplot(df_count_cooccurring_predicates):
    df_count_cooccurring_predicates = get_numerical_coordinates(df_count_cooccurring_predicates)
    
    dd = get_labels_from_cat_codes(df_count_cooccurring_predicates)
    
    norm = LogNorm()
    g = sns.relplot(
        data=df_count_cooccurring_predicates,
        x="pred_left_int",
        y="pred_right_int",
        size="count",
        hue="count",
        hue_norm=norm,
        sizes=(10,250),
        size_norm=norm, 
        palette="vlag",
        height=7
        )
    g.set(xticks=np.arange(len(dd)), yticks=np.arange(len(dd)))
    _=g.set_xticklabels(list(dd.values()), rotation=45, horizontalalignment='right', fontsize=6)
    _=g.set_yticklabels(list(dd.values()), horizontalalignment='right', fontsize=6)
    sns.despine(left=True, bottom=True)
    
