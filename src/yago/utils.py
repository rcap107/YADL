from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm


def import_from_yago(filepath: Path, engine="polars"):
    """Given a parquet file, read it assuming YAGO format. The last row is dropped.
    Polars and Pandas are supported as engines.

    Args:
        filepath (Path): Path to the yago-like file.
        engine (str, optional): Dataframe engine to use. Defaults to "polars".

    Raises:
        ValueError: Raise ValueError if the supplied engine is not `pandas` or `polars`.

    Returns:
        _type_: Triplets DataFrame.
    """
    if engine == "polars":
        triplets = pl.read_parquet(filepath)[:-1]
    elif engine == "pandas":
        triplets = pd.read_parquet(filepath)
        triplets.drop(triplets.tail(1).index, inplace=True)
    else:
        raise ValueError(f"Unknown engine {engine}")
    triplets.columns = ["id", "subject", "predicate", "cat_object", "num_object"]
    return triplets


def find_unique_predicates(df: Union[pd.DataFrame, pl.DataFrame]):
    """Given a triplet dataframe, return the unique values in column `predicate`.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Input dataframe.

    Raises:
        TypeError: Raise TypeError if `df` has the incorrect type.

    Returns:
        _type_: A df that contains the unique predicates.
    """
    if type(df) == pd.DataFrame:
        return df["predicate"].unique()
    elif type(df) == pl.DataFrame:
        return df.lazy().select(pl.col("predicate").unique()).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")


def count_occurrences_by_columns(
    df: Union[pd.DataFrame, pl.DataFrame], column: str = None, descending=True
):
    """Given a dataframe `df` and a column `column`, return a dataframe that contains the count of values
    in the given column, sorted by default in descending order.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): Dataframe to evaluate.
        column (str, optional): Column to sort by. Defaults to None.
        descending (bool, optional): Whether to sort in descending order or not. Defaults to True.

    Raises:
        ValueError: Raise ValueError if the column is not provided.
        KeyError: Raise KeyError if the column is not found in the table.
        TypeError: Raise TypeError if the dataframe is not `pd.DataFrame` or `pl.DataFrame`.

    Returns:
        _type_: Dataframe that contains the values and their occurences.
    """
    if column is None:
        raise ValueError("Invalid column.")
    if column not in df.columns:
        raise KeyError(f"Column {column} not found. ")

    if type(df) == pd.DataFrame:
        return df.value_counts(column)
    elif type(df) == pl.DataFrame:
        return (
            df.lazy()
            .groupby(column)
            .agg([pl.count()])
            .sort("count", descending=descending)
        ).collect()
    else:
        raise TypeError("Inappropriate dataframe type.")


# Function used to prepare the node layout for the graph picture.
def bipartite(left, right, aspect_ratio=4 / 3, scale=1):
    left = list(set(left))
    right = list(set(right))
    aspect_ratio = 1
    height = 1
    width = aspect_ratio * height
    nodes = left + right
    # offset = (width / 2, height / 2)
    offset = 0

    left_xs = np.repeat(0, len(left))
    right_xs = np.repeat(width, len(right))
    left_ys = np.linspace(0, height, len(left))
    right_ys = np.linspace(0, height, len(right))

    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset

    pos = np.concatenate([top_pos, bottom_pos])
    dict_pos = dict(zip(nodes, pos))
    return dict_pos


def plot_graph(graph, left, right, image_path="", title="", width=800, height=600):
    coord_list = bipartite(left, right)

    edge_trace = get_edge_trace(graph, coord_list)

    node_trace = get_node_trace(graph, coord_list)

    node_text = []
    ns = list(graph.nodes())

    node_colors, node_text = get_node_info(graph, node_text)

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=15),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
        ),
    )
    fig.show()
    if image_path:
        fig.write_html("{}".format(image_path))


def get_node_info(graph, node_text):
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(
            f"{adjacencies[0]}: # of connections: " + str(len(adjacencies[1]))
        )

    return node_adjacencies, node_text


def get_node_trace(graph, coord_list):
    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = coord_list[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        #     text=node_text,
        marker=dict(
            showscale=True,
            # colorscale="Blackbody",
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Number of connections",
                xanchor="left",
                titleside="right",
                # tickvals=[-1, 1, 0],
                # ticktext=["RID", "CID", "Node"],
            ),
            line_width=2,
        ),
    )

    return node_trace


def get_edge_trace(graph, ll):
    edge_x = []
    edge_y = []
    for edge in graph.edges:
        x0, y0 = ll[edge[0]]
        x1, y1 = ll[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    return edge_trace


def get_cooccurring_predicates(df: pl.DataFrame):
    return (
        df.lazy()
        .join(df.lazy(), left_on="subject", right_on="subject", how="left")
        .select([pl.col("predicate"), pl.col("predicate_right")])
        .collect()
    )


def get_count_cooccurring_predicates(df: pl.DataFrame):
    return (
        df.lazy()
        .groupby(["predicate", "predicate_right"])
        .agg(pl.count())
        .sort("count", descending=True)
        .collect()
    )


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


def get_labels_from_cat_codes(df: pd.DateOffset):
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
    
def join_types_predicates(yagotypes, yagofacts, types_subset):    
    types_predicates=(yagotypes.lazy().filter(
        pl.col("cat_object").is_in(types_subset["type"])
    ).join(
        yagofacts.lazy(),
        left_on="subject",
        right_on="subject",
        how="left"
    ).select(
        pl.col("subject"),
        pl.col("cat_object").alias("type"),
        pl.col("predicate_right").alias("predicate")
    ).unique(
        ).drop_nulls(
            ).select(
                [
                    pl.col("type"),
                    pl.col("predicate")
                ]
            ).groupby(
                [
                    pl.col("type"),
                    pl.col("predicate")
                ]
            ).agg(
                [
                    pl.count()
                ]
            ).sort("count", descending=True).collect())
    return types_predicates
