from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import networkx as nx


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
        triplets.drop(triplets.tail(1).index, inplace=True)
        triplets = pd.read_parquet(filepath)
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
        node_text.append(f"{adjacencies[0]}: # of connections: " + str(len(adjacencies[1])))

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
