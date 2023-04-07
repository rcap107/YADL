import numpy as np
import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def plot_bipartite_type_pred_graph(df: pd.DataFrame):
    if ("type" not in df.columns) or ("predicate" not in df.columns):
        raise KeyError("Required columns not found.")

    G = nx.Graph()
    for edge in df.iter_rows():
        left, right, weight = edge
        G.add_edge(left,right, weight=weight)
    left, right = df["type"], df["predicate"]
    plot_graph(G, left, right)



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
