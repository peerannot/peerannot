import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import no_update
from pathlib import Path
import json
from torchvision import transforms
from peerannot.training.load_data import load_data
from peerannot.models.Soft import Soft
import plotly.express as px
import plotly.graph_objs as go
from tqdm.auto import tqdm
import dash_daq as daq
import numpy as np
import pandas as pd
from scipy.special import entr
import re
from ..app import app
import base64
from PIL import Image
import io

# import modin.pandas as pd
from plotly.express.colors import sample_colorscale

with open(
    Path(__file__).parent.parent.resolve() / "tmp" / "tmp.json", "r"
) as f:
    data = json.load(f)
folderpath = data["folderpath"]
metadatapath = data["metadatapath"]
votespath = data["votespath"]
path = Path(folderpath).resolve()
metadata = Path(metadatapath).resolve()
with open(Path(votespath).resolve(), "r") as f:
    votes = json.load(f)
with open(metadata, "r") as f:
    metadata = json.load(f)
soft = Soft(votes, n_classes=metadata["n_classes"])
labs = soft.get_probas()
labs = soft.baseline
dataset = load_data(path, None, None, data_augmentation=False)
dataset_name = metadata["name"]
COLORMAP = "rainbow"
COLORMAPSAMPLE = sample_colorscale(
    COLORMAP, list(np.linspace(0, 1, len(dataset.class_to_idx)))
)


def image_to_base64(img_src):
    im = Image.open(img_src)
    im = im.convert("RGB")
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def aum_from_logits(data, burn=0, n_classes=10):
    tasks = {"index": [], "aum_yang": [], "aum_pleiss": []}
    for idx in tqdm(data["index"].unique(), desc="Tasks"):
        tmp = data[data["index"] == idx]
        y = tmp.label.iloc[0]
        target_values = tmp.label_logit.values[burn:]
        logits = tmp.values[burn:, -n_classes:]
        llogits = np.copy(logits)
        _ = np.put_along_axis(
            logits, logits.argmax(1).reshape(-1, 1), float("-inf"), 1
        )
        masked_logits = logits
        other_logit_values, other_logit_index = masked_logits.max(
            1
        ), masked_logits.argmax(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()
        margin_values_yang = (target_values - other_logit_values).tolist()
        _ = np.put_along_axis(
            llogits, np.repeat(y, len(tmp)).reshape(-1, 1), float("-inf"), 1
        )
        masked_logits = llogits
        other_logit_values, other_logit_index = masked_logits.max(
            1
        ), masked_logits.argmax(1)
        other_logit_values = other_logit_values.squeeze()
        other_logit_index = other_logit_index.squeeze()
        margin_values_pleiss = (target_values - other_logit_values).mean()

        tasks["index"].append(idx)
        tasks["aum_yang"].append(np.mean(margin_values_yang))
        tasks["aum_pleiss"].append(np.mean(margin_values_pleiss))
    df = pd.DataFrame(tasks)
    return df


def make_figure(data, use_pleiss=True):
    fig_aum = go.Figure(
        data=[
            go.Scatter(
                x=data["aum_pleiss"] if use_pleiss else data["aum_yang"],
                y=data["entropy"],
                mode="markers",
                marker_color=["blue"] * len(data["entropy"]),
                marker_size=[10] * len(data["entropy"]),
            )
        ]
    )
    fig_aum.update_traces(hoverinfo="none", hovertemplate=None)
    fig_aum.update_layout(
        title="",
        xaxis_title="AUM",
        yaxis_title="Entropy",
    )
    return fig_aum


def make_figure_waum(data):
    fig_waum = go.Figure(
        data=[
            go.Scatter(
                x=data["waum"],
                y=data["entropy"],
                mode="markers",
                marker_color=["blue"] * len(data["entropy"]),
                marker_size=[10] * len(data["entropy"]),
            )
        ]
    )
    fig_waum.update_traces(hoverinfo="none", hovertemplate=None)
    fig_waum.update_layout(
        title="",
        xaxis_title="WAUM",
        yaxis_title="Entropy",
    )
    return fig_waum


def make_figure_logits(data, num, figure=None):
    if figure:
        figure.data = []
    else:
        figure = go.Figure()
    inv_cl = {v: k for k, v in dataset.class_to_idx.items()}
    tmp = data[(data["task"] == num)]
    x = tmp["epoch"]
    c = COLORMAPSAMPLE
    for i in range(len(dataset.class_to_idx)):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=tmp[f"logits_{i}"],
                name=inv_cl[i],
                line=dict(color=c[i]),
                # line_shape="spline",
            )
        )
    figure.update_traces(hoverinfo="text+name", mode="lines")
    figure.update_layout(
        legend=dict(y=0.5, traceorder="reversed", font_size=16)
    )
    return figure


def make_barplot(dataset, distrib, figure=None):
    x = list(dataset.class_to_idx.keys())
    distrib = np.frombuffer(distrib, dtype=float)
    data = [
        go.Bar(
            x=x,
            y=distrib,
            marker={
                "color": list(dataset.class_to_idx.values()),
                "colorscale": COLORMAP,
            },
        )
    ]
    # Customize the layout of the plot
    layout = go.Layout(
        title="Vote repartition",
        xaxis={"title": "Classes"},
        yaxis={"title": "Frequency"},
    )
    # Combine
    if figure:
        figure.data = []
        for trace in data:
            figure.add_trace(trace)
    else:
        figure = go.Figure(data=data, layout=layout)
    return figure


labs /= labs.sum(axis=1).reshape(-1, 1)
entropy = entr(labs).sum(1)
df_logits = pd.read_csv(
    path.parent / "identification" / "aum" / "full_aum_records.csv"
)
path_waum = (
    path.parent / "identification" / "waum_stacked_0.01_yang" / "waum.csv"
)
if path_waum.exists():
    df_waum = pd.read_csv(path_waum)
else:
    print("Testing to remove")
    df_waum = df_logits[df_logits["epoch"] == 1][
        ["index", "task", "logits_0"]
    ].copy()
    df_waum.rename(
        columns={"index": "index", "task": "task", "logits_0": "waum"},
        inplace=True,
    )
aums = aum_from_logits(df_logits, n_classes=len(dataset.class_to_idx))
all_data = pd.merge(df_logits, aums, on="index", how="outer")
all_data = pd.merge(all_data, df_waum, on=["index", "task"], how="outer")
all_data["entropy"] = [0] * len(all_data["task"])
for i, samp in tqdm(
    enumerate(dataset.samples),
    desc="Entropy and votes",
    total=len(dataset.samples),
):
    num = int(samp[0].split("-")[-1].split(".")[0])
    name = Path(samp[0]).name
    all_data.loc[all_data["task"] == name, "entropy"] = entropy[num]
    all_data.loc[all_data["task"] == name, "votes"] = labs[num].tobytes()

df_used = all_data[all_data.label == 0]
df_aum = df_used[~df_used["aum_pleiss"].isnull()]
NUMBER_OF_TRACES = len(df_aum)
print("Finished preparing dataset")
fig_aum = make_figure(df_aum)
fig_barplot = make_barplot(dataset, df_used["votes"].iloc[0])
fig_waum = make_figure_waum(df_aum)
fig_log = make_figure_logits(df_used, df_used["task"].iloc[0])
# define the layout
####################

row1 = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    daq.ToggleSwitch(
                        id="margin-toggle",
                        label="Use Pleiss et. al margin",
                        value=False,
                    )
                ]
            )
        ),
        dbc.Col(
            html.Div(
                [
                    "Label selection",
                    dcc.Dropdown(
                        options=list(dataset.class_to_idx.keys()),
                        id="label-dropdown",
                        value=list(dataset.class_to_idx.keys())[0],
                    ),
                ],
            ),
            width={"offset": 2},
        ),
    ],
)
row2 = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    "Global Slider",
                    dcc.Slider(
                        id="slider-global",
                        min=1e-4,
                        max=1,
                        step=0.005,
                        value=0.01,
                        marks={
                            i: f"{i:.2f}" for i in np.arange(0, 1.1, step=0.1)
                        },
                    ),
                ]
            )
        ),
        dbc.Col(
            html.Div(
                [
                    "Local Slider",
                    dcc.Slider(
                        id="slider-local",
                        min=1e-4,
                        max=1,
                        step=0.005,
                        value=0.01,
                        marks={
                            i: f"{i:.2f}" for i in np.arange(0, 1.1, step=0.1)
                        },
                    ),
                ],
            )
        ),
    ]
)
row3 = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    dcc.Graph(
                        id="figure-aum",
                        figure=fig_aum,
                        clear_on_unhover=True,
                    ),
                    dcc.Tooltip(id="fig-aum-tooltip"),
                ],
            )
        ),
        dbc.Col(
            html.Div(
                [
                    dcc.Graph(
                        id="figure-waum",
                        figure=fig_waum,
                        clear_on_unhover=True,
                    ),
                    dcc.Tooltip(id="fig-waum-tooltip"),
                ]
            )
        ),
    ]
)
row4 = dbc.Row(
    [
        dbc.Col(
            html.Div(
                [
                    dcc.Graph(id="figure-logits", figure=fig_log),
                    dcc.Loading(
                        id="loading-logits",
                        children=[html.Div(id="loading-output-logits")],
                        type="default",
                    ),
                ]
            )
        ),
        dbc.Col(html.Div(dcc.Graph(id="figure-votes", figure=fig_barplot))),
    ]
)
layout = html.Div([row1, row2, row3, row4])


@app.callback(
    Output("loading-output-logits", "children"),
    [Input("figure-aum", "hoverData"), Input("figure-waum", "hoverData")],
)
def input_triggers_spinner(value1, value2):
    return value1, value2


# Callbacks
# def update_graphs(hoverData, fig_aum, fig_waum, fig_barplot, fig_log):
#     pt = hoverData["points"][0]
#     bbox = pt["bbox"]
#     num = pt["pointNumber"]
#     df_row = df_aum.iloc[num]
#     filename = df_row["task"]
#     img_src = str(path / re.split("[^a-zA-Z]", filename)[0] / filename)
#     name = filename
#     children = [
#         html.Div(
#             [
#                 html.Img(
#                     src=image_to_base64(img_src),
#                     style={
#                         "width": "75px",
#                         "display": "block",
#                         "margin": "0 auto",
#                     },
#                 ),
#                 html.P(
#                     f"{name}",
#                     style={
#                         "color": "darkblue",
#                         "overflow-wrap": "break-word",
#                     },
#                 ),
#             ],
#         )
#     ]
#     fig_barplot = make_barplot(
#         dataset,
#         df_used[df_used["task"] == name]["votes"].iloc[0],
#         figure=fig_barplot,
#     )
#     scatter = fig_aum.data[0]
#     colors = list(scatter.marker.color)
#     colors = ["blue"] * len(colors)
#     colors[num] = "red"
#     with fig_aum.batch_update():
#         scatter.marker.color = colors
#     scatter = fig_waum.data[0]
#     colors = list(scatter.marker.color)
#     colors = ["blue"] * len(colors)
#     colors[num] = "red"
#     with fig_waum.batch_update():
#         scatter.marker.color = colors
#     fig_log = make_figure_logits(df_used, filename, figure=fig_log)
#     return bbox, children, fig_aum, fig_waum, fig_barplot, fig_log


def update_tooltip(hoverData, df_aum):
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    df_row = df_aum.iloc[num]
    filename = df_row["task"]
    img_src = str(path / re.split("[^a-zA-Z]", filename)[0] / filename)
    name = filename
    children = [
        html.Div(
            [
                html.Img(
                    src=image_to_base64(img_src),
                    style={
                        "width": "75px",
                        "display": "block",
                        "margin": "0 auto",
                    },
                ),
                html.P(
                    f"{name}",
                    style={
                        "color": "darkblue",
                        "overflow-wrap": "break-word",
                    },
                ),
            ],
        )
    ]
    return bbox, children


@app.callback(
    [
        Output("fig-aum-tooltip", "show"),
        Output("fig-aum-tooltip", "bbox"),
        Output("fig-aum-tooltip", "children"),
        Output("fig-waum-tooltip", "show"),
        Output("fig-waum-tooltip", "bbox"),
        Output("fig-waum-tooltip", "children"),
    ],
    [Input("figure-aum", "hoverData"), Input("figure-waum", "hoverData")],
)
def trigger_tooltip(hoverData, hoverData_waum):
    global df_used
    global df_aum
    global fig_aum
    global fig_waum
    global fig_log
    global fig_barplot

    ctx = dash.callback_context
    id_ = ctx.triggered[0]["prop_id"].split(".")[0]
    if id_ is None:
        return (
            False,
            no_update,
            no_update,
            False,
            no_update,
            no_update,
        )
    if id_ == "figure-aum":
        if hoverData is None:
            return (
                False,
                no_update,
                no_update,
                False,
                no_update,
                no_update,
            )
        bbox, children = update_tooltip(hoverData, df_aum)
        return (
            True,
            bbox,
            children,
            False,
            no_update,
            no_update,
        )
    elif id_ == "figure-waum":
        if hoverData_waum is None:
            return (
                False,
                no_update,
                no_update,
                False,
                no_update,
                no_update,
            )
        bbox, children = update_tooltip(hoverData_waum, df_aum)
        return (
            False,
            no_update,
            no_update,
            True,
            bbox,
            children,
        )


@app.callback(
    [
        Output("figure-aum", "figure"),
        Output("figure-waum", "figure"),
        Output("figure-votes", "figure"),
        Output("figure-logits", "figure"),
    ],
    [
        Input("figure-aum", "hoverData"),
        Input("figure-waum", "hoverData"),
        Input("label-dropdown", "value"),
    ],
)
def display(hoverData, hoverData_waum, selected_label):
    global df_used
    global df_aum
    global fig_aum
    global fig_waum
    global fig_log
    global fig_barplot

    ctx = dash.callback_context
    id_ = ctx.triggered[0]["prop_id"].split(".")[0]
    if id_ is None:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
        )
    if id_ == "label-dropdown":
        df_used = all_data[
            all_data.label == dataset.class_to_idx[selected_label]
        ]
        df_aum = df_used[~df_used["aum_pleiss"].isnull()]

        fig_aum = make_figure(df_aum)
        fig_waum = make_figure_waum(df_aum)
        fig_barplot = make_barplot(
            dataset, df_used["votes"].iloc[0], figure=fig_barplot
        )
        fig_log = make_figure_logits(
            df_used, df_used["task"].iloc[0], figure=fig_log
        )
        return (
            fig_aum,
            fig_waum,
            fig_barplot,
            fig_log,
        )
    elif id_.startswith("figure"):
        if id_ == "figure-aum":
            hv = hoverData
        else:
            hv = hoverData_waum
        if hv is None:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
            )

        pt = hv["points"][0]
        num = pt["pointNumber"]
        df_row = df_aum.iloc[num]
        name = df_row["task"]
        fig_barplot = make_barplot(
            dataset,
            df_used[df_used["task"] == name]["votes"].iloc[0],
            figure=fig_barplot,
        )
        scatter = fig_aum.data[0]
        colors = list(scatter.marker.color)
        colors = ["blue"] * len(colors)
        colors[num] = "red"
        with fig_aum.batch_update():
            scatter.marker.color = colors
        scatter = fig_waum.data[0]
        colors = list(scatter.marker.color)
        colors = ["blue"] * len(colors)
        colors[num] = "red"
        with fig_waum.batch_update():
            scatter.marker.color = colors
        fig_log = make_figure_logits(df_used, name, figure=fig_log)
        return (fig_aum, fig_waum, fig_barplot, fig_log)
    else:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
        )
