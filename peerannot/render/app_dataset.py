import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from pathlib import Path
import json
from torchvision import transforms
from peerannot.training.load_data import load_data
from peerannot.models.Soft import Soft
import plotly.express as px
import plotly.graph_objs as go
import dash_daq as daq
import numpy as np


def get_image_data(dataset, labs, index):
    normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    inv_normalization = transforms.Normalize(
        mean=[-x * 1 / y for x, y in zip(normalization[0], normalization[1])],
        std=[1 / x for x in normalization[1]],
    )
    img, _ = dataset[index]
    img = np.transpose(inv_normalization(img).numpy(), (1, 2, 0))
    filename, label = dataset.samples[index]
    num = int(filename.split("-")[-1].split(".")[0])
    distrib = labs[num]
    return img, label, filename, distrib


def render_app(folderpath, metadatapath, votespath, port, **kwargs):
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

    # ------------------
    # Define the layout
    # ------------------

    app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("", href="#")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("More pages", header=True),
                    dbc.DropdownMenuItem("Page 2", href="#"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand=f"{dataset_name}",
        brand_href="#",
        color="primary",
        dark=True,
    )

    row1 = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        daq.ToggleSwitch(
                            id="frequency-toggle",
                            label="Use Frequency",
                            value=False,
                        ),
                    ],
                ),
                width={"offset": 7},
            ),
        ],
    )
    row2 = dbc.Row(
        [
            dbc.Col(html.Div(dcc.Graph(id="image"))),
            dbc.Col(
                html.Div(
                    dcc.Graph(id="distribution"),
                )
            ),
        ],
        justify="center",
        className="g-0",
    )
    if len(dataset) <= 1000:
        marks = {k: str(k) for k in range(len(dataset)) if k % 100 == 0}
    else:
        marks = {k: str(k) for k in range(len(dataset)) if k % 1000 == 0}
    row3 = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    dcc.Slider(
                        id="slider",
                        min=0,
                        max=len(dataset),
                        step=1,
                        value=int(len(dataset) / 2),
                        marks=marks,
                    )
                )
            ),
        ],
        justify="center",
    )
    row4 = dbc.Row(
        [
            dbc.Col(html.Center(html.Div(id="true-label-output"))),
            dbc.Col(html.Center(html.Div(id="filename-output"))),
        ],
        justify="center",
        align="center",
        className="g-0",
    )

    body = html.Div([row1, row2, row3, row4])
    app.layout = html.Div([navbar, body])

    # ----------------------
    # Define the callbacks
    # ----------------------
    @app.callback(
        [
            Output("true-label-output", "children"),
            Output("filename-output", "children"),
            Output("distribution", "figure"),
            Output("image", "figure"),
        ],
        [Input("slider", "value"), Input("frequency-toggle", "value")],
    )
    def update_output(value, frequency):
        img, label, filename, distrib = get_image_data(dataset, labs, value)
        fig_img = px.imshow(img)
        fig_img.update_xaxes(showticklabels=False)
        fig_img.update_yaxes(showticklabels=False)
        x = list(dataset.class_to_idx.keys())
        data = [
            go.Bar(
                x=x, y=distrib if not frequency else distrib / distrib.sum()
            )
        ]
        # Customize the layout of the plot
        layout = go.Layout(
            title="Vote repartition",
            xaxis={"title": "Classes"},
            yaxis={"title": "Counts" if not frequency else "Frequency"},
        )
        # Combine the data and layout into a figure
        fig_distrib = go.Figure(data=data, layout=layout)
        return (
            f"Image label is {dataset.inv_class_to_idx[label]}",
            f"File at {filename}",
            fig_distrib,
            fig_img,
        )

    app.run_server(host="0.0.0.0", port=port)
