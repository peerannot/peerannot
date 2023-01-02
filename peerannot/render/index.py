import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from pathlib import Path
import json
from torchvision import transforms
from peerannot.training.load_data import load_data
from peerannot.models.Soft import Soft
import plotly.express as px
import plotly.graph_objs as go
import dash_daq as daq
import numpy as np


def render_app(folderpath, metadatapath, votespath, port, **kwargs):
    temp = Path(__file__).parent.resolve() / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    with open(temp / "tmp.json", "w") as f:
        json.dump(
            {
                "folderpath": str(folderpath),
                "metadatapath": str(metadatapath),
                "votespath": str(votespath),
            },
            f,
        )
    with open(Path(metadatapath).resolve(), "r") as f:
        metadata = json.load(f)
    dataset_name = metadata["name"]

    from .apps import page_home, page_waum
    from .app import app

    # ------------------
    # Define the layout
    # ------------------

    navbar = dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("", href="/")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("More pages", header=True),
                    dbc.DropdownMenuItem("Home", href="/"),
                    dbc.DropdownMenuItem("AUM and WAUM", href="/waum"),
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand=f"{dataset_name}",
        brand_href="/",
        color="primary",
        dark=True,
    )

    body = html.Div(id="page-content")
    app.layout = html.Div([dcc.Location(id="url"), navbar, body])

    # ----------------------
    # Define the callbacks
    # ----------------------
    @app.callback(
        Output("session", "data"),
        Input("url", "pathname"),
        State("session", "data"),
    )
    def storage(url, data):
        if url is None:
            raise PreventUpdate

        # Give a default data dict if there's no data.
        data = data or {
            "folderpath": folderpath,
            "metadatapath": metadatapath,
            "votespath": votespath,
        }
        return data

    @app.callback(
        Output("page-content", "children"), [Input("url", "pathname")]
    )
    def display_page(pathname):
        if pathname == "/":
            return html.Div(page_home.layout)
        elif pathname == "/waum":
            return page_waum.layout
        else:
            return "404"

    app.run_server(host="0.0.0.0", port=port)
