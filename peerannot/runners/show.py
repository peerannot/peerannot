import click
from pathlib import Path
from peerannot.render.index import render_app

show = click.Group(
    name="Running peerannot show",
    help="Commands related to visualizations that can be used with the PeerAnnot library",
)


@show.command(help="Launch a local app to explore the crowdsourced data")
@click.argument(
    "dataset",
    type=click.Path(),
)
@click.option(
    "--metadata",
    default=None,
    type=click.Path(),
    help="Path to the metadata file. If not specified, will be `metadata.json` at the root of the dataset folder.",
)
@click.option(
    "--votes",
    default=None,
    type=click.Path(),
    help="Path to the votes file. If not specified, will be `answers.json` at the root of the dataset folder.",
)
@click.option(
    "--port", "-p", default=8051, type=int, help="Port to run the local app"
)
def render(dataset, metadata, votes, port):
    """Visualize the dataset"""
    dataset = Path(dataset).resolve()
    if metadata is None:
        metadata = dataset / "metadata.json"
    if votes is None:
        votes = dataset / "answers.json"
    dataset = dataset / "train"
    render_app(dataset, metadata, votes, port=port)
