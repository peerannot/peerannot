# based on https://benchopt.github.io CLI
import click
from pathlib import Path
import numpy as np
import importlib.util
import inspect
import sys

datasets = click.Group(
    name="Running peerannot datasets",
    help="Commands related to datasets that can be used with the PeerAnnot library",
)


@datasets.command(
    help="Install dataset from `.py` file",
    epilog="""Each dataset is a folder with:

    \b
- name.py: python file containing how to download and format data
- answers.json: json file containing each task voted labels
- metadata.json: all metadata for dataset, at least the name, n_task and n_classes""",
)
@click.argument(
    "path",
    type=click.Path(),
)
def install(path):
    """Download and install dataset

    :param path: path to python file including the class with method `setfolders` to install the data
    :type path: click.Path
    """
    spec = importlib.util.spec_from_file_location("dataset", path)
    mydata = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = mydata
    spec.loader.exec_module(mydata)
    mm = [
        (name, cls)
        for name, cls in inspect.getmembers(mydata, inspect.isclass)
        if cls.__module__ == "dataset"
    ][0]
    df = mm[1]()
    df.setfolders()
