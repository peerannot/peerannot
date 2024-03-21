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
@click.option(
    "--no-task",
    is_flag=True,
    help="True if no task is associated with the dataset",
    # (only an answers file as in the krippendorff example dataset)
)
@click.option(
    "--answers-format",
    default=0,
    type=click.INT,
    help="annotation file format",
    # 0 == Rodrigues matrix format, 1 == JSON answers/worker format, 2 == JSON worker/answers format
)
@click.option(
    "--answers",
    default="",
    type=click.Path(exists=False),
    help="annotation file",
)
@click.option(
    "--metadata",
    default="",
    type=click.Path(exists=False),
    help="metadata information file",
)
@click.option(
    "--label-names",
    default="",
    type=click.Path(exists=False),
    help="path to label names files",
)
@click.option(
    "--files-path",
    default="",
    type=click.Path(exists=False),
    help="path to train filenames",
)
@click.option(
    "--train-path",
    default="",
    type=click.Path(exists=False),
    help="path to train data",
)
@click.option(
    "--test-ground-truth-format",
    default=-1,
    type=click.INT,
    help="annotation file format",
    # 0 == Rodrigues matrix format, 1 == JSON answers/worker format, 2 == JSON worker/answers format
)
@click.option(
    "--test-ground-truth",
    default="",
    type=click.Path(exists=False),
    help="test ground truth file",
)
@click.option(
    "--test-path",
    default="",
    type=click.Path(exists=False),
    help="path to test data",
)
@click.option(
    "--val-path",
    default="",
    type=click.Path(exists=False),
    help="path to val data",
)
def install(
    path,
    no_task,
    answers_format,
    answers,
    metadata,
    label_names,
    files_path,
    train_path,
    test_ground_truth_format,
    test_ground_truth,
    test_path,
    val_path,
):
    """Download and install dataset

    :param path: path to python file including the class with method `setfolders` to install the data
    :type path: click.Path
    """
    print(train_path)
    pathFilename = Path(path).resolve().as_posix().split("/")[-1].split(".")[0]
    if pathFilename == "customDataset":
        if not no_task:
            if train_path == "":
                click.echo("Please provide a valid train path")
                sys.exit(1)
            if files_path == "":
                click.echo("Please provide a valid filenames path")
                sys.exit(1)
        if answers == "":
            click.echo("Please provide a valid answers file")
            sys.exit(1)
        if test_ground_truth_format == -1:
            test_ground_truth_format = answers_format

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
        df.setfolders(
            no_task,
            answers_format,
            answers,
            metadata,
            label_names,
            files_path,
            train_path,
            test_ground_truth_format,
            test_ground_truth,
            test_path,
            val_path,
        )
    else:
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
