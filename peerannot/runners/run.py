# based on https://benchopt.github.io CLI
import click
from pathlib import Path
import json
import re
import ast
import numpy as np
import peerannot.models as pmod

# import peerannot.helpers as phelp

agg_strategies = pmod.agg_strategies
agg_strategies = {k.lower(): v for k, v in agg_strategies.items()}
run = click.Group(
    name="Running peerannot",
    help="Commands that can be used with the PeerAnnot library",
)


def check_dataset(path, path_metadata):
    assert (
        path / "answers.json"
    ).exists(), (
        "Dataset path should contain the votes in a `answers.json` file"
    )
    assert (
        path_metadata.exists()
    ), "Dataset path should contain a `metadata.json` file with necessary information such as number of classes. Please read doc for more information"


@run.command(help="Display possible aggregation methods")
def agginfo():
    print("Available aggregation scheme with `peerannot aggregate`:")
    print("-" * 10)
    for agg in agg_strategies.keys():
        print(f"- {agg}")
    print("-" * 10)
    return


@run.command(help="Crowdsourcing strategy using deep learning models")
@click.argument(
    "dataset",
    default=Path.cwd(),
    type=click.Path(exists=True),
)
@click.option(
    "--n-classes",
    "-K",
    default=10,
    type=int,
    help="Number of classes to separate",
)
@click.option(
    "--output-name",
    "-o",
    type=str,
    help="Name of the generated results file",
)
@click.option(
    "--strategy",
    "-s",
    type=str,
    default="conal",
    help="Deep learning strategy",
)
@click.option(
    "--model", type=str, default="resnet18", help="Neural network to train on"
)
@click.option(
    "--answers",
    type=click.Path(),
    default=Path.cwd() / "answers.json",
    help="Crowdsourced labels in json file",
)
@click.option(
    "--img-size", type=int, default=224, help="Size of image (square)"
)
@click.option(
    "--pretrained",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use torch available weights to initialize the network",
)
@click.option(
    "--n-epochs", type=int, default=100, help="Number of training epochs"
)
@click.option("--lr", type=float, default=0.1, help="Learning rate")
@click.option(
    "--momentum", type=float, default=0.9, help="Momentum for the optimizer"
)
@click.option(
    "--decay", type=float, default=5e-4, help="Weight decay for the optimizer"
)
@click.option(
    "--scheduler",
    is_flag=True,
    show_default=True,
    default=False,
    help="Use a multistep scheduler for the learning rate",
)
@click.option(
    "--milestones",
    "-m",
    type=int,
    multiple=True,
    default=[50],
    help="Milestones for the learning rate decay scheduler",
)
@click.option(
    "--n-params",
    type=int,
    default=int(32 * 32 * 3),
    help="Number of parameters for the logistic regression only",
)
@click.option(
    "--lr-decay",
    type=float,
    default=0.1,
    help="Learning rate decay for the scheduler",
)
@click.option("--num-workers", type=int, default=1, help="Number of workers")
@click.option("--batch-size", default=64, type=int, help="Batch size")
@click.option(
    "--optimizer",
    "-optim",
    type=str,
    default="SGD",
    help="Optimizer for the neural network",
)
@click.option(
    "--data-augmentation",
    is_flag=True,
    default=False,
    show_default=True,
    help="Perform data augmentation on training set with a random choice between RandomAffine(shear=15), RandomHorizontalFlip(0.5) and RandomResizedCrop",
)
def aggregate_deep(**kwargs):
    print("Running the following configuration:")
    print("-" * 10)
    print(
        f"- Data at {kwargs['dataset']} will be saved with prefix {kwargs['output_name']}"
    )
    print(f"- number of classes: {kwargs['n_classes']}")
    for key, value in kwargs.items():
        print(f"- {key}: {value}")
    print("-" * 10)
    strat_name, options = get_options(kwargs["strategy"])
    if strat_name.lower() == "conal":
        strat = agg_strategies[strat_name]
        strat = strat(
            tasks_path=kwargs["dataset"],
            scale=options.get("scale", 1e-4),
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Not implemented yet, sorry, maybe a simple `peerannot aggregate` is enough ;)"
        )
    strat.run(**kwargs)


@run.command(
    help="Aggregate crowdsourced labels stored in the provided directory",
    epilog="All aggregated labels are stored in the associated"
    " dataset directory with the strategy name",
)
@click.argument(
    "dataset",
    default=Path.cwd(),
    type=click.Path(exists=True),
)
@click.option(
    "--strategy",
    "-s",
    default="MV",
    type=str,
    help="Aggregation strategy to compute estimated labels from",
)
@click.option(
    "--hard",
    is_flag=True,
    show_default=True,
    default=False,
    help="Only consider hard labels even if the strategy produces soft labels",
)
@click.option(
    "--metadata_path",
    type=click.Path(),
    default=None,
    help="Path to the metadata of the dataset if different than default",
)
@click.option(
    "--answers-file",
    type=str,
    default="answers.json",
    help="Name (with json extension) of the path to the crowdsourced labels",
)
def aggregate(**kwargs):
    """Aggregate labels from a dictionnary of crowdsourced tasks according to a given strategy

    The dataset given is a path to the dataset directory
    """
    # load answers and metadata
    kwargs["dataset"] = Path(kwargs["dataset"]).resolve()
    if kwargs["metadata_path"] is None:
        kwargs["metadata_path"] = kwargs["dataset"] / "metadata.json"
    else:
        kwargs["metadata_path"] = Path(["metadata_path"]).resolve()
    check_dataset(kwargs["dataset"], kwargs["metadata_path"])
    with open(kwargs["dataset"] / kwargs["answers_file"], "r") as answers:
        answers = json.load(answers)
    with open(kwargs["metadata_path"], "r") as metadata:
        metadata = json.load(metadata)
    strat_name, options = get_options(kwargs["strategy"])
    strat = agg_strategies[strat_name]
    print(f"Running aggregation {strat_name} with options {options}")
    if strat_name in list(map(lambda x: x.lower(), ["MV", "NaiveSoft"])):
        strat = strat(answers, metadata["n_classes"])
    elif strat_name in list(map(lambda x: x.lower(), ["DS", "GLAD", "DSwc"])):
        strat = strat(answers, metadata["n_classes"], **options)
        strat.run()
    else:
        raise ValueError(
            f"Strategy {strat_name} is not one of {list(agg_strategies.keys())}"
        )
    filename = f"labels_{metadata['name']}_{strat_name}"
    if kwargs["hard"]:
        yhat = strat.get_answers()
        filename += "_hard"
    else:
        yhat = strat.get_probas()
    path_results = kwargs["dataset"] / "labels"
    path_results.mkdir(parents=True, exist_ok=True)
    path_file = path_results / (filename + ".npy")
    np.save(path_file, yhat)
    print(f"Aggregated labels stored at {path_file} with shape {yhat.shape}")


def get_options(strat):
    strat_name = "".join(re.split(r"\[.*\]", strat)).lower()
    options = re.findall(r"\[.*\]", strat)

    if len(options) == 0:
        return strat_name, {}
    elif len(options) > 1:
        raise ValueError("Only one set of brackets is allowed")
    else:
        match = options[0]
        match = match[1:-1]  # remove brackets
        all_options = re.findall(r"'[^'\"]*'", match)
        all_options += re.findall(r'"[^\'"]*"', match)
        all_options += re.findall(
            r"(?<![a-zA-Z0-9_])[+-]?[0-9]+[.]?[0-9]*[eE][-+]?[0-9]+", match
        )
        for oo in all_options:
            match = match.replace(oo, str(hash(oo)))
        match = re.sub(r"[a-zA-Z][a-zA-Z0-9._-]*", r"'\g<0>'", match)
        for oo in all_options:
            match = match.replace(str(hash(oo)), oo)
        match = "{" + match.replace("=", ":") + "}"
        for token in ["True", "False", "None"]:
            match = match.replace(f'"{token}"', token)
            match = match.replace(f"'{token}'", token)
        result = ast.literal_eval(match)
    return strat_name, result
