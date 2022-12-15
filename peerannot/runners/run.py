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
    with open(kwargs["dataset"] / "answers.json", "r") as answers:
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
        for match in all_options:
            match = options.replace(match, str(hash(match)))
        match = re.sub(r"[a-zA-Z][a-zA-Z0-9._-]*", r"'\g<0>'", match)
        for match in all_options:
            match = match.replace(str(hash(match)), match)
        match = "{" + match.replace("=", ":") + "}"
        for token in ["True", "False", "None"]:
            match = match.replace(f'"{token}"', token)
            match = match.replace(f"'{token}'", token)
        result = ast.literal_eval(match)
    return strat_name, result


if __name__ == "__main__":
    aggregate(["--help"])
