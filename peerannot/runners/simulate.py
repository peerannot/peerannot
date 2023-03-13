import click
from pathlib import Path
import json
import numpy as np

from peerannot.helpers.simulations_strategies import simulation_strategies

simulation = click.Group(
    name="Running peerannot simulations",
    help="Simulate labels from multiple strategies",
)


@simulation.command(
    help="Crowdsourcing simulations of workers",
)
@click.option("--n-worker", type=int, default=1, help="Number of workers")
@click.option("--n-task", type=int, default=20, help="Number of tasks")
@click.option(
    "--n-classes", "-K", type=int, default=2, help="Number of classes"
)
@click.option(
    "--folder",
    type=click.Path(),
    default=Path.cwd() / "simulation",
    help="Folder in which produces simulations are stored.",
)
@click.option(
    "--strategy",
    "-s",
    type=str,
    default="hammer-spammer",
    help="Type of worker simulation",
)
@click.option(
    "--matrix-file",
    type=click.Path(exists=True),
    help="Numpy file containing a tensor of confusion matrices of size (n_worker, n_classes, n_classes)",
)
@click.option(
    "--ratio",
    "-r",
    type=float,
    default=0.1,
    help="Number in (0,1) representing the ratio of spammers/students/good workers amongst total number of workers (depending on the strategy used)",
)
@click.option(
    "--ratio-diff",
    type=float,
    default=1,
    help="Ratio of easy tasks amongst hard. Only used in simulations based on task difficulty",
)
@click.option(
    "--random",
    type=float,
    default=0,
    help="Probability for a given task to have a difficulty `random` ie to be unidentifiable",
)
@click.option(
    "--workerload",
    "-wl",
    type=int,
    help="Upper bound on the number of tasks answered per worker",
)
@click.option(
    "--feedback",
    "-fb",
    type=int,
    help="Upper bound on the number of labels per task",
)
@click.option(
    "--imbalance-votes",
    is_flag=True,
    default=False,
    help="If set, the number of votes per task is randomly chosen between 1 and the possible number of votes considering the constraint on the workerload and feedback force.",
)
@click.option(
    "--seed", type=int, default=0, help="Randome state for reproducibility"
)
@click.option(
    "--verbose", is_flag=True, default=False, help="Display more information"
)
def simulate(**kwargs):
    n_worker, n_task, K, path_ = (
        kwargs.pop("n_worker"),
        kwargs.pop("n_task"),
        kwargs.pop("n_classes"),
        kwargs.get("folder"),
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    rng = np.random.default_rng(kwargs["seed"])
    strat = kwargs["strategy"]
    assert (
        strat.lower() in simulation_strategies.keys()
    ), f"Strategy should be one of {list(simulation_strategies.keys())}"

    path_ = Path(path_)
    path_.mkdir(parents=True, exist_ok=True)
    strat = simulation_strategies[strat.lower()]
    true_labels = rng.choice(K, size=n_task, replace=True)
    answers = strat(n_worker, true_labels, K, rng, **kwargs)
    with open(path_ / "answers.json", "w") as f:
        json.dump(answers, f, indent=3)
    metadata = {
        "name": kwargs["strategy"],
        "n_workers": n_worker,
        "n_classes": K,
        "n_task": n_task,
        **kwargs,
    }
    with open(path_ / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=3)
    np.save(path_ / "ground_truth.npy", true_labels)
    print(
        f"""
    Saved answers at {path_ / 'answers.json'} \n
    Saved ground truth at {path_ / 'ground_truth.npy'}
    """
    )
