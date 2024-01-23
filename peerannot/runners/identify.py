import click
import torch
import peerannot.training.load_data as ptrain
from pathlib import Path
from .train import get_model, get_optimizer, run_epoch, evaluate
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import peerannot.models as pmod

identification_strategies = pmod.identification_strategies
identification_strategies = {
    k.lower(): v for k, v in identification_strategies.items()
}

identification = click.Group(
    name="Running task identification with peerannot",
    help="Commands that can be used to identify ambiguous tasks in crowdsourcing settings with the PeerAnnot library. This uses the AUM/WAUM metric",
)
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


class DatasetWithIndex(Dataset):
    """A wrapper to make dataset return the task index

    :param Dataset: Dataset with tasks to handle
    :type Dataset: torch.Dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.samples)

    def __getitem__(self, index):
        return (*self.dataset[index], self.dataset.true_labels[index], index)


class DatasetWithIndexAndWorker(Dataset):
    """A wrapper to make dataset return the task index

    :param Dataset: Dataset with tasks to handle
    :type Dataset: torch.Dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (
            *self.dataset[index],
            self.dataset.workers[index],
            self.dataset.true_index[index],
            index,
        )


def adapt_dataset_to_method(dataset, method, n_classes, votes=None):
    if method.lower() == "AUM".lower():
        # use the original labels during training
        ll = []
        targets = []
        for i, samp in enumerate(dataset.samples):
            ll.append((samp[0], dataset.true_labels[i]))
            targets.append(dataset.true_labels[i])
        dataset.samples = ll
        dataset.targets = targets
        # the dataset should return the index id
        dataset = DatasetWithIndex(dataset)
    elif method.lower() == "WAUMperworker".lower():
        dataset = DatasetWithIndex(dataset)
    elif method.lower() == "WAUM".lower():
        # extend the dataset with task x answers
        assert votes, "WAUM need the full json of votes"
        ll = []
        targets = []
        imgs = []
        workers = []
        true_idx = []
        dataset.base_samples = dataset.samples
        for i, samp in enumerate(dataset.samples):
            img, label = samp
            num = int(img.split("-")[-1].split(".")[0])
            for worker, worker_vote in votes[num].items():
                ll.append((img, worker_vote))
                targets.append(worker_vote)
                workers.append(int(worker))
                true_idx.append(i)
                imgs.append(img)
        dataset.targets = targets
        dataset.samples = ll
        dataset.true_index = true_idx
        dataset.workers = workers
        dataset.imgs = imgs
        dataset = DatasetWithIndexAndWorker(dataset)
    return dataset


@identification.command(
    help="Display available method to identify ambiguous tasks"
)
def identificationinfo():
    print("Available methods for identification")
    print("-" * 10)
    for meth in identification_strategies.keys():
        print(f"- {meth}")
    print("-" * 10)
    return


def dump(js, file, level=1):
    if level == 1:
        json.dump(
            {int(k): v for k, v in js.items()},
            file,
            indent=3,
            ensure_ascii=False,
        )
    elif level == 2:
        json.dump(
            {int(k): {int(t): v for t, v in js[k].items()} for k in js.keys()},
            file,
            indent=3,
            ensure_ascii=False,
        )


@identification.command(
    help="Identify ambiguous tasks using different methods available in `peerannot identificationinfo`"
)
@click.argument(
    "folderpath",
    default=Path.cwd(),
    type=click.Path(exists=True),
)
@click.option(
    "--hard-labels",
    type=None,
    help="Path to file of hard labels (only for AUM)",
)
@click.option(
    "--n-classes", "-K", default=2, type=int, help="Number of classes"
)
@click.option(
    "--method",
    "-s",
    type=str,
    default="WAUM",
    help="Method to find ambiguous tasks",
)
@click.option(
    "--labels",
    default=Path.cwd() / "answers.json",
    type=click.Path(),
    help="Path to file of crowdsourced answers",
)
@click.option(
    "--use-pleiss",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use Pleiss et. al (2020) margin instead of Yang's",
)
@click.option(
    "--model",
    type=str,
    default="resnet18",
    help="Name of neural network to use. The list is available at `peerannot modelinfo`",
)
@click.option(
    "--n-epochs", type=int, default=50, help="Number of training epochs"
)
@click.option(
    "--alpha", type=float, default=0.01, help="Cutoff hyperparameter"
)
@click.option("--topk", type=int, default=0, help="Use TopK WAUM with k=XXX")
@click.option(
    "--n-params",
    type=int,
    default=int(32 * 32 * 3),
    help="Number of parameters for the logistic regression only",
)
@click.option("--lr", type=float, default=0.1, help="Learning rate")
@click.option(
    "--pretrained",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use torch available weights to initialize the network",
)
@click.option(
    "--momentum", type=float, default=0.9, help="Momentum for the optimizer"
)
@click.option(
    "--metadata_path",
    type=click.Path(),
    default=None,
    help="Path to the metadata of the dataset if different than default",
)
@click.option(
    "--decay", type=float, default=5e-4, help="Weight decay for the optimizer"
)
@click.option(
    "--img-size", type=int, default=224, help="Size of image (square)"
)
@click.option(
    "--maxiter-DS",
    type=int,
    default=50,
    help="Maximum number of iterations for the Dawid and Skene algorithm",
)
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
@click.option(
    "--freeze",
    is_flag=True,
    default=False,
    show_default=True,
    help="Freeze all layers of the network except for the last one",
)
@click.option(
    "--matrix-file",
    type=click.Path(),
    default=None,
    help="Path to confusion matrices saved with an aggregation method like DS. If not provided, run DS model",
)
@click.option(
    "--hard-labels",
    type=click.Path(),
    default=None,
    help="Path to file of hard labels",
)
@click.option("--seed", default=0, type=int, help="random seed")
def identify(folderpath, n_classes, method, **kwargs):
    print("Running the following configuration:")
    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    print("-" * 10)
    print(f"- Data at {folderpath}")
    print(f"- number of classes: {n_classes}")
    for key, value in kwargs.items():
        print(f"- {key}: {value}")
    print("-" * 10)
    kwargs["scheduler"] = False
    if kwargs["metadata_path"] is None:
        kwargs["metadata_path"] = Path(folderpath) / "metadata.json"
    else:
        kwargs["metadata_path"] = Path(["metadata_path"]).resolve()
    with open(kwargs["metadata_path"], "r") as metadata:
        metadata = json.load(metadata)
    kwargs["n_workers"] = metadata["n_workers"]
    if method.lower() != "AUM".lower():
        votes = Path(kwargs["labels"]).resolve() if kwargs["labels"] else None
        if votes:
            with open(votes, "r") as f:
                votes = json.load(f)
            votes = dict(sorted({int(k): v for k, v in votes.items()}.items()))
        labels_to_load = None
    else:
        votes = None
        labels_to_load = kwargs.get("hard_labels", None)
        if labels_to_load:
            labels_to_load = Path(labels_to_load).resolve()
    path_folders = Path(folderpath).resolve()

    if "aum" not in method.lower():
        strategy = identification_strategies[method.lower()]
        strat = strategy(votes, n_classes=n_classes, **kwargs)
        strat.run(path=folderpath)
        return
    trainset = ptrain.load_data(
        path_folders / "train", labels_to_load, **kwargs
    )

    trainset = adapt_dataset_to_method(trainset, method, n_classes, votes)
    print(f"Train set: {len(trainset)} tasks")
    model = get_model(
        kwargs["model"],
        n_classes,
        n_params=kwargs["n_params"],
        pretrained=kwargs["pretrained"],
        cifar="cifar" in str(path_folders).lower(),
        freeze=kwargs.get("freeze", False),
    )
    optimizer, _ = get_optimizer(model, **kwargs)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    n_epochs = int(kwargs.get("n_epochs", 50))
    alpha = kwargs["alpha"]
    print(f"Running identification with method: {method}")
    logger = {"val_loss": [], "val_accuracy": []}  # pretend it's val
    if method.lower() == "aum":
        from peerannot.models import AUM

        aum = AUM(
            DataLoader(
                trainset,
                batch_size=64,
                pin_memory=True,
                num_workers=1,
                shuffle=True,
            ),
            n_classes,
            model,
            criterion,
            optimizer,
            n_epochs,
            verbose=True,
            use_pleiss=kwargs["use_pleiss"],
        )
        aum.run()
        logger = evaluate(
            model,
            DataLoader(
                ptrain.load_data(path_folders / "train", None, **kwargs),
                batch_size=64,
                pin_memory=True,
                num_workers=1,
                shuffle=False,
            ),
            criterion,
            logger,
            test=False,
            n_classes=n_classes,
        )
        kwargs["model"] = kwargs["model"].lower()
        logger["train_accuracy"] = logger["val_accuracy"]
        logger["train_loss"] = logger["val_loss"]
        del logger["val_accuracy"]
        del logger["val_loss"]
        print(logger)
        who = "pleiss" if kwargs["use_pleiss"] else "yang"
        path_aum = path_folders / "identification" / kwargs["model"] / "aum"
        path_aum.mkdir(exist_ok=True, parents=True)
        aum.AUM_recorder.to_csv(path_aum / "full_aum_records.csv", index=False)
        print(f"Saved full log at {path_aum / 'full_aum_records.csv'}")

        aum.aums.to_csv(path_aum / "aum_values.csv", index=False)
        print(f"Saved AUM values at {path_aum / 'aum_values.csv'}")
        np.savetxt(
            path_aum / f"too_hard_{alpha}.txt",
            aum.too_hard.astype(int),
            fmt="%i",
        )
        print(f"Saved too hard index at {path_aum / f'too_hard_{alpha}.txt'}")

    elif method.lower() == "WAUM_perworker".lower():
        from peerannot.models import WAUM_perworker

        waum = WAUM_perworker(
            trainset,
            votes,
            n_classes,
            model,
            criterion,
            optimizer,
            n_epochs,
            verbose=True,
            maxiterDS=kwargs["maxiter_ds"],
            n_workers=kwargs["n_workers"],
        )
        who = "pleiss" if kwargs["use_pleiss"] else "yang"
        waum.run(alpha=kwargs["alpha"])
        path_waum = (
            path_folders
            / "identification"
            / kwargs["model"]
            / f"waum_perworker_{alpha}_{who}"
        ).resolve()
        path_waum.mkdir(exist_ok=True, parents=True)
        waum.waum.to_csv(path_waum / "waum.csv", index=False)
        print(f"Saved WAUM per worker values at {path_waum / 'waum.csv'}")
        with open(path_waum / "score_per_worker.json", "w") as f:
            dump(waum.score_per_worker, f, level=2)
        print(
            f"Saved score per worker values at {path_waum / 'score_per_worker.json'}"
        )
        with open(path_waum / "aum_per_worker.json", "w") as f:
            dump(waum.aum_per_worker, f, level=2)
        print(
            f"Saved AUM per worker values at {path_waum / 'aum_per_worker.json'}"
        )
        np.savetxt(
            path_waum / f"too_hard_{alpha}.txt",
            waum.too_hard.astype(int),
            fmt="%i",
        )
        print(f"Saved too hard index at {path_waum / f'too_hard_{alpha}.txt'}")
    elif method.lower() == "WAUM".lower():
        from peerannot.models import WAUM

        waum = WAUM(
            DataLoader(
                trainset, batch_size=64, pin_memory=True, num_workers=1
            ),
            votes,
            n_classes,
            model,
            criterion,
            optimizer,
            n_epochs,
            verbose=True,
            topk=kwargs["topk"],
            maxiterDS=kwargs["maxiter_ds"],
            use_pleiss=kwargs["use_pleiss"],
            n_workers=kwargs["n_workers"],
        )
        who = "pleiss" if kwargs["use_pleiss"] else "yang"
        waum.run(alpha=kwargs["alpha"])
        path_waum = (
            path_folders
            / "identification"
            / kwargs["model"]
            / f"waum_{alpha}_{who}"
        )
        path_waum.mkdir(exist_ok=True, parents=True)
        waum.waum.to_csv(path_waum / "waum.csv", index=False)
        print(f"Saved WAUM values at {path_waum / 'waum.csv'}")
        with open(path_waum / "score_per_worker.json", "w") as f:
            dump(waum.score_per_worker, f, level=2)
        print(
            f"Saved score per worker values at {path_waum / 'score_per_worker.json'}"
        )
        with open(path_waum / "aum_per_worker.json", "w") as f:
            dump(waum.aum_per_worker, f, level=2)
        print(
            f"Saved AUM per worker values at {path_waum / 'aum_per_worker.json'}"
        )
        np.savetxt(
            path_waum / f"too_hard_{alpha}.txt",
            waum.too_hard.astype(int),
            fmt="%i",
        )
        print(f"Saved too hard index at {path_waum / f'too_hard_{alpha}.txt'}")

    if method.startswith("WAUM"):
        path_results = path_folders / "labels"
        path_results.mkdir(parents=True, exist_ok=True)
        path_file = path_results / f"labels_{method.lower()}_{str(alpha)}.npy"
        yhat = waum.get_probas()
        np.save(path_file, yhat)
        print(
            f"Aggregated labels stored at {path_file} with shape {yhat.shape}"
        )
