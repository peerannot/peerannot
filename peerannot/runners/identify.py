import click
import torch
import peerannot.training.load_data as ptrain
from pathlib import Path
from .train import get_model, get_optimizer, run_epoch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json

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
    if method.lower() == "AUM":
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
    elif method.lower() == "WAUM".lower():
        dataset = DatasetWithIndex(dataset)
    elif method.lower() == "WAUMstacked".lower():
        # extend the dataset with task x answers
        assert votes, "WAUMstacked need the full json of votes"
        ll = []
        targets = []
        imgs = []
        workers = []
        true_idx = []
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
    print("Available methods for ambiguity identification")
    print("-" * 10)
    for meth in ["AUM", "WAUM", "WAUMstacked"]:
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
    "--n-classes", "-K", default=2, type=int, help="Number of classes"
)
@click.option(
    "--method",
    type=str,
    default="WAUMstacked",
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
    "--data-augmentation",
    is_flag=True,
    default=False,
    show_default=True,
    help="Perform data augmentation on training set with a random choice between RandomAffine(shear=15), RandomHorizontalFlip(0.5) and RandomResizedCrop",
)
def identify(folderpath, n_classes, method, **kwargs):
    print("Running the following configuration:")
    print("-" * 10)
    print(f"- Data at {folderpath}")
    print(f"- number of classes: {n_classes}")
    for key, value in kwargs.items():
        print(f"- {key}: {value}")
    print("-" * 10)
    kwargs["scheduler"] = False
    votes = Path(kwargs["labels"]).resolve() if kwargs["labels"] else None
    if votes:
        with open(votes, "r") as f:
            votes = json.load(f)
        votes = dict(sorted({int(k): v for k, v in votes.items()}.items()))
    path_folders = Path(folderpath).resolve()
    trainset = ptrain.load_data(path_folders / "train", None, **kwargs)

    trainset = adapt_dataset_to_method(trainset, method, n_classes, votes)

    print(f"Train set: {len(trainset)} tasks")
    model = get_model(
        kwargs["model"],
        n_classes,
        n_params=kwargs["n_params"],
        pretrained=kwargs["pretrained"],
        cifar="cifar" in str(path_folders).lower(),
    )
    optimizer, _ = get_optimizer(model, "sgd", **kwargs)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    n_epochs = int(kwargs.get("n_epochs", 50))
    alpha = kwargs["alpha"]
    print(f"Running identification with method: {method}")
    if method == "AUM":
        raise NotImplementedError("Not implemented yet, sorry")
    elif method.lower() == "WAUM".lower():
        from peerannot.models.WAUM import WAUM

        waum = WAUM(
            trainset,
            votes,
            n_classes,
            model,
            criterion,
            optimizer,
            n_epochs,
            verbose=True,
            use_pleiss=kwargs["use_pleiss"],
            maxiterDS=kwargs["maxiter_ds"],
        )
        who = "pleiss" if kwargs["use_pleiss"] else "yang"
        waum.run(alpha=kwargs["alpha"])
        path_waum = path_folders / "identification" / f"waum_{alpha}_{who}"
        path_waum.mkdir(exist_ok=True, parents=True)
        with open(path_waum / "waum.json", "w") as f:
            dump(waum.waum, f)
        print(f"Saved WAUM values at {path_waum / 'waum.json'}")
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
        np.savetxt(path_waum / f"too_hard_{alpha}.txt", waum.too_hard)
        print(f"Saved too hard index at {path_waum / f'too_hard_{alpha}.txt'}")
    elif method.lower() == "WAUMstacked".lower():
        from peerannot.models.WAUM_stacked import WAUM_stacked

        waum = WAUM_stacked(
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
            use_pleiss=kwargs["use_pleiss"],
            maxiterDS=kwargs["maxiter_ds"],
        )
        who = "pleiss" if kwargs["use_pleiss"] else "yang"
        waum.run(alpha=kwargs["alpha"])
        path_waum = (
            path_folders / "identification" / f"waum_stacked_{alpha}_{who}"
        )
        path_waum.mkdir(exist_ok=True, parents=True)
        with open(path_waum / "waum.json", "w") as f:
            dump(waum.waum, f)
        print(f"Saved WAUM stacked values at {path_waum / 'waum.json'}")
        with open(path_waum / "score_per_worker", "w") as f:
            dump(waum.score_per_worker, f, level=2)
        print(
            f"Saved score per worker values at {path_waum / 'score_per_worker.json'}"
        )
        with open(path_waum / "aum_per_worker", "w") as f:
            dump(waum.aum_per_worker, f, level=2)
        print(
            f"Saved AUM per worker values at {path_waum / 'aum_per_worker.json'}"
        )
        np.savetxt(path_waum / f"too_hard_{alpha}.txt", waum.too_hard)
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
