import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import click
from pathlib import Path
from tqdm.auto import tqdm
import torchmetrics
import json
import re
import ast
import numpy as np
import peerannot.models as pmod
import peerannot.training.load_data as ptrain
from collections.abc import Iterable
from peerannot.helpers import networks as nethelp

trainmod = click.Group(
    name="Running peerannot training",
    help="Commands to train a network that can be used with the PeerAnnot library",
)

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


def load_all_data(folderpath, labels_path, **kwargs):
    print("Loading datasets")
    trainset = ptrain.load_data(folderpath / "train", labels_path, **kwargs)
    path_rm = kwargs["path_remove"]
    path_lab = kwargs["labels"]
    data_augm = kwargs["data_augmentation"]
    kwargs["path_remove"] = None  # do not remove tasks in val/test sets
    kwargs["labels"] = None
    kwargs["data_augmentation"] = False
    testset = ptrain.load_data(folderpath / "test", **kwargs)
    if (folderpath / "val").exists():
        valset = ptrain.load_data(folderpath / "val", **kwargs)
    else:
        valset = None
    kwargs["path_remove"] = path_rm
    kwargs["labels"] = path_lab
    kwargs["data_augmentation"] = data_augm
    return trainset, valset, testset


def get_model(
    model_name, n_classes, n_params=None, pretrained=False, cifar=False
):
    assert (
        model_name.lower() in nethelp.get_all_models()
    ), "The neural network asked is not one of available networks, please run `peerannot modelinfo` to get the list of available models"
    model = nethelp.networks(
        model_name, n_classes, n_params=None, pretrained=False, cifar=cifar
    )
    return model


def get_optimizer(net, optimizer, **kwargs):
    use_parameters = kwargs.get("use_parameters", True)  # cf crowdlayer
    milestones = [int(x) for x in kwargs.get("milestones", [1e6])]
    lr = kwargs.get("lr", 0.1)
    momentum = kwargs.get("momentum", 0.9)
    weight_decay = kwargs.get("decay", 5e-4)
    if optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            net.parameters() if use_parameters else net,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "adam":
        optimizer = optim.Adam(
            net.parameters() if use_parameters else net, lr=lr
        )
    else:
        raise ValueError("Not implemented yet")
    if kwargs["scheduler"]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=kwargs["lr_decay"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [1e6], gamma=1
        )
    return optimizer, scheduler


@trainmod.command(
    help="Train a classification neural network given a dataset path, an output name and the number of classes"
)
@click.argument("datapath", default=Path.cwd(), type=click.Path(exists=True))
@click.option(
    "--output-name",
    "-o",
    type=str,
    help="Name of the generated results file",
)
@click.option(
    "--n-classes",
    "-K",
    default=10,
    type=int,
    help="Number of classes to separate",
)
@click.option(
    "--labels", type=click.Path(exists=True), help="Path to file of labels"
)
@click.option(
    "--optimizer",
    "-optim",
    type=str,
    default="SGD",
    help="Optimizer for the neural network",
)
@click.option(
    "--model",
    default="resnet18",
    type=str,
    help="Name of neural network to use. The list is available at `peerannot modelinfo`",
)
@click.option(
    "--img-size", type=int, default=224, help="Size of image (square)"
)
@click.option(
    "--data-augmentation",
    is_flag=True,
    default=False,
    show_default=True,
    help="Perform data augmentation on training set with a random choice between RandomAffine(shear=15), RandomHorizontalFlip(0.5) and RandomResizedCrop",
)
@click.option(
    "--path-remove",
    type=click.Path(),
    default=None,
    help="Path to file of index to prune from the training set",
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
def train(datapath, output_name, n_classes, **kwargs):
    # load datasets and create folders
    print("Running the following configuration:")
    print("-" * 10)
    print(f"- Data at {datapath} will be saved with prefix {output_name}")
    print(f"- number of classes: {n_classes}")
    for key, value in kwargs.items():
        print(f"- {key}: {value}")
    print("-" * 10)
    path_folders = Path(datapath).resolve()
    path_labels = Path(kwargs["labels"]).resolve()
    trainset, valset, testset = load_all_data(
        path_folders, path_labels, **kwargs
    )
    trainloader, testloader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        pin_memory=(torch.cuda.is_available()),
    ), DataLoader(
        testset,
        batch_size=kwargs["batch_size"],
    )
    print(f"Train set: {len(trainloader.dataset)} tasks")
    print(f"Test set: {len(testloader.dataset)} tasks")
    if valset:
        valloader = DataLoader(
            valset,
            batch_size=kwargs["batch_size"],
        )
        print(f"Validation set: {len(valloader.dataset)} tasks")

    path_best = path_folders / "best_models"
    path_best.mkdir(exist_ok=True)

    # get model and loss
    model = get_model(
        kwargs["model"],
        n_classes,
        n_params=kwargs["n_params"],
        pretrained=kwargs["pretrained"],
        cifar="cifar" in datapath.lower(),
    )
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # load optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, **kwargs)

    # number of epochs and validation loss for early stopping
    n_epochs = int(kwargs.get("n_epochs", 100))
    min_val_loss = 1e6
    # keep history trace: if valset is given, val_loss must be recorded
    logger = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "test_loss": [],
    }

    # run training procedure
    for epoch in tqdm(range(n_epochs), desc="Training epoch"):
        # train for one epoch
        logger = run_epoch(model, trainloader, criterion, optimizer, logger)

        # evaluate the model if validation set
        if valset:
            logger = evaluate(
                model,
                valloader,
                criterion,
                logger,
                test=False,
                n_classes=n_classes,
            )

            # save if improve
            if logger["val_loss"][-1] < min_val_loss:
                torch.save(
                    model.state_dict(), path_best / f"{output_name}.pth"
                )
                min_val_loss = logger["val_loss"][-1]
        scheduler.step()
        if epoch in kwargs["milestones"]:
            print()
            print(
                f"Adjusting learning rate to = {scheduler.optimizer.param_groups[0]['lr']:.4f}"
            )

    # load and test model
    model.load_state_dict(torch.load(path_best / f"{output_name}.pth"))
    logger = evaluate(
        model,
        testloader,
        criterion,
        logger,
        n_classes=int(n_classes),
    )

    print("-" * 10)
    print("Final metrics:")
    for k, v in logger.items():
        # print(k, v)
        if isinstance(v, Iterable):
            vprint = v[-1]
        else:
            vprint = v
        print(f"- {k}: {vprint}")
    (path_folders / "results").mkdir(parents=True, exist_ok=True)
    with open(path_folders / "results" / f"{output_name}.json", "w") as f:
        json.dump(logger, f, indent=3, ensure_ascii=False)
    print(
        f"Results stored in {path_folders / 'results' / f'{output_name}.json'}"
    )


def evaluate(model, loader, criterion, logger, test=True, n_classes=10):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_calibration = 0.0
    calib = torchmetrics.CalibrationError(
        "multiclass" if n_classes > 2 else "binary",
        n_bins=15,
        norm="l1",
        num_classes=n_classes,
    )
    with torch.no_grad():
        for inputs, labels in loader:
            # move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # logits
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_accuracy += get_accuracy(outputs, labels)
            total_calibration += calib(outputs, labels) * outputs.size(0)
    # compute final metrics and log them
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100 * total_accuracy / len(loader.dataset)
    if test:  # also measure the calibration errors
        logger["test_loss"] = avg_loss
        logger["test_accuracy"] = accuracy
        logger["test_ece"] = (total_calibration / len(loader.dataset)).item()
        # logger["test_class_ece"] = np.mean(
        #     compute_ece_by_class(model, loader).cpu().numpy()
        # ).item()
        # print(logger)
    else:  # validation
        logger["val_loss"].append(avg_loss)
        logger["val_accuracy"].append(accuracy)
    return logger


def run_epoch(model, trainloader, criterion, optimizer, logger):
    model.train()
    total_loss = 0.0
    for inputs, labels in trainloader:
        # move to device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero out gradients
        optimizer.zero_grad()

        # logits
        outputs = model(inputs)

        # loss
        loss = criterion(outputs, labels)

        # gradient step
        loss.backward()
        optimizer.step()
        total_loss += loss
    # log everything
    logger["train_loss"].append(total_loss.item())
    return logger


def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct_pred = (predicted == labels).sum().item()
    return correct_pred


@trainmod.command(help="Display available model names")
def modelinfo():
    print("Available models to train:")
    print("-" * 10)
    for mod in nethelp.get_all_models():
        print(f"- {mod}")
    print("-" * 10)
    return


def compute_ece(model, dataloader, num_bins=15):
    model.eval()
    with torch.no_grad():
        total_ece = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            predictions = outputs.softmax(dim=1)

            pred_y = torch.argmax(predictions, axis=-1)
            correct = pred_y == targets
            prob_y = np.max(predictions, axis=-1)
            b = torch.linspace(start=0, stop=1.0, num=num_bins)
            bins = np.digitize(prob_y, bins=b, right=True)

            o = 0
            for b in range(num_bins):
                mask = bins == b
                if torch.any(mask):
                    o += torch.abs(torch.sum(correct[mask] - prob_y[mask]))

            total_ece += o / predictions.shape[0]
    average_ece = total_ece / len(dataloader.dataset)
    return average_ece


def compute_ece_by_class(model, dataloader, num_bins=15):
    model.eval()
    num_classes = model.output_size
    ece_by_class = torch.zeros(num_classes)
    num_samples_by_class = torch.zeros(num_classes)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)
            confidence, _ = torch.max(outputs, dim=1)
            for c in range(num_classes):
                idx = (labels == c).nonzero().view(-1)
                num_samples_by_class[c] += idx.size(0)
                if idx.size(0) == 0:
                    continue
                confidences = confidence[idx]
                accuracies = (predictions[idx] == labels[idx]).float()

                # Bin the confidence and accuracy
                bin_idx = torch.floor((num_bins - 1) * confidences).long()
                bin_count = torch.bincount(bin_idx, minlength=num_bins)
                bin_accuracy = torch.zeros(num_bins)
                for i in range(num_bins):
                    idx = (bin_idx == i).nonzero().view(-1)
                    bin_accuracy[i] = accuracies[idx].mean()
                bin_confidence = torch.zeros(num_bins)
                for i in range(num_bins):
                    idx = (bin_idx == i).nonzero().view(-1)
                    bin_confidence[i] = confidences[idx].mean()

                # Compute ECE for class c
                ece_by_class[c] += torch.sum(
                    torch.abs(bin_accuracy - bin_confidence) * bin_count
                )
    ece_by_class /= num_samples_by_class
    return ece_by_class
