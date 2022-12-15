import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import click
from pathlib import Path
from tqdm import tqdm
import json
import re
import ast
import numpy as np
import peerannot.models as pmod
import peerannot.training as ptrain
from collection.abc import Iterable
from torch.nn.functional.calibration import calibrate_ece, calibrate

train = click.Group(
    name="Running peerannot training",
    help="Commands to train a network that can be used with the PeerAnnot library",
)

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


def load_all_data(folderpath):
    trainset = ptrain.load_data(folderpath / "train")
    testset = ptrain.load_data(folderpath / "test")
    if (folderpath / "val").exists():
        valset = ptrain.load_data(folderpath / "val")
    else:
        valset = None
    return trainset, valset, testset


def get_model(model_name):
    ...


def get_optimizer(model, **kwargs):
    lr = kwargs.get("lr", 0.1)
    momentum = kwargs.get("momentum", 0.9)


def train(datapath, output_name, n_classes, **kwargs):
    # load datasets and create folders
    path_folders = Path(datapath).resolve()
    trainset, valset, testset = load_all_data(path_folders)
    trainloader, testloader = DataLoader(trainset), DataLoader(testset)
    if valset:
        valloader = DataLoader(valset)
    path_best = path_folders / "best_models"
    path_best.mkdir(exist_ok=True)

    # get model and loss
    model = get_model()
    criterion = nn.CrossEntropyLoss()

    # load optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, kwargs)

    # number of epochs and validation loss for early stopping
    n_epochs = int(kwargs.get("n_epochs", 100))
    min_val_loss = 1.0

    # keep history trace: if valset is given, val_loss must be recorded
    logger = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "test_accuracy": [],
        "test_loss": [],
        "test_calibration": [],
    }

    # run training procedure
    for epoch in tqdm(range(n_epochs), desc="Training epoch"):
        # train for one epoch
        logger = run_epoch(model, trainloader, criterion, optimizer, logger)

        # evaluate the model if validation set
        if valset:
            logger = evaluate(model, valloader, criterion, logger, test=False)

            # save if improve
            if logger["val_loss"][-1] < min_val_loss:
                torch.save(
                    model.state_dict(), path_best / f"{output_name}.pth"
                )
                min_val_loss = logger["val_loss"][-1]
            else:
                print("Validation loss stopped improving, stop training")
                break
        scheduler.step()

    # load and test model
    model.load_state_dict(torch.load(path_best / f"{output_name}.pth"))
    logger = evaluate(
        model,
        testloader,
        criterion,
        logger,
        n_classes=int(kwargs["n_classes"]),
    )
    (datapath / "results" / output_name).mkdir(parents=True, exist_ok=True)
    with open(datapath / "results" / f"{output_name}.json", "w") as f:
        json.dump(logger, f, indent=3, ensure_ascii=False)
    print(f"Results stored in {datapath / 'results' / f'{output_name}.json'}")
    print("-" * 10)
    print("Final metrics:")
    for k, v in logger.items():
        if isinstance(v, Iterable):
            vprint = v[-1]
        else:
            vprint = v
        print(f"- {k}: {vprint}")


def evaluate(model, loader, criterion, logger, test=True, n_classes=10):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    ece = []
    class_ece = []
    for inputs, labels in loader:
        # move to device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # logits
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_accuracy += get_accuracy(outputs, labels)
            if test:  # calibration errors
                ece.append(calibrate_ece(outputs, labels, 15).item())
                cl_ce = torch.zeros(n_classes)
                calibrate(outputs, labels, cl_ce, 15)
                class_ece.append(cl_ce.cpu().numpy())

    # compute final metrics and log them
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100 * total_accuracy / len(loader.dataset)
    if test:  # also measure the calibration errors
        logger["test_loss"] = avg_loss
        logger["test_accuracy"] = accuracy
        logger["test_ece"] = np.mean(ece).item()
        logger["test_class_ece"] = (
            np.mean(np.stack(class_ece), axis=1).mean().item()
        )
    else:  # validation
        logger["val_loss"].append(avg_loss)
        logger["val_accuracy"].append(accuracy)


def run_epoch(model, trainloader, criterion, optimizer, logger):
    model.train()
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

        # log everything
        logger["train_loss"].append(loss.item())


def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct_pred = (predicted == labels).sum().item()
    return correct_pred
