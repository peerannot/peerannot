"""
===================================
Crowdlayer (Rodrigues et. al 2018)
===================================

End-to-end learning strategy with multiple votes per task

Using:
- Crowd layer added to network

Code:
- Tensorflow original code available at https://github.com/fmpr/CrowdLayer
- Code adaptated in Python
"""
import torch
from torch import nn
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from collections.abc import Iterable
from .template import CrowdModel
from pathlib import Path
from tqdm.auto import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader

DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"


def reformat_labels(votes, n_workers):
    answers = []
    for task, ans in votes.items():
        answers.append([-1] * n_workers)
        for worker, lab in ans.items():
            answers[int(task)][int(worker)] = lab
    return np.array(answers)


class Crowdlayer_net(nn.Module):
    def __init__(
        self,
        n_class,
        n_annotator,
        classifier,
        scale,
        criterion,
    ):
        super().__init__()

        self.classifier = classifier
        self.n_worker = n_annotator
        self.n_classes = n_class
        self.scale = scale
        self.criterion = criterion
        self.workers = [torch.eye(n_class) for _ in range(self.n_worker)]
        self.confusion = nn.parameter.Parameter(
            torch.stack(self.workers), requires_grad=True
        )

    def forward(self, x, labels):
        z_pred = F.softmax(self.classifier(x), dim=1)
        pm = F.softmax(self.confusion, dim=2)
        ann_pred = torch.einsum("ik,jkl->ijl", z_pred, pm).view(
            (-1, self.n_classes)
        )

        reg = torch.zeros(1).to(DEVICE)
        for i in range(self.n_worker):
            reg += pm[i, 0, 0].log()

        labels = labels.view(-1)
        loss = self.criterion(ann_pred, labels) + self.scale * reg
        return loss


class Crowdlayer(CrowdModel):
    def __init__(
        self,
        tasks_path,
        answers,
        model,
        n_classes,
        optimizer,
        n_epochs,
        scale=0,
        verbose=True,
        pretrained=False,
        output_name="conal",
        **kwargs,
    ):
        from peerannot.runners.train import (
            get_model,
            get_optimizer,
            load_all_data,
        )  # avoid circular imports

        self.scale = scale
        self.tasks_path = Path(tasks_path).resolve()
        self.answers = Path(answers).resolve()
        with open(self.answers, "r") as ans:
            self.answers = json.load(ans)
        super().__init__(self.answers)
        kwargs["labels"] = None  # to prevent any loading of labels
        kwargs["path_remove"] = None  # XXX TODO: add index removal
        self.trainset, self.valset, self.testset = load_all_data(
            self.tasks_path, labels_path=None, **kwargs
        )
        self.input_dim = np.prod(self.trainset[0][0].shape).item()
        self.model = get_model(
            model,
            n_classes=n_classes,
            pretrained=pretrained,
            cifar="cifar" in tasks_path.lower(),
        )
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.n_workers = len(self.converter.table_worker)
        self.output_name = output_name
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.crowdlayer_net = Crowdlayer_net(
            self.n_classes,
            self.n_workers,
            self.model,
            self.scale,
            self.criterion,
        )
        self.optimizer, self.scheduler = get_optimizer(
            self.crowdlayer_net.classifier, optimizer, **kwargs
        )
        kwargs[
            "use_parameters"
        ] = False  # disable parameters for the optimizer
        self.optimizer2, self.scheduler2 = get_optimizer(
            self.crowdlayer_net.confusion, optimizer, **kwargs
        )
        kwargs["use_parameters"] = True
        self.setup(**kwargs)

    def setup(self, **kwargs):
        # get correct training labels
        targets, ll = [], []
        self.numpyans = reformat_labels(self.answers, self.n_workers)
        for i, samp in enumerate(self.trainset.samples):
            img, true_label = samp
            num = int(img.split("-")[-1].split(".")[0])
            ll.append((img, self.numpyans[num]))
            targets.append(self.numpyans[num])
        self.trainset.samples = ll
        self.trainset.targets = targets

        self.trainloader, self.testloader = DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=kwargs["batch_size"],
            num_workers=kwargs["num_workers"],
            pin_memory=(torch.cuda.is_available()),
        ), DataLoader(
            self.testset,
            batch_size=kwargs["batch_size"],
        )
        print(f"Train set: {len(self.trainloader.dataset)} tasks")
        print(f"Test set: {len(self.testloader.dataset)} tasks")
        self.valloader = DataLoader(
            self.valset,
            batch_size=kwargs["batch_size"],
        )
        print(f"Validation set: {len(self.valloader.dataset)} tasks")

    def run(self, **kwargs):
        from peerannot.runners.train import evaluate

        self.crowdlayer_net = self.crowdlayer_net.to(DEVICE)
        path_best = self.tasks_path / "best_models"
        path_best.mkdir(exist_ok=True)

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
        for epoch in tqdm(range(self.n_epochs), desc="Training epoch"):
            # train for one epoch
            logger = self.run_epoch(
                self.crowdlayer_net,
                self.trainloader,
                self.criterion,
                self.optimizer,
                self.optimizer2,
                logger,
            )

            # evaluate the self.conal_net if validation set
            if self.valset:
                logger = evaluate(
                    self.crowdlayer_net.classifier,
                    self.valloader,
                    self.criterion,
                    logger,
                    test=False,
                    n_classes=self.n_classes,
                )

                # save if improve
                if logger["val_loss"][-1] < min_val_loss:
                    torch.save(
                        {
                            "confusion": self.crowdlayer_net.confusion.state_dict(),
                            "classifier": self.crowdlayer_net.classifier.state_dict(),
                        },
                        path_best / f"{self.output_name}.pth",
                    )
                    min_val_loss = logger["val_loss"][-1]

            self.scheduler.step()
            self.scheduler2.step()
            if epoch in kwargs["milestones"]:
                print()
                print(
                    f"Adjusting learning rate to = {self.scheduler.optimizer.param_groups[0]['lr']:.4f}"
                )

        # load and test self.conal_net
        checkpoint = torch.load(path_best / f"{self.output_name}.pth")
        self.crowdlayer_net.classifier.load_state_dict(
            checkpoint["classifier"]
        )
        logger = evaluate(
            self.crowdlayer_net.classifier,
            self.testloader,
            self.criterion,
            logger,
            n_classes=int(self.n_classes),
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
        (self.tasks_path / "results").mkdir(parents=True, exist_ok=True)
        with open(
            self.tasks_path / "results" / f"{self.output_name}.json", "w"
        ) as f:
            json.dump(logger, f, indent=3, ensure_ascii=False)
        print(
            f"Results stored in {self.tasks_path / 'results' / f'{self.output_name}.json'}"
        )

    def run_epoch(
        self, model, trainloader, criterion, optimizer, optimizer2, logger
    ):
        model.train()
        total_loss = 0.0
        for inputs, labels in trainloader:
            # move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero out gradients
            optimizer.zero_grad()  # model.zero_grad() to be Xtra safe

            # compute the loss directly !!!!!
            loss = model(inputs, labels)

            # gradient step
            loss.backward()
            optimizer.step()
            optimizer2.step()
            total_loss += loss
        # log everything
        logger["train_loss"].append(total_loss.item())
        return logger
