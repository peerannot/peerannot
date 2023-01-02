"""
=====================================================
CoNAL (Common Noise Adaptation Layer), Chu et.al 2021
=====================================================
Implementation based from the unofficial repository
https://github.com/seunghyukcho/CoNAL-pytorch
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


class AuxiliaryNetwork(nn.Module):
    def __init__(self, x_dim, e_dim, w_dim):
        super().__init__()

        self.weight_v1 = nn.Linear(x_dim, 128)
        self.weight_v2 = nn.Linear(128, w_dim)
        self.weight_u = nn.Linear(e_dim, w_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x, e):
        v = self.weight_v1(x)
        v = self.weight_v2(v)
        v = f.normalize(v)
        u = self.weight_u(e)
        u = f.normalize(u)
        u = torch.transpose(u, 0, 1)
        w = torch.matmul(v, u)
        w = self.activation(w)
        return w


class NoiseAdaptationLayer(nn.Module):
    def __init__(self, n_class, n_annotator):
        super().__init__()

        self.global_confusion_matrix = nn.Parameter(
            torch.eye(n_class, n_class) * 2, requires_grad=True
        )
        self.local_confusion_matrices = nn.Parameter(
            torch.stack(
                [torch.eye(n_class, n_class) * 2 for _ in range(n_annotator)]
            ),
            requires_grad=True,
        )

    def forward(self, f, w):
        global_prob = torch.einsum(
            "ij,jk->ik", f, self.global_confusion_matrix
        )
        local_probs = torch.einsum(
            "ik,jkl->ijl", f, self.local_confusion_matrices
        )

        h = (
            w[:, :, None] * global_prob[:, None, :]
            + (1 - w[:, :, None]) * local_probs
        )

        return h


class CoNAL_net(nn.Module):
    def __init__(
        self,
        input_dim,
        n_class,
        n_annotator,
        classifier,
        annotator_dim,
        embedding_dim,
    ):
        super().__init__()

        self.auxiliary_network = AuxiliaryNetwork(
            input_dim, annotator_dim, embedding_dim
        )
        self.classifier = classifier
        self.noise_adaptation_layer = NoiseAdaptationLayer(
            n_class, n_annotator
        )

    def forward(self, x, annotator=None):
        f = self.classifier(x)
        if annotator is None:
            return f

        x_flatten = torch.flatten(x, start_dim=1)
        w = self.auxiliary_network(x_flatten, annotator)
        h = self.noise_adaptation_layer(f, w)

        return h, f


class CoNAL(CrowdModel):
    def __init__(
        self,
        tasks_path,
        answers,
        model,
        n_classes,
        optimizer,
        n_epochs,
        scale=1e-5,
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
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif

        kwargs["labels"] = None  # to prevent any loading of labels
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
        self.n_workers = kwargs["n_workers"]
        self.conal_net = CoNAL_net(
            self.input_dim,
            self.n_classes,
            self.n_workers,
            self.model,
            annotator_dim=self.n_workers,
            embedding_dim=20,
        )
        self.optimizer, self.scheduler = get_optimizer(
            self.conal_net, optimizer, **kwargs
        )
        self.output_name = output_name
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
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

        self.conal_net = self.conal_net.to(DEVICE)
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
                self.conal_net,
                self.trainloader,
                self.criterion,
                self.optimizer,
                logger,
            )

            # evaluate the self.conal_net if validation set
            if self.valset:
                logger = evaluate(
                    self.conal_net.classifier,
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
                            "auxiliary": self.conal_net.auxiliary_network.state_dict(),
                            "noise_adaptation": self.conal_net.noise_adaptation_layer.state_dict(),
                            "classifier": self.conal_net.classifier.state_dict(),
                        },
                        path_best / f"{self.output_name}.pth",
                    )
                    min_val_loss = logger["val_loss"][-1]

            self.scheduler.step()
            if epoch in kwargs["milestones"]:
                print()
                print(
                    f"Adjusting learning rate to = {self.scheduler.optimizer.param_groups[0]['lr']:.4f}"
                )

        # load and test self.conal_net
        checkpoint = torch.load(path_best / f"{self.output_name}.pth")
        self.conal_net.classifier.load_state_dict(checkpoint["classifier"])
        logger = evaluate(
            self.conal_net.classifier,
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

    def run_epoch(self, model, trainloader, criterion, optimizer, logger):
        model.train()
        total_loss = 0.0
        for inputs, labels in trainloader:
            # move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero out gradients
            model.zero_grad()  # model.zero_grad() to be Xtra safe

            annotator = torch.eye(self.n_workers).to(DEVICE)
            # logits
            ann_out, cls_out = model(inputs, annotator)

            # annotators loss
            ann_out = torch.reshape(ann_out, (-1, self.n_classes))
            labels = labels.view(-1)
            loss = criterion(ann_out, labels)

            # Regularization term
            confusion_matrices = model.noise_adaptation_layer
            matrices = (
                confusion_matrices.local_confusion_matrices
                - confusion_matrices.global_confusion_matrix
            )
            for matrix in matrices:
                loss -= self.scale * torch.linalg.norm(matrix)

            # gradient step
            loss.backward()
            optimizer.step()
            total_loss += loss
        # log everything
        logger["train_loss"].append(total_loss.item())
        return logger
