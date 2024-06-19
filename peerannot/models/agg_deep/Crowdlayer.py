import torch
from torch import nn
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from collections.abc import Iterable
from ..template import CrowdModel
from pathlib import Path
from tqdm.auto import tqdm
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
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


class Crowdlayer_net(nn.Module):
    """
    =====================================================
    Neural network classifier architecture for CrowdLayer
    =====================================================
    """

    def __init__(
        self,
        n_class,
        n_annotator,
        classifier,
    ):
        """Architecture for the CrowdLayer network

        :param n_class: Number of classes
        :type n_class: int
        :param n_annotator: Number of workers
        :type n_annotator: int
        :param classifier: Neural network classifier backbone
        :type classifier: nn.Module
        """
        super().__init__()

        self.classifier = classifier
        self.n_worker = n_annotator
        self.n_classes = n_class
        self.workers = [torch.eye(n_class) for _ in range(self.n_worker)]
        self.confusion = nn.Parameter(torch.stack(self.workers), requires_grad=True)

    def forward(self, x):
        """Computes the backbone prediction and multiplies it with local confusion weights."""
        z_pred = self.classifier(x).softmax(1)
        ann_pred = torch.einsum("ik,jkl->ijl", z_pred, self.confusion)
        return ann_pred


class Crowdlayer(CrowdModel):
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
        output_name="crowdlayer",
        **kwargs,
    ):
        """CrowdLayer deep learning strategy.
        Learn a classifier with crowdsourced labels by modeling worker-specific confusions.

        During training, given the classifier :math:`\\mathcal{C}` with output scores :math:`z_i`, local confusions :math:`\\pi^{(j)}`,  the model computes the following:

        .. math::

            h_i^{(j)} = \\sigma((\\pi^{(j)})\sigma (z_i )),

        The final loss is the crossentropy between the worker-specific prediction and the given label.

        .. math::

            \\mathcal{L} = \\frac{1}{n_{\\texttt{task}}}\\sum_{i=1}^{n_{\\texttt{task}}}\\sum_{j\\in\\mathcal{A}(x_i)} \\mathrm{CE}(h_i^{(j)}, y_i^{(j)}).

        :param tasks_path: Path to images to train from
        :type tasks_path: path
        :param answers: Path to answers (json format)

          .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: path
        :param model: Backbone classifier architecture: should be from `torchvision.models`
        :type model: string
        :param n_classes: Number of classes
        :type n_classes: int
        :param optimizer: Pytorch optimizer name (either sgd or Adam)
        :type optimizer: string
        :param n_epochs: Number of training epochs
        :type n_epochs: int
        :param scale: Regularization parameter in the loss, defaults to 1e-5 (Regularization not implemented yet)
        :type scale: float
        :param verbose: Verbosity level, defaults to True
        :type verbose: bool, optional
        :param pretrained: Use the pretrained version of the backbone classifier, defaults to False
        :type pretrained: bool, optional
        :param output_name: Generated file prefix, defaults to "conal"
        :type output_name: str, optional

        The batch size, learning rate, scheduler and milestones can be specified as keyword arguments.
        Visit the `computo paper <https://computo.sfds.asso.fr/published-202402-lefort-peerannot/>`__ or the tutorial for examples.
        """
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
        self.answers_orig = self.answers

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
            freeze=kwargs.get("freeze", False),
        )
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.n_workers = kwargs["n_workers"]
        self.output_name = output_name
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        self.crowdlayer_net = Crowdlayer_net(
            self.n_classes,
            self.n_workers,
            self.model,
        )
        self.optimizer, self.scheduler = get_optimizer(
            self.crowdlayer_net.classifier, optimizer, **kwargs
        )
        kwargs["use_parameters"] = False  # disable parameters for the optimizer
        self.optimizer2, self.scheduler2 = get_optimizer(
            self.crowdlayer_net.confusion, optimizer, **kwargs
        )
        kwargs["use_parameters"] = True
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Create training, validation and test dataloaders from the dataset."""
        # get correct training labels
        targets, ll = [], []
        print(len(self.answers), len(self.trainset.samples))
        self.numpyans = reformat_labels(self.answers_orig, self.n_workers)
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
        """Train the CrowdLayer model and evaluate on the test set.

        Uses a gpu by default is `torch.cuda.is_available() is True`.

        Results are stored in `<tasks_path>/results/<output_name>.json` and the best model in `<tasks_path>/best_models/<output_name>.pth`. Results contain the train, validation and test loss as well as the validation and test accuracy.
        """

        from peerannot.runners.train import evaluate

        print(f"Running on {DEVICE}")

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
                            "confusion": self.crowdlayer_net.confusion,
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
        self.crowdlayer_net.classifier.load_state_dict(checkpoint["classifier"])
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
        with open(self.tasks_path / "results" / f"{self.output_name}.json", "w") as f:
            json.dump(logger, f, indent=3, ensure_ascii=False)
        print(
            f"Results stored in {self.tasks_path / 'results' / f'{self.output_name}.json'}"
        )

    def run_epoch(self, model, trainloader, criterion, optimizer, optimizer2, logger):
        """Run one epoch and monitor metrics"""
        model.train()
        total_loss = 0.0
        for inputs, labels in trainloader:
            # move to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # zero out gradients
            optimizer.zero_grad()  # model.zero_grad() to be Xtra safe
            optimizer2.zero_grad()

            # compute the loss directly !!!!!
            ann_pred = model(inputs)
            ann_pred = torch.reshape(ann_pred, (-1, self.n_classes))

            labels = labels.view(-1).type(torch.long)
            loss = criterion(ann_pred, labels)

            # gradient step
            loss.backward()
            optimizer.step()
            optimizer2.step()
            total_loss += loss
        # log everything
        logger["train_loss"].append(total_loss.item())
        return logger
