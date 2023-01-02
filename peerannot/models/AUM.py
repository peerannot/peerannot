"""
=============================
AUM (Pleiss et. al, 2020)
=============================

Measures the AUM per task given the ground truth label.

Using:
- Margin estimation
- Trust score per task
"""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Subset
from tqdm.auto import tqdm
from dataclasses import dataclass
import numpy as np
from typing import Union
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sample_identifier = Union[int, str]


class AUM:
    def __init__(
        self,
        tasks,
        n_classes,
        model,
        criterion,
        optimizer,
        n_epoch,
        verbose=False,
        use_pleiss=False,
        **kwargs,
    ):
        """Compute the AUM score for each task

        :param tasks: Dataset of tasks as
            (x_i, _, y_i^*, i)_(i,j)
        :type tasks: torch Dataset
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int
        :param model: Neural network to use
        :type model: torch Module
        :param criterion: loss to minimize for the network
        :type criterion: torch loss
        :param optimizer: Optimization strategy for the minimization
        :type optimizer: torch optimizer
        :param n_epoch: Number of epochs
        :type n_epoch: int
        :param verbose: Print details in log, defaults to False
        :type verbose: bool, optional
        :param use_pleiss: Use Pleiss margin instead of Yang, defaults to False
        :type use_pleiss: bool, optional
        """

        self.n_classes = n_classes
        self.model = model
        self.DEVICE = kwargs.get("DEVICE", DEVICE)
        self.optimizer = optimizer
        self.tasks = tasks
        self.criterion = criterion
        self.verbose = verbose
        self.use_pleiss = use_pleiss
        self.n_epoch = n_epoch
        self.initial_lr = self.optimizer.param_groups[0]["lr"]
        self.checkpoint = {
            "epoch": n_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self.filenames = np.array(
            [Path(samp[0]).name for samp in self.tasks.dataset.dataset.samples]
        )
        self.path = Path("./temp/").mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, "./temp/checkpoint_aum.pth")

    def make_step(self, batch):
        """One optimization step

        :param batch: Batch of tasks
            Batch:
                - index 0: tasks (x_i)_i
                - index 1: whatever
                - index 2: labels
                - index 3: tasks index (i)_i
        :type batch: batch
        :return: Tuple with length, logits, targets, ground turths and index
        :rtype: tuple
        """
        xi = batch[0]
        labels = batch[2]
        idx = batch[3].tolist()
        self.optimizer.zero_grad()
        xi, labels = xi.to(self.DEVICE), labels.to(self.DEVICE)
        out = self.model(xi)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()
        return out, labels, idx

    def get_aum(self):
        AUM_recorder = {
            "index": [],
            "task": [],
            "label": [],
            "epoch": [],
            "label_logit": [],
            "label_prob": [],
            "other_max_logit": [],
            "other_max_prob": [],
            "secondlogit": [],
            "secondprob": [],
        }
        for i in range(self.n_classes):
            AUM_recorder[f"logits_{i}"] = []
        self.model.to(self.DEVICE)
        self.model.train()
        for epoch in (
            tqdm(range(self.n_epoch), desc="Epoch", leave=False)
            if self.verbose
            else range(self.n_epoch)
        ):
            for batch in self.tasks:
                out, labels, idx = self.make_step(batch)
                len_ = len(idx)
                AUM_recorder["task"].extend(self.filenames[idx])
                AUM_recorder["index"].extend(idx)
                AUM_recorder["label"].extend(labels.tolist())
                AUM_recorder["epoch"].extend([epoch] * len_)
                # AUM_recorder["all_logits"].extend(out.tolist())
                # s_y and P_y
                AUM_recorder["label_logit"].extend(
                    out.gather(1, labels.view(-1, 1)).squeeze().tolist()
                )
                probs = out.softmax(dim=1)
                AUM_recorder["label_prob"].extend(
                    probs.gather(1, labels.view(-1, 1)).squeeze().tolist()
                )
                # (s\y)[1] and (P\y)[1]
                masked_logits = torch.scatter(
                    out, 1, labels.view(-1, 1), float("-inf")
                )
                masked_probs = torch.scatter(
                    probs, 1, labels.view(-1, 1), float("-inf")
                )
                (
                    other_logit_values,
                    other_logit_index,
                ) = masked_logits.max(1)
                (
                    other_prob_values,
                    other_prob_index,
                ) = masked_probs.max(1)
                if len(other_logit_values) > 1:
                    other_logit_values = other_logit_values.squeeze()
                    other_prob_values = other_prob_values.squeeze()
                AUM_recorder["other_max_logit"].extend(
                    other_logit_values.tolist()
                )
                AUM_recorder["other_max_prob"].extend(
                    other_prob_values.tolist()
                )

                # s[2] ans P[2]
                second_logit = torch.sort(out, axis=1)[0][:, -2]
                second_prob = torch.sort(probs, axis=1)[0][:, -2]
                AUM_recorder["secondlogit"].extend(second_logit.tolist())
                AUM_recorder["secondprob"].extend(second_prob.tolist())
                for cl in range(self.n_classes):
                    AUM_recorder[f"logits_{cl}"].extend(out[:, cl].tolist())
        self.AUM_recorder = pd.DataFrame(AUM_recorder)

    def compute_aum(self):
        data = self.AUM_recorder
        tasks = {
            "sample_id": [],
            "filename": [],
            "AUM_yang": [],
            "AUM_pleiss": [],
        }
        burn = 0

        for index in data["index"].unique():
            tmp = data[data["index"] == index]
            y = tmp.label.iloc[0]
            target_values = tmp.label_logit.values[burn:]
            logits = tmp.values[burn:, -self.n_classes :]
            llogits = np.copy(logits)
            _ = np.put_along_axis(
                logits, logits.argmax(1).reshape(-1, 1), float("-inf"), 1
            )
            masked_logits = logits
            other_logit_values, other_logit_index = masked_logits.max(
                1
            ), masked_logits.argmax(1)
            other_logit_values = other_logit_values.squeeze()
            other_logit_index = other_logit_index.squeeze()
            margin_values_yang = (target_values - other_logit_values).tolist()
            _ = np.put_along_axis(
                llogits,
                np.repeat(y, len(tmp)).reshape(-1, 1),
                float("-inf"),
                1,
            )
            masked_logits = llogits
            other_logit_values, other_logit_index = masked_logits.max(
                1
            ), masked_logits.argmax(1)
            other_logit_values = other_logit_values.squeeze()
            other_logit_index = other_logit_index.squeeze()
            margin_values_pleiss = (target_values - other_logit_values).mean()
            tasks["sample_id"].append(index)
            tasks["filename"].append(self.filenames[index])
            tasks["AUM_yang"].append(np.mean(margin_values_yang))
            tasks["AUM_pleiss"].append(np.mean(margin_values_pleiss))
        self.aums = pd.DataFrame(tasks)

    def run(self):
        """Run AUM identification"""
        self.get_aum()
        self.compute_aum()
