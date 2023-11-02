"""
===================================
PlantNet consensus
===================================
"""
from ..template import CrowdModel
import numpy as np
import warnings
from pathlib import Path
from tqdm.auto import tqdm


class PlantNet(CrowdModel):
    def __init__(
        self,
        answers,
        n_classes,
        AI="ignored",
        parrots="ignored",
        alpha=1,
        beta=1,
        IAweight=1,  # if AI is fixed
        authors=None,  # path to txt file containing authors id for each task
        **kwargs,
    ):
        """Two Third agreement: accept label reaching two third consensus

        :param answers: Dictionary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional
        :param AI: AI mode, defaults to "ignored" in (ignored, worker, fixed)
        :type AI: str, optional
        """
        self.AI = AI
        super().__init__(answers)
        self.n_workers = kwargs["n_workers"]
        self.parrots = parrots
        self.alpha = alpha
        self.beta = beta
        if self.AI == "ignored":
            for task in self.answers:
                self.answers[task] = {
                    k: v for k, v in self.answers[task].items() if k != "AI"
                }
            self.weight_AI = -1
        elif self.IA == "worker":
            self.n_workers += 1
            for task in self.answers:
                for worker, label in self.answers[task].items():
                    if worker == "AI":
                        self.answers[task][self.n_workers] = self.answers[
                            task
                        ]["AI"].pop(worker)
            self.weight_AI = -1
        elif self.IA == "fixed":
            self.weight_AI = IAweight
            ans_ai = -np.ones(len(self.answers))
            for i, task in enumerate(self.answers):
                for worker, label in self.answers[task].items():
                    if worker == "AI":
                        ans_ai[i] = label
                self.answers[task] = {
                    k: v for k, v in self.answers[task].items() if k != "AI"
                }
            self.ans_ai = ans_ai
        else:
            raise ValueError(
                f"Option {self.AI} should be one of worker, fixed or ignored"
            )
        self.n_classes = n_classes
        self.authors = authors
        if self.authors is None:
            self.authors = -np.ones(self.n_workers)
        else:
            self.authors = np.loadtxt(self.authors, dtype=int)
        if kwargs.get("dataset", None):
            self.path_save = (
                Path(kwargs["dataset"]) / "identification" / "plantnet"
            )
        else:
            self.path_save = None
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif

    def get_probas(self):
        warnings.warn(
            """
            PlantNet agreement only returns hard labels.
            Defaulting to `get_answers()`.
            """
        )
        return self.get_answers()

    def get_wmv(self, weights):
        yhat = -np.ones(self.n_task)
        if self.AI == "fixed":
            for i in range(self.n_task):
                init = np.zeros(self.n_classes)
                for worker, label in self.answers[i].items():
                    init[label] += weights[int(worker) - 1]
                if self.ans_ai[i] != -1:
                    init[self.ans_ai[i]] += self.weight_AI
                yhat[i] = np.argmax(init)
        else:
            for i in range(self.n_task):
                init = np.zeros(self.n_classes)
                for worker, label in self.answers[i].items():
                    init[label] += weights[int(worker) - 1]
                yhat[i] = np.argmax(init)
        return yhat

    def get_conf_acc(self, yhat, weights):
        conf = np.zeros(self.n_task)
        acc = np.zeros(self.n_task)
        for i in range(self.n_task):
            sum_weights = 0
            for worker, label in self.answers[i].items():
                sum_weights += weights[int(worker) - 1]
                conf[i] += weights[int(worker) - 1] * label == yhat[i]
            acc[i] = conf[i] / sum_weights
        return acc, conf

    def get_valid_tasks(self, valid, acc, conf):
        mask = np.where((conf > 2) & (acc > 0.7), 1, 0)
        valid[mask] = 1
        valid[~mask] = 0
        return valid

    def is_author(self, task, worker):
        if self.authors[int(task)] == int(worker):
            return 1
        else:
            return 1 / 10

    def get_weights(self):
        return self.n_j**self.alpha - self.n_j**self.beta + np.log(2.1)

    def get_n(self, valid, yhat):
        taxa_obs = np.zeros(self.n_workers)
        taxa_votes = np.zeros(self.n_workers)
        for (task_id, label_task) in zip(self.answers.keys(), yhat):
            for worker, lab_worker in self.answers[task_id].items():
                if lab_worker == label_task:
                    if self.is_author(task_id, worker):
                        if valid[int(task_id)]:
                            taxa_obs[int(worker) - 1] += 1
                    else:
                        taxa_votes[int(worker) - 1] += 1
        self.n_j = np.array(
            [
                taxa_obs[w] + np.round(taxa_votes[w])
                for w in range(self.n_workers)
            ]
        )

    def run(self, maxiter=100, epsilon=1e-5):  # epsilon = diff in weights
        self.n_task = len(self.answers)
        valid = np.ones(self.n_task)
        weights = np.ones(self.n_workers)
        # print("Begin WMV init")
        init_yhat = self.get_wmv(weights)
        # print("Begin acc, conf init")
        acc, conf = self.get_conf_acc(init_yhat, weights)
        valid = self.get_valid_tasks(valid, acc, conf)
        self.get_n(valid, init_yhat)
        for step in tqdm(range(maxiter)):
            n_j = self.n_j
            weights = self.get_weights()
            yhat = self.get_wmv(weights)
            acc, conf = self.get_conf_acc(init_yhat, weights)
            valid = self.get_valid_tasks(valid, acc, conf)
            self.get_n(valid, init_yhat)
            if (
                np.sum(np.abs(self.n_j - n_j)) / self.n_task <= epsilon
                and step > 5
            ):
                break
        self.labels_hat = yhat
        self.valid = valid
        self.weights = weights

    def get_answers(self):
        """
        :return: Hard labels and None when no consensus is reached
        :rtype: numpy.ndarray
        """
        ans = self.labels_hat
        if self.path_save:
            noconsensus = np.where(np.array(self.valid) == 0)[0]
            tab = np.ones((noconsensus.shape[0], 2))
            tab[:, 1] = noconsensus
            tab[:, 0] = -1
            if not self.path_save.exists():
                self.path_save.mkdir(parents=True, exist_ok=True)
            np.savetxt(self.path_save / "too_hard.txt", tab, fmt="%1i")
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))

    def get_probas(self):
        warnings.warn(
            """
            PlantNet agreement only returns hard labels.
            Defaulting to `get_answers()`.
            """
        )
        return self.get_answers()
