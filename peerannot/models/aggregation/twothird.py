"""
===================================
Two third agreement
===================================
"""
from ..template import CrowdModel
import numpy as np
import warnings
from pathlib import Path
from tqdm.auto import tqdm


class TwoThird(CrowdModel):
    def __init__(self, answers, n_classes=2, sparse=False, **kwargs):
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
        """
        super().__init__(answers)
        self.n_classes = n_classes
        self.sparse = sparse
        if kwargs.get("dataset", None):
            self.path_save = (
                Path(kwargs["dataset"]) / "identification" / "twothird"
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
            TwoThird agreement only returns hard labels.
            Defaulting to `get_answers()`.
            """
        )
        return self.get_answers()

    def get_answers(self):
        """Argmax of soft labels, in this case corresponds to a majority vote
        with two third consensus.
        If the consensus is not reached, a -1 is used as input.
        Additionally, if a `dataset` path is provided,
        tasks index with a -1 label are saved at
        `<dataset>/identification/twothird/too_hard.txt`

        CLI only: the <dataset> key is the shared input between aggregation
        strategies used as follows
           `peerannot aggregate <dataset> --answers answers.json -s <strategy>`

        :return: Hard labels and None when no consensus is reached
        :rtype: numpy.ndarray
        """
        if not self.sparse:
            baseline = np.zeros((len(self.answers), self.n_classes))
            for task_id in list(self.answers.keys()):
                task = self.answers[task_id]
                for vote in list(task.values()):
                    baseline[task_id, vote] += 1
            sum_ = baseline.sum(axis=1).reshape(-1, 1)
            self.baseline = baseline
            enough_votes = np.where(sum_ >= 2, 1, 0).flatten()
            ans = [
                np.random.choice(
                    np.flatnonzero(self.baseline[i] == self.baseline[i].max())
                )
                if enough_votes[i] == 1
                and self.baseline[i].max() / sum_[i] >= 2 / 3
                else -1
                for i in range(len(self.answers))
            ]
        else:  # sparse
            ans = -np.ones(len(self.answers))
            for task_id in tqdm(self.answers.keys()):
                task = self.answers[task_id]
                count = np.bincount(np.array(list(task.values())))
                n_votes = len(task)
                max_ = count.max()
                if n_votes >= 2 and max_ / n_votes >= 2 / 3:
                    ans[int(task_id)] = np.random.choice(
                        np.flatnonzero(count == max_)
                    )
        self.ans = ans
        if self.path_save:
            noconsensus = np.where(np.array(ans) == -1)[0]
            tab = np.ones((noconsensus.shape[0], 2))
            tab[:, 1] = noconsensus
            tab[:, 0] = -1
            if not self.path_save.exists():
                self.path_save.mkdir(parents=True, exist_ok=True)
            np.savetxt(self.path_save / "too_hard.txt", tab, fmt="%1i")
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))
