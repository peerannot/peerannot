"""
===================================
Naive soft: Frequency distribution
===================================
"""
from .template import CrowdModel
import numpy as np
from peerannot.models.DS import Dawid_Skene


class WDS(CrowdModel):
    def __init__(self, answers, n_classes=2, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes
        self.n_workers = kwargs["n_workers"]
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif

    def run(self):
        ds = Dawid_Skene(
            self.answers, self.n_classes, n_workers=self.n_workers
        )
        ds.run()
        self.pi = ds.pi
        self.ds = ds

    def get_probas(self):
        """Get soft labels distribution for each task

        :return: Weighted label frequency for each task
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        baseline = np.zeros((len(self.answers), self.n_classes))
        self.answers = dict(sorted(self.answers.items()))
        for task_id, tt in enumerate(list(self.answers.keys())):
            task = self.answers[tt]
            for worker, vote in task.items():
                baseline[task_id, int(vote)] += self.pi[
                    self.ds.converter.table_worker[int(worker)]
                ][int(vote), int(vote)]
        self.baseline = baseline
        return np.where(
            baseline.sum(axis=1).reshape(-1, 1),
            baseline / baseline.sum(axis=1).reshape(-1, 1),
            0,
        )

    def get_answers(self):
        """Argmax of soft labels, in this case corresponds to a majority vote

        :return: Hard labels (majority vote)
        :rtype: numpy.ndarray
        """
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
