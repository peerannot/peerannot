"""
=========================
Majority voting
=========================
Most answered label per task
"""

from ..template import CrowdModel
import numpy as np
from tqdm.auto import tqdm


class IterativeWawa(CrowdModel):
    def __init__(self, answers, n_classes=2, sparse=False, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes
        self.sparse = sparse
        self.original_answers = self.answers
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

    def compute_baseline(self, weight=None):
        """Compute label frequency per task"""
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for worker, vote in task.items():
                baseline[task_id, vote] += weight[int(worker)]
        self.baseline = baseline

    def MV(self, weight=None):
        if not self.sparse:
            self.compute_baseline(weight)
            ans = [
                # np.random.choice(
                np.flatnonzero(self.baseline[i] == self.baseline[i].max())[0]
                # )
                for i in range(len(self.answers))
            ]
        else:  # sparse problem
            ans = -np.ones(len(self.answers), dtype=int)
            for task_id in tqdm(self.answers.keys()):
                task = self.answers[task_id]
                count = np.zeros(self.n_classes)
                for w, lab in task.items():
                    count[lab] += weight[int(w) - 1]
                ans[int(task_id)] = int(
                    # np.random.choice(
                    np.flatnonzero(count == count.max())[0]
                    # )
                )
        self.ans = ans

    def accuracy_by_mv(self):
        worker_score = np.zeros(self.n_workers)
        worker_n_ans = np.zeros(self.n_workers)
        for task_id, mv_lab in zip(self.answers.keys(), self.ans):
            for worker, lab in self.answers[task_id].items():
                if int(lab) == mv_lab:
                    worker_score[int(worker)] += 1
                worker_n_ans[int(worker)] += 1
        worker_score = np.divide(
            worker_score,
            worker_n_ans,
            out=np.zeros_like(worker_score),
            where=worker_n_ans != 0,
        )
        self.worker_score = worker_score

    def run(self, nb_iter=10, **kwargs):
        self.MV(weight=np.ones(self.n_workers))
        self.accuracy_by_mv()
        for _ in range(nb_iter):
            # print(self.worker_score)
            self.MV(weight=self.worker_score)
            self.accuracy_by_mv()

    def get_answers(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return np.array(self.ans)

    def get_probas(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return self.get_answers()
