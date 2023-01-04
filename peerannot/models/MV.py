"""
=========================
Majority voting
=========================
Most answered label per task
"""
from .template import CrowdModel
import numpy as np


class MV(CrowdModel):
    def __init__(self, answers, n_classes=2, **kwargs):
        """Majority voting strategy: most answered label

        :param answers: Dictionnary of workers answers with format
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
        self.original_answers = self.answers
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif

    def compute_baseline(self):
        """Compute label frequency per task"""
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline

    def get_answers(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        self.compute_baseline()
        ans = [
            np.random.choice(
                np.flatnonzero(self.baseline[i] == self.baseline[i].max())
            )
            for i in range(len(self.answers))
        ]
        self.ans = ans
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))

    def get_probas(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        self.compute_baseline()
        ans = [
            np.random.choice(
                np.flatnonzero(self.baseline[i] == self.baseline[i].max())
            )
            for i in range(len(self.answers))
        ]
        self.ans = ans
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))
