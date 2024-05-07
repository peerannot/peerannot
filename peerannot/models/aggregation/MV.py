from ..template import CrowdModel
import numpy as np
from tqdm.auto import tqdm


class MV(CrowdModel):
    """
    =========================
    Majority voting
    =========================
    Most answered label per task
    """

    def __init__(self, answers, n_classes=2, sparse=False, **kwargs):
        """Majority voting strategy: most answered label

        .. math::

            \mathrm{MV}(i, \mathcal{D}) = \\underset{k\in[K]}{\mathrm{argmax}} \sum_{j\in\mathcal{A}(x_i)}\mathbf{1}(y_i^{(j)} = k)


        :param answers: Dictionary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param sparse: If the number of workers/tasks/label is large (:math:`>10^{6}` for at least one), use sparse=True to run per task
        :type sparse: bool, optional
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional
        """

        super().__init__(answers)
        self.n_classes = n_classes
        self.sparse = sparse
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
        if not self.sparse:
            self.compute_baseline()
            ans = [
                np.random.choice(
                    np.flatnonzero(self.baseline[i] == self.baseline[i].max())
                )
                for i in range(len(self.answers))
            ]
        else:  # sparse problem
            ans = -np.ones(len(self.answers))
            for task_id in tqdm(self.answers.keys()):
                task = self.answers[task_id]
                count = np.bincount(np.array(list(task.values())))
                ans[int(task_id)] = np.random.choice(
                    np.flatnonzero(count == count.max())
                )
        self.ans = ans
        return np.vectorize(self.converter.inv_labels.get)(np.array(ans))

    def get_probas(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return self.get_answers()
