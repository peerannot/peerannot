from .template import CrowdModel
import numpy as np


class MV(CrowdModel):
    def __init__(self, answers, n_classes=2, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes

    def compute_baseline(self):
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline

    def get_answers(self):
        self.compute_baseline()
        ans = [
            np.random.choice(
                np.flatnonzero(self.baseline[i] == self.baseline[i].max())
            )
            for i in range(len(self.answers))
        ]
        return np.vectorize(self.converter.inv_labels.get)(
            np.array(ans)[self.converter.inv_task]
        )
