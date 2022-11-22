from .template import CrowdModel
import numpy as np


class Soft(CrowdModel):
    def __init__(self, answers, n_classes, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes

    def get_probas(self):
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline
        return (baseline / baseline.sum(axis=1).reshape(-1, 1))[
            self.converter.inv_task
        ]

    def get_answers(self):
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
