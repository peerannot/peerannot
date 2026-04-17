"""
=================================
Parent template to all strategies
=================================
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from peerannot.models.aggregation.types import AnswersDict, FilePathInput


class CrowdModel:
    def __init__(
        self,
        answers: AnswersDict,
        path_remove: FilePathInput = None,
    ) -> None:
        self.answers = answers
        transformed_answers = self.transform(answers)
        self.answers: dict[int, Any] = {
            int(k): v
            for k, v in sorted(
                transformed_answers.items(),
                key=lambda item: int(item[0]),
            )
        }

        self.path_remove: FilePathInput = path_remove
        if self.path_remove:
            self._exclude_answers()

    def _exclude_answers(self) -> None:
        answers_modif = {}
        if self.path_remove is not None:
            to_remove = np.loadtxt(self.path_remove, dtype=int)
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    answers_modif[i] = val
                    i += 1
            self.answers = answers_modif

    # TODO@jzftran: can we define array size?
    def get_tasks(self) -> npt.NDArray[Any]:
        tasks = np.array(list(self.answers.keys()))
        self.task_type = tasks.dtype
        return tasks

    def get_workers(self) -> npt.NDArray[Any]:
        return np.unique(
            np.array(
                [
                    el
                    for els in [list(j.keys()) for j in self.answers.values()]
                    for el in els
                ],
            ),
        )

    def get_labels(self) -> npt.NDArray[Any]:
        return np.unique(
            np.array(
                [
                    el
                    for els in [
                        list(j.values()) for j in self.answers.values()
                    ]
                    for el in els
                ],
            ),
        )

    def map_string(self) -> None:
        self.table_task = {val: i for i, val in enumerate(self.get_tasks())}
        self.table_worker = {
            val: i for i, val in enumerate(self.get_workers())
        }
        labs = self.get_labels()
        self.lab_type = labs.dtype
        if self.lab_type == "int":
            self.table_labels = {str(val): val for i, val in enumerate(labs)}
        else:
            self.table_labels = {val: i for i, val in enumerate(labs)}
        self.inv_transform()

    def inv_transform(self):
        if self.task_type == "int":
            self.inv_task = np.argsort(list(self.table_task.keys()))
        else:
            self.inv_task = np.arange(len(self.table_labels))
        self.inv_table_worker = {
            val: i for i, val in self.table_worker.items()
        }
        if self.lab_type == "int":
            self.inv_labels = {
                int(val): int(i) for i, val in self.table_labels.items()
            }
        else:
            self.inv_labels = {
                int(val): i for i, val in self.table_labels.items()
            }
        self.inv_labels[-1] = -1

    def check_index(self):
        keys = self.answers.keys()
        if set(map(type, keys)) == {int}:
            min_ = min(keys)
        else:
            min_ = min([int(x) for x in keys])
        if min_ > 0:
            self.recall = min_
        else:
            self.recall = 0

    def transform(self, answers: AnswersDict):
        # TODO@jzftran: checking if key is 'AI' makes the 'answers'
        # input type not uniform.  Should be moved to another class?
        all_ans = {}
        self.map_string()
        self.check_index()
        for task in answers:
            n_ = int(task) - self.recall
            all_ans[n_] = {}
            for key, value in answers[task].items():
                if key != "AI":
                    all_ans[n_][int(key)] = value
                else:
                    all_ans[n_]["AI"] = value
        return all_ans
