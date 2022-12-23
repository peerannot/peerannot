import numpy as np

# XXX TODO: clean this file and depend less on it


class Converter:
    def __init__(self, answers):
        self.answers = answers

    def get_tasks(self):
        tasks = np.array(list(self.answers.keys()))
        self.task_type = tasks.dtype
        return tasks

    def get_workers(self):
        return np.unique(
            np.array(
                [
                    el
                    for els in [list(j.keys()) for j in self.answers.values()]
                    for el in els
                ]
            )
        )

    def get_labels(self):
        uniques = np.unique(
            np.array(
                [
                    el
                    for els in [
                        list(j.values()) for j in self.answers.values()
                    ]
                    for el in els
                ]
            )
        )
        return uniques

    def map_string(self):
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

    def transform(self):
        all_ans = {}
        self.map_string()
        for task in self.answers:
            all_ans[int(task)] = {
                int(key): int(value)
                for key, value in self.answers[task].items()
            }
        return all_ans
