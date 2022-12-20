"""
=============================
WAUM (2022)
=============================

Measures the WAUM per worker and task without duplication for each task by the number of workers that responded.
Once too prone to confusion tasks are removed, the final label is a
weighted distribution by the diagonal of the estimated confusion matrix.

Using:
- Margin estimation
- Trust score per worker and task
"""

from .template import CrowdModel
import pandas as pd
from .DS import Dawid_Skene as DS
import torch
from pathlib import Path
from torch.utils.data import Subset
from tqdm.auto import tqdm
from torchvision import transforms
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_json_to_pd(crowd_data):
    data_ = {"task": [], "worker": [], "label": []}
    for task, all_ in crowd_data.items():
        for rev, ans in all_.items():
            data_["task"].append(task)
            data_["worker"].append(rev)
            data_["label"].append(ans)
    data_ = pd.DataFrame(data_)
    return data_


class WAUM(CrowdModel):
    def __init__(
        self,
        tasks,
        answers,
        n_classes,
        model,
        criterion,
        optimizer,
        n_epoch,
        verbose=False,
        use_pleiss=False,
        **kwargs
    ):
        """Compute the WAUM score for each task using a stacked version of the dataset (stacked over workers)

        :param tasks: Dataset of tasks as
            (x_i, y_i^(j), y_i^*, i)_(i,j)
        :type tasks: torch Dataset
        :param answers: Dictionnary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int
        :param model: Neural network to use
        :type model: torch Module
        :param criterion: loss to minimize for the network
        :type criterion: torch loss
        :param optimizer: Optimization strategy for the minimization
        :type optimizer: torch optimizer
        :param n_epoch: Number of epochs
        :type n_epoch: int
        :param verbose: Print details in log, defaults to False
        :type verbose: bool, optional
        :param use_pleiss: Use Pleiss margin instead of Yang, defaults to False
        :type use_pleiss: bool, optional
        """

        super().__init__(answers)
        self.maxiterDS = kwargs.get("maxiterDS", 60)
        self.n_classes = n_classes
        self.n_workers = len(self.converter.table_worker)
        self.n_task = len(self.answers)
        self.model = model
        self.DEVICE = kwargs.get("DEVICE", DEVICE)
        self.optimizer = optimizer
        self.tasks = tasks
        self.criterion = criterion
        self.verbose = verbose
        self.use_pleiss = use_pleiss
        self.n_epoch = n_epoch
        self.initial_lr = self.optimizer.param_groups[0]["lr"]
        self.crowd_data = convert_json_to_pd(self.answers)
        self.checkpoint = {
            "epoch": n_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        self.path = Path("./temp/").mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, "./temp/checkpoint_waum.pth")

    def run_DS(self, cut=False):
        if not cut:
            self.ds = DS(self.answers, self.n_classes)
            self.ds.run(maxiter=self.maxiterDS)
        else:
            self.answers_waum = {
                key: val
                for key, val in self.answers.items()
                if key not in self.too_hard
            }
            self.ds = DS(self.answers_waum, self.n_classes)
            self.ds.run(maxiter=self.maxiterDS)

        self.pi = self.ds.pi

    def make_step(self, data_j, batch):
        """One optimization step

        :param batch: Batch of tasks
            Batch:
                - index 0: tasks (x_i)_i
                - index 1: labels
                - index 2: tasks index (i)_i
        :type batch: batch
        :return: Tuple with length, logits, targets, ground turths and index
        :rtype: tuple
        """

        xi = batch[0]
        truth = batch[2]
        idx = batch[3].tolist()
        all_ans = data_j[data_j["task"].isin(idx)]
        all_ans.index = all_ans["task"]
        all_ans = all_ans.reindex(idx)
        capture = np.where(~np.isnan(all_ans["task"]))[0]
        if len(capture) > 0:
            self.optimizer.zero_grad()
            labels = torch.tensor(all_ans["label"].values).type(torch.long)
            xi, labels, idx = (
                xi[capture],
                labels[capture],
                np.array(idx)[capture],
            )
            xi, labels = xi.to(self.DEVICE), labels.to(self.DEVICE)
            out = self.model(xi)
            # print(out, labels, len(capture))
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()
            len_ = len(idx)
            return len_, out, labels, idx, truth
        return None, None, None, None, None

    def get_aum(self):
        AUM_recorder = {
            "task": [],
            "worker": [],
            "label": [],
            "truth": [],
            "epoch": [],
            "label_logit": [],
            "label_prob": [],
            "other_max_logit": [],
            "other_max_prob": [],
            "secondlogit": [],
            "secondprob": [],
            "score": [],
        }
        workers = self.crowd_data["worker"].unique()
        for id_worker, j in (
            tqdm(enumerate(workers), total=len(workers), desc="workers")
            if self.verbose
            else enumerate(workers)
        ):
            data_j = self.crowd_data[self.crowd_data["worker"] == j]
            if data_j.empty:
                continue
            sub = Subset(
                self.tasks,
                [int(i) for i in list(data_j["task"].values)],
            )
            dl = torch.utils.data.DataLoader(
                sub, batch_size=50, worker_init_fn=0, shuffle=True
            )
            pij = (
                torch.tensor(self.pi[int(j)])
                .type(torch.FloatTensor)
                .to(self.DEVICE)
            )
            self.model.to(self.DEVICE)
            self.model.train()
            for epoch in (
                tqdm(range(self.n_epoch), desc="Epoch", leave=False)
                if self.verbose
                else range(self.n_epoch)
            ):
                for batch in dl:
                    len_, out, labels, idx, truth = self.make_step(
                        data_j, batch
                    )
                    if len_ is None:
                        continue
                    AUM_recorder["task"].extend(idx)
                    AUM_recorder["label"].extend(labels.tolist())
                    AUM_recorder["truth"].extend(truth.tolist())
                    AUM_recorder["worker"].extend([j] * len_)
                    AUM_recorder["epoch"].extend([epoch] * len_)

                    # s_y and P_y
                    if len_ > 1:
                        AUM_recorder["label_logit"].extend(
                            out.gather(1, labels.view(-1, 1))
                            .squeeze()
                            .tolist()
                        )
                        probs = out.softmax(dim=1)
                        AUM_recorder["label_prob"].extend(
                            probs.gather(1, labels.view(-1, 1))
                            .squeeze()
                            .tolist()
                        )
                    else:
                        AUM_recorder["label_logit"].extend(
                            out.gather(1, labels.view(-1, 1))
                            .squeeze(0)
                            .tolist()
                        )
                        probs = out.softmax(dim=1)
                        AUM_recorder["label_prob"].extend(
                            probs.gather(1, labels.view(-1, 1))
                            .squeeze(0)
                            .tolist()
                        )

                    # (s\y)[1] and (P\y)[1]
                    masked_logits = torch.scatter(
                        out, 1, labels.view(-1, 1), float("-inf")
                    )
                    masked_probs = torch.scatter(
                        probs, 1, labels.view(-1, 1), float("-inf")
                    )
                    (
                        other_logit_values,
                        other_logit_index,
                    ) = masked_logits.max(1)
                    (
                        other_prob_values,
                        other_prob_index,
                    ) = masked_probs.max(1)
                    if len(other_logit_values) > 1:
                        other_logit_values = other_logit_values.squeeze()
                        other_prob_values = other_prob_values.squeeze()
                    AUM_recorder["other_max_logit"].extend(
                        other_logit_values.tolist()
                    )
                    AUM_recorder["other_max_prob"].extend(
                        other_prob_values.tolist()
                    )

                    # s[2] ans P[2]
                    second_logit = torch.sort(out, axis=1)[0][:, -2]
                    second_prob = torch.sort(probs, axis=1)[0][:, -2]
                    AUM_recorder["secondlogit"].extend(second_logit.tolist())
                    AUM_recorder["secondprob"].extend(second_prob.tolist())
                    for ll in range(len_):
                        AUM_recorder["score"].append(
                            self.get_psuccess(probs[ll], pij).cpu().numpy()
                        )
            self.reset()
        self.reset()
        self.AUM_recorder = pd.DataFrame(AUM_recorder)
        recorder2 = self.AUM_recorder.copy()
        for task in (
            tqdm(recorder2.task.unique())
            if self.verbose
            else recorder2.task.unique()
        ):
            tmp = recorder2[recorder2.task == task]
            for j in tmp.worker.unique():
                recorder2.loc[
                    recorder2[
                        (recorder2.task == task) & (recorder2.worker == j)
                    ].score.index,
                    "score",
                ] = tmp[
                    (tmp.worker == j) & (tmp.epoch == self.n_epoch - 1)
                ].score.values[
                    0
                ]
        self.AUM_recorder = recorder2

    def reset(self):
        check_ = torch.load("./temp/checkpoint_waum.pth")
        self.n_epoch = check_["epoch"]
        self.model.load_state_dict(self.checkpoint["model"])
        self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        self.optimizer.param_groups[0]["lr"] = self.initial_lr

    def get_psuccess(self, probas, pij):
        with torch.no_grad():
            return probas @ torch.diag(pij)

    def get_psi1_waum(self):
        aum_df = self.AUM_recorder
        dico_cpt_aum = {}
        aum_df["margin"] = np.array(aum_df["label_prob"]) - np.array(
            aum_df["other_max_prob"]
        )
        unique_task = np.unique(np.array(aum_df["task"]))
        aum_per_worker = {}
        score_per_worker = {}
        for i, each_task in (
            tqdm(
                enumerate(unique_task),
                total=len(unique_task),
                desc="computing WAUM",
            )
            if self.verbose
            else enumerate(unique_task)
        ):
            aum_per_worker[each_task] = {}
            score_per_worker[each_task] = {}
            temp = aum_df[aum_df["task"] == each_task]
            avg = []
            score = []
            for j in np.unique(np.array(temp["worker"])):
                tempj = temp[temp["worker"] == j]
                aum_per_worker[each_task][j] = tempj["margin"].mean()
                avg.append((np.array(tempj["margin"]) * tempj["score"]).mean())
                score.append(tempj["score"].iloc[0])
                score_per_worker[each_task][j] = score[-1]
            dico_cpt_aum[each_task] = np.sum(avg) / sum(score)
        self.waum = dico_cpt_aum
        self.score_per_worker = score_per_worker
        self.aum_per_worker = aum_per_worker

    def get_psi5_waum(self):
        aum_df = self.AUM_recorder
        dico_cpt_aum = {}
        aum_df["margin"] = np.array(aum_df["label_prob"]) - np.array(
            aum_df["secondprob"]
        )
        unique_task = np.unique(np.array(aum_df["task"]))
        aum_per_worker = {}
        score_per_worker = {}
        for i, each_task in (
            tqdm(
                enumerate(unique_task),
                total=len(unique_task),
                desc="computing WAUM",
            )
            if self.verbose
            else enumerate(unique_task)
        ):
            aum_per_worker[each_task] = {}
            score_per_worker[each_task] = {}
            temp = aum_df[aum_df["task"] == each_task]
            avg = []
            score = []
            for j in np.unique(np.array(temp["worker"])):
                tempj = temp[temp["worker"] == j]
                if tempj.empty:
                    continue
                aum_per_worker[each_task][j] = tempj["margin"].mean()
                avg.append((np.array(tempj["margin"]) * tempj["score"]).mean())
                score.append(tempj["score"].iloc[0])
                score_per_worker[each_task][j] = score[-1]
            dico_cpt_aum[each_task] = (
                np.sum(avg) / sum(score) if sum(score) != 0 else 0
            )
        self.waum = dico_cpt_aum
        self.score_per_worker = score_per_worker
        self.aum_per_worker = aum_per_worker

    def cut_lowests(self, alpha=0.01):
        quantile = np.nanquantile(list(self.waum.values()), alpha)
        tasks_too_hard = [
            index
            for index in list(self.waum.keys())
            if self.waum[index] <= quantile
        ]
        self.quantile = quantile
        self.too_hard = tasks_too_hard
        return tasks_too_hard

    def run(self, alpha=0.01):
        """Run WAUM identification and label aggregation using the cut-off hyperparameter alpha

        :param alpha: WAUM quantile below which tasks are removed, defaults to 0.01
        :type alpha: float, optional
        """

        self.run_DS()
        self.ds1 = self.ds
        self.pi1 = self.ds1.pi
        self.get_aum()
        if not self.use_pleiss:
            self.get_psi5_waum()
        else:
            self.get_psi1_waum()
        self.cut_lowests(alpha)
        self.run_DS(cut=True)
        self.ds2 = self.ds
        self.pi2 = self.ds2.pi

    def get_probas(self):
        """Get soft labels distribution for each task

        :return: Weighted label frequency for each task
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        baseline = np.zeros((len(self.answers_waum), self.n_classes))
        self.answers_waum = dict(sorted(self.answers_waum.items()))
        for task_id, tt in enumerate(list(self.answers_waum.keys())):
            task = self.answers_waum[tt]
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

        :return: Hard labels
        :rtype: numpy.ndarray
        """
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
