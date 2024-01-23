"""
=============================
WAUM (2022)
=============================

Measures the WAUM per worker and task by duplicating each task by the number
of workers that responded.
Once too prone to confusion tasks are removed, the final label is a
weighted distribution by the diagonal of the estimated confusion matrix.

Using:
- Margin estimation
- Trust score per worker and task
"""
from ..template import CrowdModel
import pandas as pd
from peerannot.models.aggregation.DS import Dawid_Skene as DS
import torch
from pathlib import Path
from tqdm.auto import tqdm
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
        topk=False,
        **kwargs
    ):
        """Compute the WAUM score for each task using a stacked version of the dataset (stacked over workers)

        :param tasks: Loader for dataset of tasks as
            (x_i, y_i^(j), w^(j), y_i^*, i)_(i,j)
        :type tasks: torch Dataset
        :param answers: Dictionary of workers answers with format
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
        self.n_workers = kwargs["n_workers"]
        self.n_task = len(self.answers)
        self.model = model
        self.DEVICE = kwargs.get("DEVICE", DEVICE)
        self.optimizer = optimizer
        self.tasks = tasks
        self.verbose = verbose
        self.criterion = criterion
        self.n_epoch = n_epoch
        self.use_pleiss = use_pleiss
        self.initial_lr = self.optimizer.param_groups[0]["lr"]
        self.crowd_data = convert_json_to_pd(self.answers)
        self.checkpoint = {
            "epoch": n_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if topk is False:
            self.topk = 1
        else:
            self.topk = int(topk)
        self.filenames = np.array(
            [
                Path(samp[0]).name
                for samp in self.tasks.dataset.dataset.base_samples
            ]
        )

        self.path = Path("./temp/").mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, "./temp/checkpoint_waum.pth")

    def run_DS(self, cut=False):
        if not cut:
            self.ds = DS(
                self.answers, self.n_classes, n_workers=self.n_workers
            )
            self.ds.run(maxiter=self.maxiterDS)
        else:
            self.answers_waum = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in self.too_hard[:, 1]:
                    self.answers_waum[i] = val
                    i += 1
            self.ds = DS(
                self.answers_waum, self.n_classes, n_workers=self.n_workers
            )
            self.ds.run(maxiter=self.maxiterDS)

        self.pi = self.ds.pi

    def make_step(self, batch):
        """One optimization step

        :param batch: Batch of tasks
            Batch:
                - index 0: tasks (x_i)_i
                - index 1: labels
                - index 2: worker
                - index 3: true index (witout redundancy)
                - index 4: tasks index (i)_i
        :type batch: batch
        :return: Tuple with length, logits, targets, workers, ground turths and index
        :rtype: tuple
        """
        xi, yy, ww, dd, idx = batch
        ww = list(map(int, ww.tolist()))
        dd = list(map(int, dd.tolist()))
        idx = list(map(int, idx.tolist()))
        if type(yy) is torch.Tensor:
            y = yy.type(torch.long)
        else:
            y = torch.Tensor(yy).type(torch.long)

        self.optimizer.zero_grad()
        xi, y = xi.to(self.DEVICE), y.to(self.DEVICE)
        out = self.model(xi)
        # print(out, labels, len(capture))
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        len_ = len(idx)
        return len_, out, y, ww, dd, idx

    def get_aum(self):
        AUM_recorder = {
            "index": [],
            "task": [],
            "worker": [],
            "label": [],
            "epoch": [],
            "label_logit": [],
            "label_prob": [],
            "other_max_logit": [],
            "other_max_prob": [],
            "secondlogit": [],
            "secondprob": [],
            "score": [],
        }
        pij = torch.tensor(self.pi).type(torch.FloatTensor).to(self.DEVICE)
        self.model.to(self.DEVICE)
        self.model.train()
        for epoch in (
            tqdm(range(self.n_epoch), desc="epoch")
            if self.verbose
            else range(self.n_epoch)
        ):
            for batch in self.tasks:
                len_, out, y, ww, dd, idx = self.make_step(batch)
                if len_ is None:
                    continue
                AUM_recorder["task"].extend(self.filenames[dd])
                AUM_recorder["index"].extend(dd)
                AUM_recorder["label"].extend(y.cpu().tolist())
                AUM_recorder["worker"].extend(ww)
                AUM_recorder["epoch"].extend([epoch] * len_)

                # s_y and P_y
                if len_ > 1:
                    AUM_recorder["label_logit"].extend(
                        out.gather(1, y.view(-1, 1)).squeeze().tolist()
                    )
                    probs = out.softmax(dim=1)
                    AUM_recorder["label_prob"].extend(
                        probs.gather(1, y.view(-1, 1)).squeeze().tolist()
                    )
                else:
                    AUM_recorder["label_logit"].extend(
                        out.gather(1, y.view(-1, 1)).squeeze(0).tolist()
                    )
                    probs = out.softmax(dim=1)
                    AUM_recorder["label_prob"].extend(
                        probs.gather(1, y.view(-1, 1)).squeeze(0).tolist()
                    )

                # (s\y)[1] and (P\y)[1]
                masked_logits = torch.scatter(
                    out, 1, y.view(-1, 1), float("-inf")
                )
                masked_probs = torch.scatter(
                    probs, 1, y.view(-1, 1), float("-inf")
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
                second_logit = torch.sort(out, axis=1)[0][:, -(self.topk + 1)]
                second_prob = torch.sort(probs, axis=1)[0][:, -(self.topk + 1)]
                AUM_recorder["secondlogit"].extend(second_logit.tolist())
                AUM_recorder["secondprob"].extend(second_prob.tolist())
                for ll in range(len_):
                    AUM_recorder["score"].append(
                        self.get_psuccess(probs[ll], pij[int(ww[ll])])
                        .cpu()
                        .numpy()
                    )
        self.AUM_recorder = pd.DataFrame(AUM_recorder)
        uni_ = self.AUM_recorder["index"].unique()
        for task in (
            tqdm(uni_, total=len(uni_), desc="Scores")
            if self.verbose
            else uni_
        ):
            workers = self.AUM_recorder.loc[
                self.AUM_recorder["index"] == task, "worker"
            ].unique()
            for j in workers:
                value = self.AUM_recorder.loc[
                    (self.AUM_recorder["index"] == task)
                    & (self.AUM_recorder["worker"] == j)
                    & (self.AUM_recorder["epoch"] == self.n_epoch - 1),
                    "score",
                ].values[0]
                self.AUM_recorder.loc[
                    (self.AUM_recorder["index"] == task)
                    & (self.AUM_recorder["worker"] == j)
                    & (self.AUM_recorder["epoch"] <= self.n_epoch - 1),
                    "score",
                ] = value
        self.AUM_recorder

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
        dico_cpt_aum = {"index": [], "task": [], "waum": []}
        aum_df["margin"] = np.array(aum_df["label_prob"]) - np.array(
            aum_df["other_max_prob"]
        )
        unique_task = np.unique(np.array(aum_df["index"]))
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
            temp = aum_df[aum_df["index"] == each_task]
            avg = []
            score = []
            for j in np.unique(np.array(temp["worker"])):
                tempj = temp[temp["worker"] == j]
                aum_per_worker[each_task][j] = tempj["margin"].mean()
                avg.append((np.array(tempj["margin"]) * tempj["score"]).mean())
                score.append(tempj["score"].iloc[0])
                score_per_worker[each_task][j] = score[-1]
            dico_cpt_aum["index"].append(each_task)
            dico_cpt_aum["task"].append(self.filenames[each_task])
            dico_cpt_aum["waum"].append((np.sum(avg) / sum(score)).item())
        self.waum = pd.DataFrame(dico_cpt_aum)
        self.score_per_worker = score_per_worker
        self.aum_per_worker = aum_per_worker

    def get_psi5_waum(self):
        aum_df = self.AUM_recorder
        dico_cpt_aum = {"index": [], "task": [], "waum": []}
        aum_df["margin"] = np.array(aum_df["label_prob"]) - np.array(
            aum_df["secondprob"]
        )
        unique_task = np.unique(np.array(aum_df["index"]))
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
            temp = aum_df[aum_df["index"] == each_task]
            avg = []
            score = []
            for j in np.unique(np.array(temp["worker"])):
                tempj = temp[temp["worker"] == j]
                aum_per_worker[each_task][j] = tempj["margin"].mean()
                avg.append((np.array(tempj["margin"]) * tempj["score"]).mean())
                score.append(tempj["score"].iloc[0])
                score_per_worker[each_task][j] = score[-1]
            dico_cpt_aum["index"].append(each_task)
            dico_cpt_aum["task"].append(self.filenames[each_task])
            dico_cpt_aum["waum"].append((np.sum(avg) / sum(score)).item())
        self.waum = pd.DataFrame(dico_cpt_aum)
        self.score_per_worker = score_per_worker
        self.aum_per_worker = aum_per_worker

    def cut_lowests(self, alpha=0.01):
        quantile = np.nanquantile(list(self.waum["waum"].to_numpy()), alpha)
        too_hard = self.waum[self.waum["waum"] <= quantile]
        self.index_too_hard = too_hard["index"].to_numpy()
        self.tasks_too_hard = [
            int(filename.split("-")[-1].split(".")[0])
            for filename in too_hard["task"].to_numpy()
        ]
        self.quantile = quantile
        self.too_hard = np.column_stack(
            (self.index_too_hard, self.tasks_too_hard)
        ).astype(int)

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

        :return: Weighted label frequency for each task in D_pruned
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        baseline = np.zeros((len(self.answers), self.n_classes))
        self.answers = dict(sorted(self.answers.items()))
        for task_id, tt in enumerate(list(self.answers.keys())):
            if tt not in self.too_hard[:, 1]:
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
        """Argmax of soft labels.

        :return: Hard labels
        :rtype: numpy.ndarray
        """

        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
