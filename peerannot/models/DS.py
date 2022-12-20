"""
=============================
Dawid and skene model (1979)
=============================

Assumptions:
- independent workers

Using:
- EM algorithm

Estimating:
- One confusion matrix for each workers
"""

from .template import CrowdModel
import numpy as np
from tqdm.auto import tqdm


class Dawid_Skene(CrowdModel):
    def __init__(self, answers, n_classes, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes
        self.n_workers = len(self.converter.table_worker)
        self.n_task = len(self.answers)

    def get_crowd_matrix(self):
        matrix = np.zeros((self.n_task, self.n_workers, self.n_classes))
        for task, ans in self.answers.items():
            for worker, label in ans.items():
                matrix[task, worker, label] += 1
        self.crowd_matrix = matrix

    def init_T(self):
        T = self.crowd_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        self.T = np.where(tdim > 0, T / tdim, 0)

    def m_step(self):
        """Maximizing log likelihood (see eq. 2.3 and 2.4 Dawid and Skene 1979)

        Returns:
            p: (p_j)_j probabilities that instance has true response j if drawn
        at random (class marginals)
            pi: number of times worker k records l when j is correct / number
        of instandon’t think it will work as it’s not in the open-vsx registry. Extensions from the Microsoft VS Code extension marketplace only works on the Microsoft VS Codeces seen by worker k where j is correct
        """
        p = self.T.sum(0) / self.n_task
        pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))
        for q in range(self.n_classes):
            pij = self.T[:, q] @ self.crowd_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            pi[:, q, :] = pij / np.where(denom <= 0, -1e9, denom).reshape(
                -1, 1
            )
        self.p, self.pi = p, pi

    def e_step(self):
        """Estimate indicator variables (see eq. 2.5 Dawid and Skene 1979)
        Returns:
            T: New estimate for indicator variables (n_task, n_worker)
            denom: value used to compute likelihood easily
        """
        T = np.zeros((self.n_task, self.n_classes))
        for i in range(self.n_task):
            for j in range(self.n_classes):
                num = (
                    np.prod(
                        np.power(self.pi[:, j, :], self.crowd_matrix[i, :, :])
                    )
                    * self.p[j]
                )
                T[i, j] = num
        self.denom_e_step = T.sum(1, keepdims=True)
        T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
        self.T = T

    def log_likelihood(self):
        return np.log(np.sum(self.denom_e_step))

    def run(self, epsilon=1e-6, maxiter=50, verbose=False):
        self.get_crowd_matrix()
        self.init_T()
        ll = []
        k, eps = 0, 1e1
        pbar = tqdm(total=maxiter, desc="Dawid and Skene")
        while k < maxiter and eps > epsilon:
            self.m_step()
            self.e_step()
            likeli = self.log_likelihood()
            ll.append(likeli)
            if len(ll) >= 2:
                eps = np.abs(ll[-1] - ll[-2])
            k += 1
            pbar.update(1)
        else:
            pbar.set_description("Finished")
        pbar.close()
        self.c = k
        if eps > epsilon and verbose:
            print(f"DS did not converge: err={eps}")
        return ll, k

    def get_answers(self):
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )

    def get_probas(self):
        return self.T
