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

from ..template import CrowdModel
import numpy as np
from tqdm.auto import tqdm
import warnings


class Dawid_Skene(CrowdModel):
    def __init__(self, answers, n_classes, sparse=False, **kwargs):
        super().__init__(answers)
        self.n_classes = n_classes
        self.n_workers = kwargs["n_workers"]
        self.sparse = sparse
        if kwargs.get("path_remove", None):
            to_remove = np.loadtxt(kwargs["path_remove"], dtype=int)
            self.answers_modif = {}
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    self.answers_modif[i] = val
                    i += 1
            self.answers = self.answers_modif

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
            pi: number of times worker k records l when j is correct
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
        if not self.sparse:
            self.get_crowd_matrix()
            self.init_T()
            ll = []
            k, eps = 0, np.inf
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
        else:
            self.run_sparse(epsilon, maxiter, verbose)

    def get_answers(self):
        if self.sparse:
            return np.vectorize(self.converter.inv_labels.get)(
                self.T.argmax(axis=1)
            )
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )

    def get_probas(self):
        if self.sparse:
            warnings.warn("Sparse implementation only returns hard labels")
            return self.get_answers()
        return self.T

    def run_sparse(self, epsilon=1e-6, maxiter=50, verbose=False):
        pass
        # from scipy.sparse import coo_array
        # import scipy.sparse as sp

        # labels_row, labels_col, labels_data = [], [], []
        # for task, ans in self.answers.items():
        #     for worker, label in ans.items():
        #         labels_row.append(int(task))
        #         labels_col.append(int(worker))
        #         labels_data.append(int(label))
        # labels = sp.coo_array(
        #     (labels_data, (labels_row, labels_col)),
        #     shape=(len(self.answers), self.n_workers),
        # )

        # self.pi = sp.rand(self.n_classes, self.n_classes)
        # self.p = sp.rand(self.n_workers, self.n_classes)
        # for _ in range(maxiter):
        #     # E-step
        #     T = sp.coo_array((labels.shape[0], self.n_classes))
        #     for j in range(self.n_classes):
        #         pj = self.p[:, j]
        #         pij = self.pi[j, :]

        #         # Calculate the likelihood
        #         likelihood = labels @ np.log(pij)

        #         # Calculate the denominator for normalization
        #         denom = labels @ np.log(pj)
        #         denom = np.exp(denom).sum(axis=1)

        #         # Calculate the posterior probability
        #         posterior = np.exp(likelihood) / denom.reshape(-1, 1)
        #         T[:, j] = posterior
        #         # M-step
        #         self.p = np.array(T.sum(axis=0) / T.sum())
        #         self.pi = np.array(T.transpose().dot(T) / T.sum(axis=0))

        # self.crowd_matrix = tensor
        # T_data, T_row, T_col = [], [], []
        # for i in range(len(self.crowd_matrix)):
        #     for k in range(self.n_classes):
        #         data = self.crowd_matrix[k].sum(axis=1)
        #         if len(nonzero) > 0:
        #             T_col.extend([k] * nn)
        #     ss = sum(data)
        #     if ss > 0:
        #         T_data.extend(data[non_zero] / ss)
        #     else:
        #         T_data.extend([0] * self.n_classes)
        #     T_row.extend([i] * len(non_zero))
        # self.T = coo_array(
        #     (T_data, (T_row, T_col)), shape=(len(self.answers), self.n_classes)
        # )
        # ll = []
        # k, eps = 0, np.inf
        # pbar = tqdm(total=maxiter, desc="Dawid and Skene sparse")
        # while k < maxiter and eps > epsilon:
        #     # mstep
        #     self.crowd_matrix = sp.vstack(self.crowd_matrix)
        #     p = self.T.sum(axis=0) / self.n_task
        #     pi = sp.coo_array(
        #         (self.n_workers, self.n_classes, self.n_classes),
        #         dtype=np.float64,
        #     )
        #     for q in range(self.n_classes):
        #         # Calculate pij using sparse operations
        #         pij = self.T[:, q] @ self.crowd_matrix.T
        #         denom = np.asarray(pij.sum(axis=1)).flatten()
        #         denominator = np.where(denom <= 0, -1e9, denom)
        #         # Create a new COO matrix for pi
        #         rows, cols = pij.nonzero()
        #         data = pij.data / denominator[rows]
        #         pi += sp.coo_array(
        #             (data, (rows, cols)),
        #             shape=(self.n_workers, self.n_classes, self.n_classes),
        #         )
        #     self.p, self.pi = p, pi
        #     T = sp.coo_array((self.n_task, self.n_classes), dtype=np.float64)
        #     for i in range(self.n_task):
        #         for j in range(self.n_classes):
        #             num = (
        #                 np.prod(
        #                     self.pi[:, j, :].power(self.crowd_matrix[i, :, :])
        #                 )
        #                 * self.p[j]
        #             )
        #             T[i, j] = num
        #     # Calculate self.denom_e_step using sparse operations
        #     self.denom_e_step = T.sum(axis=1, keepdims=True)
        #     # Update T using sparse operations
        #     T = sp.coo_array(
        #         np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
        #     )
        #     self.T = T
        #     likeli = self.log_likelihood()
        #     ll.append(likeli)
        #     if len(ll) >= 2:
        #         eps = np.abs(ll[-1] - ll[-2])
        #     k += 1
        #     pbar.update(1)
        # else:
        #     pbar.set_description("Finished")
        # pbar.close()
        # self.c = k
        # if eps > epsilon and verbose:
        #     print(f"DS did not converge: err={eps}")
        # return ll, k
