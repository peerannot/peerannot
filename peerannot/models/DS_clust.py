"""
====================================================================
Dawid and skene model with worker clustering (Imomura et. al 2018)
====================================================================

Assumptions:
- clusters of workers

Using:
- Variational inference

Estimating:
- One confusion matrix per cluster of workers
"""

from .template import CrowdModel
import numpy as np


class Dawid_Skene_clust(CrowdModel):
    def __init__(self, answers, n_classes, L=2, **kwargs):
        """Dawid and Skene model with clusterized confusion matrices using variational inference.

        :param answers: Dictionnary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes
        :type n_classes: int
        :param L: Number of clusters of workers, defaults to 2
        :type L: int, optional
        """
        super().__init__(answers)
        self.n_classes = n_classes
        self.n_workers = len(self.converter.table_worker)
        self.n_task = len(self.answers)
        self.L = L
        assert self.L <= self.n_workers, "L must be <= n_workers"

    def get_crowd_matrix(self):
        """Compute matrix of size (n_task, n_workers, n_classes) to
        store proposed votes
        """
        matrix = np.zeros((self.n_task, self.n_workers, self.n_classes))
        for task, ans in self.answers.items():
            for worker, label in ans.items():
                matrix[task, worker, label] += 1
        self.crowd_matrix = matrix

    def _is_chance_rate(self, theta):
        n = self.n_task
        K = self.n_classes
        sum_ = np.sum(theta, axis=0)
        for k in range(K):
            if int(sum_[k]) == n:
                return True
        return False

    def initialize_parameter(self, x, K, L, random=True, delta=1e-10):
        n = x.shape[0]
        m = x.shape[1]

        if random:
            theta = np.clip(
                np.einsum(
                    "ki->ik", np.einsum("ijk->ki", x) / np.einsum("ijk->i", x)
                )
                + np.random.normal(scale=0.1, size=[n, K]),
                0.0,
                1.0,
            )
            theta = np.einsum(
                "ki->ik", np.einsum("ik->ki", theta) / np.sum(theta, axis=1)
            )
        else:
            theta = np.einsum(
                "ki->ik", np.einsum("ijk->ki", x) / np.einsum("ijk->i", x)
            )

        x += delta
        pi = np.einsum(
            "mjk->jkm",
            np.einsum("ik,ijm->mjk", theta, x)
            / np.einsum("ik,ijm->jk", theta, x),
        )
        order = np.array([np.linalg.norm(pi[j], ord="nuc") for j in range(m)])
        sigma = np.array(
            sorted(
                np.c_[np.arange(m), order],
                key=lambda pair: pair[1],
                reverse=True,
            )
        )[:, 0].astype(dtype=int)
        J = np.array(
            [sigma[int(m * l / L) : int(m * (l + 1) / L)] for l in range(L)]
        )
        lambda_ = np.array(
            [(np.sum(pi[J[l]], axis=0)) * L / m for l in range(L)]
        )

        phi = np.zeros([m, L])
        for l in range(L):
            for j in J[l]:
                phi[j, l] = 1.0

        rho = np.einsum("ik->k", theta) / n
        tau = np.einsum("jl->l", phi) / m
        return theta, phi, rho, tau, lambda_

    def variational_update(
        self, x, theta, phi, rho, tau, lambda_, delta=1e-10
    ):
        theta_prime = np.exp(
            np.einsum("ijm,jl,lkm->ik", x, phi, np.log(lambda_ + delta))
            + np.log(rho + delta)
        )
        phi_prime = np.exp(
            np.einsum("ijm,ik,lkm->jl", x, theta, np.log(lambda_ + delta))
            + np.log(tau + delta)
        )
        theta = np.einsum(
            "ki->ik", theta_prime.T / np.sum(theta_prime.T, axis=0)
        )
        phi = np.einsum("lj->jl", phi_prime.T / np.sum(phi_prime.T, axis=0))
        return theta, phi

    def hyper_parameter_update(self, x, theta, phi):
        n = x.shape[0]
        m = x.shape[1]

        lambda_prime = np.einsum("ik,jl,ijm->mlk", theta, phi, x)
        lambda_ = np.einsum(
            "mlk->lkm", lambda_prime / np.sum(lambda_prime, axis=0)
        )

        rho = np.einsum("ik->k", theta) / n
        tau = np.einsum("jl->l", phi) / m

        return rho, tau, lambda_

    def elbo(self, x, theta, phi, rho, tau, lambda_, delta=1e-10):
        l = (
            np.einsum(
                "ik,jl,ijm,lkm->", theta, phi, x, np.log(lambda_ + delta)
            )
            + np.einsum("ik,k->", theta, np.log(rho + delta))
            + np.einsum("jl,l->", phi, np.log(tau + delta))
            - np.einsum("ik,ik->", theta, np.log(theta + delta))
            - np.einsum("jl,jl->", phi, np.log(phi + delta))
        )
        if np.isnan(l):
            print("theta = ", theta)
            print("phi = ", phi)
            print("rho = ", rho)
            print("tau = ", tau)
            print("Lambda = ", lambda_)
            raise ValueError("ELBO is Nan!")
        return l

    def convergence_condition(self, elbo_new, elbo_old, epsilon):
        if elbo_new - elbo_old < 0:
            return False
        elif elbo_new - elbo_old < epsilon:
            return True
        else:
            return False

    def one_iteration(self, x, K, L, epsilon=1e-4, random=False):
        theta, phi, rho, tau, lambda_ = self.initialize_parameter(
            x, K, L, random=random
        )
        l = -1e100
        while True:
            theta, phi = self.variational_update(
                x, theta, phi, rho, tau, lambda_
            )
            rho, tau, lambda_ = self.hyper_parameter_update(x, theta, phi)
            l_ = self.elbo(x, theta, phi, rho, tau, lambda_)
            if self.convergence_condition(l_, l, epsilon):
                break
            else:
                l = l_
        return theta, phi, lambda_, rho

    def run(self, epsilon=1e-4, maxiter=100):
        """Run variational inference for the worker-clusterized DS model

        :param epsilon: convergence tolerance between two elbo values, defaults to 1e-4
        :type epsilon: float, optional
        :param maxiter: Maximum number of iterations, defaults to 100
        :type maxiter: int, optional
        :return: hard labels, (confusion matrices, prevalence), number of iterations
        :rtype: tuple(
            np.ndarray(n_task, n_classes),
            tuple(
                np.ndarray(n_worker, n_task, n_task),
                np.ndarray(n_classes)
                ),
            int)
        """
        self.get_crowd_matrix()
        x = self.crowd_matrix
        K = self.n_classes
        L = self.L
        c = 1
        theta, phi, lambda_, rho = self.one_iteration(
            x, K, L, epsilon=epsilon, random=False
        )

        while self._is_chance_rate(theta):
            c += 1
            theta, phi, lambda_, rho = self.one_iteration(
                x, K, L, epsilon=epsilon, random=True
            )
            print("Drop into " + str(c) + "th chance rate!!")
            if c >= maxiter:
                break

        g_hat = np.argmax(theta, axis=1)
        pi_hat = lambda_[np.argmax(phi, axis=1)]
        self.y_hat = g_hat
        self.pi = pi_hat
        self.rho = rho
        self.c = c
        self.probas = theta
        return g_hat, [pi_hat, rho], c

    def get_probas(self):
        """Get soft labels distribution for each task

        :return: Estimated soft labels for each task
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        return self.probas

    def get_answers(self):
        """Argmax of soft labels

        :return: Hard labels
        :rtype: numpy.ndarray(n_task)
        """
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
