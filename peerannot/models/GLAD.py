"""
=============================
GLAD (Whitehill et. al 2009)
=============================

Each worker ability is modeled using one scalar.
Each task has a difficulty level represented as a positive scalar.
Knowing these coefficients, the probability to have the right answer is a sigmoid of their product.

Assumption:
- The errors are uniform over classes

Using:
- One scalar per task and worker (task difficulty and worker ability)
"""
# Adapted from https://github.com/notani/python-glad

from .template import CrowdModel
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
from tqdm.auto import tqdm


def sigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [
            lambda i: 1 / (1 + np.exp(-i)),
            lambda i: np.exp(i) / (1 + np.exp(i)),
        ],
    )


def logsigmoid(x):
    return np.piecewise(
        x,
        [x > 0],
        [
            lambda i: -np.log(1 + np.exp(-i)),
            lambda i: i - np.log(1 + np.exp(i)),
        ],
    )


class GLAD(CrowdModel):
    def __init__(
        self,
        answers,
        n_classes,
        **kwargs,
    ):
        """Aggregate labels with a bilinear trust score using a scalar per task indicating the difficulty and a scalar per worker indicating the worker ability

        :param answers: Dictionnary of workers answers with format
         .. code-block:: javascript

             {
                 task0: {worker0: label, worker1: label},
                 task1: {worker1: label}
             }
         :type answers: dict
         :param n_classes: Number of possible classes
         :type n_classes: int
        """
        super().__init__(answers)
        self.n_classes = n_classes
        self.n_workers = len(self.converter.table_worker)
        self.n_task = len(self.answers)

        self.labels = np.zeros((self.n_task, self.n_workers))
        for task, ans in self.answers.items():
            for worker, lab in ans.items():
                self.labels[task, worker] = lab + 1

        # Initialize Probs
        self.priorZ = np.array([1 / n_classes] * n_classes)
        self.priorAlpha = np.ones(self.n_workers)
        self.priorBeta = np.ones(self.n_task)
        self.probZ = np.empty((self.n_task, self.n_classes))
        self.beta = np.empty(self.n_task)
        self.alpha = np.empty(self.n_workers)

    def EM(self, epsilon, maxiter):
        """Infer true labels, tasks' difficulty and workers' ability"""
        # Initialize parameters to starting values
        print("- Running EM")
        self.alpha = self.priorAlpha.copy()
        self.beta = self.priorBeta.copy()
        self.probZ[:] = self.priorZ[:]

        pbar = tqdm(total=maxiter)
        self.EStep()
        lastQ = self.computeQ()
        self.MStep()
        Q = self.computeQ()
        counter = 1
        pbar.update(1)
        while abs((Q - lastQ) / lastQ) > epsilon and counter <= maxiter:
            lastQ = Q
            self.EStep()
            self.MStep()
            Q = self.computeQ()
            counter += 1
            pbar.update(1)
        else:
            pbar.set_description("Finished")
        pbar.close()
        # if abs((Q - lastQ) / lastQ) > epsilon:
        #     print(f"GLAD did not converge: err={abs((Q - lastQ) / lastQ)}")

    def calcLogProbL(self, item, *args):
        j = int(item[0])
        delta = args[0][j]
        # print(delta)
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)
        exponents = item[1:]
        correct = logsigmoid(exponents[delta]).sum()
        wrong = (
            logsigmoid(-exponents[oneMinusDelta])
            - np.log(float(self.n_classes - 1))
        ).sum()
        return correct + wrong

    def EStep(self):
        """Evaluate the posterior probability of true labels given observed labels and parameters"""

        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])
        ab = np.c_[np.arange(self.n_task), ab]

        for k in range(self.n_classes):
            self.probZ[:, k] = np.apply_along_axis(
                self.calcLogProbL,
                1,
                ab,
                (self.labels == k + 1),
                (self.labels == 0),
            )

        # Exponentiate and renormalize
        self.probZ = np.exp(self.probZ)
        s = self.probZ.sum(axis=1)
        self.probZ = (self.probZ.T / s).T

    def packX(self):
        return np.r_[self.alpha.copy(), self.beta.copy()]

    def unpackX(self, x):
        self.alpha = x[: self.n_workers].copy()
        self.beta = x[self.n_workers :].copy()

    def getBoundsX(self, alpha=(-100, 100), beta=(-100, 100)):
        alpha_bounds = np.array(
            [[alpha[0], alpha[1]] for i in range(self.n_workers)]
        )
        beta_bounds = np.array(
            [[beta[0], beta[1]] for i in range(self.n_workers)]
        )
        return np.r_[alpha_bounds, beta_bounds]

    def f(self, x):
        """Return the value of the objective function"""
        self.unpackX(x)
        return -self.computeQ()

    def df(self, x):
        """Return gradient vector"""
        self.unpackX(x)
        dQdAlpha, dQdBeta = self.gradientQ()
        return np.r_[-dQdAlpha, -dQdBeta]

    def MStep(self):
        initial_params = self.packX()
        params = sp.optimize.minimize(
            fun=self.f,
            x0=initial_params,
            method="CG",
            jac=self.df,
            tol=0.01,
            options={"maxiter": 25},
        )
        self.unpackX(params.x)

    def computeQ(self):
        """Calculate the expectation of the joint likelihood"""
        Q = 0
        Q += (self.probZ * np.log(self.priorZ)).sum()
        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])
        logSigma = logsigmoid(ab)
        idxna = np.isnan(logSigma)
        if np.any(idxna):
            logSigma[idxna] = ab[idxna]
        logOneMinusSigma = logsigmoid(-ab) - np.log(float(self.n_classes - 1))
        idxna = np.isnan(logOneMinusSigma)
        if np.any(idxna):
            logOneMinusSigma[idxna] = -ab[idxna]

        for k in range(self.n_classes):
            delta = self.labels == k + 1
            Q += (self.probZ[:, k] * logSigma.T).T[delta].sum()
            oneMinusDelta = (self.labels != k + 1) & (self.labels != 0)
            Q += (self.probZ[:, k] * logOneMinusSigma.T).T[oneMinusDelta].sum()
        Q += np.log(sp.stats.norm.pdf(self.alpha - self.priorAlpha)).sum()
        Q += np.log(sp.stats.norm.pdf(self.beta - self.priorBeta)).sum()
        if np.isnan(Q):
            return -np.inf
        return Q

    def dAlpha(self, item, *args):
        i = int(item[0])
        sigma_ab = item[1:]
        delta = args[0][:, i]
        noResp = args[1][:, i]
        oneMinusDelta = (~delta) & (~noResp)

        probZ = args[2]

        correct = (
            probZ[delta] * np.exp(self.beta[delta]) * (1 - sigma_ab[delta])
        )
        wrong = (
            probZ[oneMinusDelta]
            * np.exp(self.beta[oneMinusDelta])
            * (-sigma_ab[oneMinusDelta])
        )
        # Note: The derivative in Whitehill et al.'s appendix
        # has the term ln(K-1), which is incorrect.
        return correct.sum() + wrong.sum()

    def dBeta(self, item, *args):
        j = int(item[0])
        sigma_ab = item[1:]
        delta = args[0][j]
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)

        probZ = args[2][j]

        correct = probZ * self.alpha[delta] * (1 - sigma_ab[delta])
        wrong = probZ * self.alpha[oneMinusDelta] * (-sigma_ab[oneMinusDelta])

        return correct.sum() + wrong.sum()

    def gradientQ(self):

        dQdAlpha = -(self.alpha - self.priorAlpha)
        dQdBeta = -(self.beta - self.priorBeta)

        ab = np.array([np.exp(self.beta)]).T @ np.array([self.alpha])

        sigma = sigmoid(ab)
        sigma[np.isnan(sigma)] = 0

        labelersIdx = np.arange(self.n_workers).reshape((1, self.n_workers))
        sigma = np.r_[labelersIdx, sigma]
        sigma = np.c_[np.arange(-1, self.n_task), sigma]

        for k in range(self.n_classes):
            dQdAlpha += np.apply_along_axis(
                self.dAlpha,
                0,
                sigma[:, 1:],
                (self.labels == k + 1),
                (self.labels == 0),
                self.probZ[:, k],
            )

            dQdBeta += (
                np.apply_along_axis(
                    self.dBeta,
                    1,
                    sigma[1:],
                    (self.labels == k + 1),
                    (self.labels == 0),
                    self.probZ[:, k],
                )
                * np.exp(self.beta)
            )

        return dQdAlpha, dQdBeta

    def run(self, epsilon=1e-5, maxiter=50):
        """Run the label aggregation via EM algorithm

        :param epsilon: tolerance hyperparameter, relative change in likelihood, defaults to 1e-5
        :type epsilon: float, optional
        :param maxiter: Maximum number of iterations, defaults to 100
        :type maxiter: int, optional
        """
        self.EM(epsilon, maxiter)

    def get_probas(self):
        """Get soft labels distribution for each task

        :return: Soft labels
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        return self.probZ

    def get_answers(self):
        """Argmax of soft labels.

        :return: Hard labels
        :rtype: numpy.ndarray
        """
        return np.vectorize(self.converter.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )
