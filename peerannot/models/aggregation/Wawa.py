from ..aggregation.IWMV import IWMV
import numpy as np
import warnings


class Wawa(IWMV):
    """
    ===================================
    Worker Agreement With Aggregate
    ===================================
    """

    def __init__(self, answers, n_classes=2, sparse=False, **kwargs):
        """WAWA aggregation weighs each worker by their accuracy with a majority vote.

        .. math::

            \mathrm{WAWA}(i, \mathcal{D}) = \\underset{k\in[K]}{\mathrm{argmax}} \sum_{j\in\mathcal{A}(x_i)}\\beta_j \mathbf{1}(y_i^{(j)} = k)

            \\beta_j = \\frac{1}{|\{y_{i'}^{(j)}\}_{i'}|} \sum_{i'=1}^{n_{\\texttt{task}}} \mathbb{1}(y_{i'}^{(j)} = \mathrm{MV}(i', \{y_{i'}^{(j)}\}_j)


        :param answers: Dictionary of workers answers with format
        .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param sparse: If the number of workers/tasks/label is large (:math:`>10^{6}` for at least one), use sparse=True to run per task
        :type sparse: bool, optional
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional
        """
        super().__init__(answers, n_classes, sparse, **kwargs)

    def run(self, **kwargs):
        """Runs a single step of IWMV aggregation"""
        super().run(1, **kwargs)

    def get_answers(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return np.array(self.ans)

    def get_probas(self):
        """Get labels obtained with majority voting aggregation

        :raises Warning: WAWA aggregation only returns hard labels, using `get_answers()`
        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        warnings.warn(
            """
            Wawa aggregation only returns hard labels.
            Defaulting to ``get_answers()``.
            """
        )
        return self.get_answers()
