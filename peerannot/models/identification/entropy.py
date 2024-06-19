import numpy as np
from scipy.special import entr
from peerannot.models.aggregation.NaiveSoft import NaiveSoft as NS
from ..template import CrowdModel
from pathlib import Path


class Entropy(CrowdModel):
    """Computes the votes's distribution entropy per task."""

    def __init__(self, answers, **kwargs):
        """For each task, first compute the Naive Soft distribution. Then, obtain the entropy of the distribution.

        With enough votes, the higher the entropy, the more uncertain the task.

        .. math::

            \\mathrm{H}(i, \{y_i^{(j)}\}_j) = -\\sum_{k=1}^{K} p_k \\log(p_k) \\ŧext{with} p=\\mathrm{NS}(\{y_i^{(j)}\}_j)

        :param answers: Dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict

        The number of classes `n_classes` should be specified as keyword argument.
        """
        self.n_classes = kwargs["n_classes"]
        self.answers = answers

    def run(self, path):
        """Computes the entropy and save it for each task

        :param path: path to save the entropies. The file will be saved as `<path>/entropies.npy`
        :type path: str
        """
        ns = NS(self.answers, self.n_classes)
        labs = ns.get_probas()
        entropies = entr(labs).sum(1)
        filesave = Path(path).resolve() / "identification"
        filesave.mkdir(exist_ok=True, parents=True)
        filesave = filesave / "entropies.npy"
        np.save(
            filesave,
            entropies,
        )
        print(f"Entropies saved at {filesave}")
