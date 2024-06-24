import numpy as np
from peerannot.models.aggregation.DS import Dawid_Skene as DS
from ..template import CrowdModel
from pathlib import Path


class Spam_Score(CrowdModel):
    """
    ======================================
    Spammer score (Raykar and Yu, 2011)
    ======================================

    Compute the distance between the confusion matrix of each worker and the closest rank-1 matrix. The closer to 0, it is likely the worker is a spammer.
    """

    def __init__(self, answers, **kwargs):
        """Compute the spammer score for each worker, the larger the sore, the more likely we can trust the worker. On the contrary, the closer to 0, the more likely the worker is a spammer.

        This is the Frobenius norm between the estimated confusion matrix :math:`\\hat{\\pi}^{(j)}` and the closest rank-1 matrix. Denote :math:`\\mathbf{e}` the vector of ones in :math:`\\mathbb{R}^K`.

        .. math::

            \\forall j\\in [n_\\texttt{worker}],\\ s_j = \\|\\pi^{(j)}- \\mathbf{e}u_j^\\top\|_F^2\enspace
             \\text{with } u_j = \\underset{u\\in\\mathbb{R}^K, u_j\\top \\mathbf{e}=1}{\\mathrm{argmin}} \\|\\pi^{(j)}- \\mathbf{e}u^\\top\|_F^2 \\enspace.

        Solving this problem and standardizing the result in :math:`[0,1]` gives the spammer score:

        .. math::

            \\forall j \in [n_\\texttt{worker}],\\ s_j = \\frac{1}{K(K-1)}\\sum_{1\\leq k<k'\\leq K}\\sum_{\\ell\\in[k]} (\\pi^{(j)}_{k,\\ell} - \\pi^{(j)}_{k',\\ell})^2 \\enspace.


        :param answers: Dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict

        The number of classes ``n_classes`` and the number of workers ``n_workers`` should be specified as keyword argument.
        If the matrices are known and stored in a ``npy`` or ``pth`` file, it can be specified as ``matrix_file``. Otherwise, the model will run the DS model to obtain the matrices.
        """
        self.n_classes = kwargs["n_classes"]
        self.answers = answers
        self.n_workers = kwargs["n_workers"]
        mf = kwargs["matrix_file"]
        if mf:
            if mf.suffix == "npy":
                self.matrices = np.load(mf)
            else:
                import torch

                self.matrices = torch.load(mf).numpy()
        else:
            print("Running DS model")
            ds = DS(self.answers, self.n_classes, n_workers=self.n_workers)
            ds.run()
            self.matrices = ds.pi

    def run(self, path):
        """Compute the spam score for each worker and save it at <path>/identification/spam_score.npy in a numpy array of size ``n_worker``.

        :param path: path to save the results
        :type path: str
        """
        spam = []
        for idx in range(self.n_workers):
            A = self.matrices[idx]
            spam.append(
                1
                / (self.n_classes * (self.n_classes - 1))
                * np.sum(((A[np.newaxis, :, :] - A[:, np.newaxis, :]) ** 2))
                / 2
            )

        filesave = Path(path).resolve() / "identification"
        filesave.mkdir(exist_ok=True, parents=True)
        filesave = filesave / "spam_score.npy"
        np.save(
            filesave,
            spam,
        )
        print(f"Spam scores saved at {filesave}")
