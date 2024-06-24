import numpy as np
from peerannot.models.aggregation.DS import Dawid_Skene as DS
from ..template import CrowdModel
from pathlib import Path


class Trace_confusion(CrowdModel):
    """Use Dawid and Skene confusion matrices to obtain a scalar indicator of the worker's confusion"""

    def __init__(self, answers, **kwargs):
        """Get the trace of the confusion matrices for each worker. The closer to K (the number of classes), the better the worker.

        .. math::

            \\mathrm{Tr}_j = \\sum_{k=1}^{K} \\pi_{k,k}^{(j)}

        :param answers: Dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict

        The number of classes `n_{classes}` and the number of workers `n_{workers}` should be specified as keyword argument.
        If the matrices are known and stored in a `npy` or `pth` file, it can be specified as `matrix_file`. Otherwise, the model will run the DS model to obtain the matrices.
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
        """From workers confusion matrices, compute the trace and save it.

        :param path: path to save the results <path>/identification/trace_confusion.npy
        :type path: str
        """
        trace = np.trace(self.matrices, axis1=1, axis2=2)
        filesave = Path(path).resolve() / "identification"
        filesave.mkdir(exist_ok=True, parents=True)
        filesave = filesave / "traces_confusion.npy"
        np.save(
            filesave,
            trace,
        )
        print(f"Traces saved at {filesave}")
