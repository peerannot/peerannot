import numpy as np
from peerannot.models.aggregation.DS import Dawid_Skene as DS
from ..template import CrowdModel
from pathlib import Path


class Trace_confusion(CrowdModel):
    def __init__(self, answers, **kwargs):
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
        trace = np.trace(self.matrices, axis1=1, axis2=2)
        filesave = Path(path).resolve() / "identification"
        filesave.mkdir(exist_ok=True, parents=True)
        filesave = filesave / "traces_confusion.npy"
        np.save(
            filesave,
            trace,
        )
        print(f"Traces saved at {filesave}")
