import numpy as np
from peerannot.models.aggregation.DS import Dawid_Skene as DS
from ..template import CrowdModel
from pathlib import Path


class Spam_Score(CrowdModel):
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
