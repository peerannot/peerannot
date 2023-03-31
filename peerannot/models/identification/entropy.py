import numpy as np
from scipy.special import entr
from peerannot.models.aggregation.NaiveSoft import NaiveSoft as NS
from ..template import CrowdModel
from pathlib import Path


class Entropy(CrowdModel):
    def __init__(self, answers, **kwargs):
        self.n_classes = kwargs["n_classes"]
        self.answers = answers

    def run(self, path):
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
