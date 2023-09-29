"""
=============================
Krippendorff Alpha
=============================

Python implementation from Thomas Grill (https://github.com/grrrr/krippendorff-alpha)
"""
import numpy as np
from ..template import CrowdModel
from pathlib import Path


class Krippendorff_Alpha(CrowdModel):
    def nominal_metric(self,a, b):
        return a != b


    def interval_metric(self,a, b):
        return (a-b)**2


    def ratio_metric(self,a, b):
        return ((a-b)/(a+b))**2
    
    def __init__(self, answers, **kwargs):
        self.n_classes = kwargs["n_classes"]
        self.answers = answers
        self.n_workers = kwargs["n_workers"]
        self.metric = self.nominal_metric
    

    def run(self, path):

        # Create pair values to compare
        units = dict((it, list(d.values())) for it, d in self.answers.items() if len(d) > 1)  # units with pairable values

        n = sum(len(pv) for pv in units.values())
        if n == 0:
            raise ValueError("No items to compare.")

        np_metric = (np is not None) and (self.metric in (self.interval_metric, self.nominal_metric, self.ratio_metric))

        alpha = -1.
        Do = 0.
        for grades in units.values():
            gr = np.asarray(grades)
            Du = sum(np.sum(self.metric(gr, gri)) for gri in gr)
            Do += Du/float(len(grades)-1)
        Do /= float(n)

        if Do == 0:
            alpha = 1.
        
        if(alpha<0):
            De = 0.
            for g1 in units.values():
                d1 = np.asarray(g1)
                for g2 in units.values():
                    De += sum(np.sum(self.metric(d1, gj)) for gj in g2)
            De /= float(n*(n-1))

            alpha = 1.-Do/De if (Do and De) else 1.

        filesave = Path(path).resolve() / "identification"
        filesave.mkdir(exist_ok=True, parents=True)
        filesave = filesave / "krippendorff_alpha.npy"
        np.save(
            filesave,
            alpha,
        )
        print(f"alpha saved at {filesave}")