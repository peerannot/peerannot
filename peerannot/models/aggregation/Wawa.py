"""
=========================
Majority voting
=========================
Most answered label per task
"""

from ..template import CrowdModel

from ..aggregation.IterativeWawa import IterativeWawa
import numpy as np
from tqdm.auto import tqdm

# implementatio of
# https://github.com/Toloka/crowd-kit/blob/v1.2.1/crowdkit/aggregation/classification/wawa.py
# that takes into account large sparse datasets


class Wawa(IterativeWawa):
    def __init__(self, answers, n_classes=2, sparse=False, **kwargs):
        super().__init__(answers, n_classes, sparse, **kwargs)

    def run(self, **kwargs):
        super().run(1, **kwargs)

    def get_answers(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return np.array(self.ans)

    def get_probas(self):
        """Get labels obtained with majority voting aggregation

        :return: Most answered labels per task
        :rtype: numpy.ndarray
        """
        return self.get_answers()
