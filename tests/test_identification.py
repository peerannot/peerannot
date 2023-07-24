import numpy as np
from pathlib import Path
import json
import shutil

dir_toydata = Path(__file__).parents[1] / "datasets" / "toy-data"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


ANSWERS = load_json(dir_toydata / "answers.json")


def test_entropy():
    from peerannot.models import Entropy

    ent = Entropy(ANSWERS, n_classes=2)
    ent.run(dir_toydata)
    entropies = np.load(dir_toydata / "identification" / "entropies.npy")
    assert entropies.shape == (3,)
    assert entropies[2] == 0


def test_spamscore():
    from peerannot.models import Spam_Score

    spam = Spam_Score(ANSWERS, n_classes=2, n_workers=4, matrix_file=None)
    spam.run(dir_toydata)
    spamscores = np.load(dir_toydata / "identification" / "spam_score.npy")
    assert spamscores.shape == (4,)
    assert all(
        [
            e == spamscores_
            for e, spamscores_ in zip(np.array([0, 1 / 4, 1, 1]), spamscores)
        ]
    )
    shutil.rmtree(dir_toydata / "identification")  # clean ups
