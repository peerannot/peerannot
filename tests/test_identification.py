import numpy as np
from pathlib import Path
import json
import shutil

dir_toydata = Path(__file__).parents[1] / "datasets" / "toy-data"
dir_krippendorffdata = Path(__file__).parents[1] / "datasets" / "krippendorff-data"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


ANSWERS_TOY = load_json(dir_toydata / "answers.json")
ANSWERS_KRIPPENDORFF = load_json(dir_krippendorffdata / "answers.json")


def test_entropy():
    from peerannot.models import Entropy

    ent = Entropy(ANSWERS_TOY, n_classes=2)
    ent.run(dir_toydata)
    entropies = np.load(dir_toydata / "identification" / "entropies.npy")
    assert entropies.shape == (3,)
    assert entropies[2] == 0


def test_spamscore():
    from peerannot.models import Spam_Score

    spam = Spam_Score(ANSWERS_TOY, n_classes=2, n_workers=4, matrix_file=None)
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


def test_krippendorff():
    from peerannot.models import Krippendorff_Alpha

    krippendorff = Krippendorff_Alpha(ANSWERS_KRIPPENDORFF, n_classes=2, n_workers=3)
    krippendorff.run(dir_krippendorffdata)
    krippendorffAlpha = np.load(
        dir_krippendorffdata / "identification" / "krippendorff_alpha.npy"
    )
    print(krippendorffAlpha)
    assert krippendorffAlpha == 0.691358024691358
