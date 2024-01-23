import numpy as np
from pathlib import Path
import json
import shutil

dir_toydata = Path(__file__).parents[1] / "datasets" / "toy-data"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


ANSWERS = load_json(dir_toydata / "answers.json")


def test_mv():
    from peerannot.models import MV

    mv = MV(ANSWERS, n_classses=2)
    y = mv.get_answers()
    expected = np.array([1, 0, 1])
    assert all([e == y_ for e, y_ in zip(expected, y)])


def test_ns():
    from peerannot.models import NaiveSoft as NS

    ns = NS(ANSWERS, n_classes=2)
    y = ns.get_probas()
    expected = np.array([[0.25, 0.75], [2 / 3, 1 / 3], [0, 1]])
    assert y.shape == (3, 2)
    assert np.isclose((expected - y).sum(), 0)


def test_ds():
    from peerannot.models import Dawid_Skene as DS

    expected = np.array([1, 0, 1])
    assert all([e == y_ for e, y_ in zip(expected, y)])

    ds = DS(ANSWERS, n_classes=2, n_workers=4)
    ds.run(maxiter=10)
    y = ds.get_probas()
    expected = np.array([[0.25, 0.75], [2 / 3, 1 / 3], [0, 1]])
    assert y.shape == (3, 2)
    assert np.isclose((expected - y).sum(), 0)


def test_wawa():
    from peerannot.models import Wawa

    wawa = Wawa(ANSWERS, n_classses=2, n_workers=4, sparse=True)
    wawa.run()
    y = wawa.get_answers()
    expected = np.array([1, 0, 1])
    assert np.isclose(
        (wawa.worker_score - np.array([1 / 2, 1, 1, 1 / 2])).sum(), 0
    )
    assert all([e == y_ for e, y_ in zip(expected, y)])


def test_glad():
    from peerannot.models import GLAD

    glad = GLAD(
        ANSWERS, n_classes=2, n_workers=4, dataset=dir_toydata / "temp"
    )
    glad.run()
    y = glad.get_answers()
    expected = np.array([1, 0, 1])
    assert y.shape == (3,)
    assert all([e == y_ for e, y_ in zip(expected, y)])
    assert glad.beta.shape == (3,)
    assert glad.alpha.shape == (4,)
    assert (
        len(
            list(
                (dir_toydata / "temp" / "identification" / "glad").glob(
                    "*.npy"
                )
            )
        )
        == 2
    )
    shutil.rmtree(dir_toydata / "temp")  # cleanup
