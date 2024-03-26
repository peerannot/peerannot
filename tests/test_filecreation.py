import numpy as np
from pathlib import Path
import os
import os.path
import json
import shutil
import subprocess


dir_toydata = Path(__file__).parents[1] / "datasets" / "toy-data"


def verify_json(path):
    with open(path, "r") as f:
        answers = json.load(f)

    with open(dir_toydata / "answers.json", "r") as f:
        answers_sure = json.load(f)

    assert answers == answers_sure


def test_custom_creation():
    dir_train = dir_toydata / "train"
    dir_test = dir_toydata / "test"
    dir_val = dir_toydata / "val"
    dir_temp = dir_toydata / "temp"

    # Test the answers txt format (Rodrigues matrix format), with a test set and a val set
    os.makedirs(dir_temp)
    currentPath = Path(__file__).parents[1] / "datasets" / "customDataset.py"
    result = subprocess.run(
        [
            "peerannot",
            "install",
            Path(__file__).parents[1] / "datasets" / "customDataset.py",
            "--train-path",
            dir_train,
            "--test-path",
            dir_test,
            "--val-path",
            dir_val,
            "--answers",
            dir_toydata / "answers.txt",
            "--files-path",
            dir_toydata / "filenames.txt",
            "--label-names",
            dir_toydata / "labelNames.txt",
        ],
        capture_output=True,
        cwd=dir_temp,
    )
    nb_files = len([name for name in os.listdir(dir_temp)])
    assert nb_files == 6

    nb_labels = len([name for name in os.listdir(dir_temp / "train")])
    assert nb_labels == 2

    nb_data = len([name for name in os.listdir(dir_temp / "train" / "smiles")])
    assert nb_data == 2

    verify_json(dir_temp / "answers.json")
    shutil.rmtree(dir_temp)  # cleanup

    ############################################################################

    # Test the Peerannot answers format (task->worker), with a train and no val and test set
    os.makedirs(dir_temp)
    result = subprocess.run(
        [
            "peerannot",
            "install",
            Path(__file__).parents[1] / "datasets" / "customDataset.py",
            "--train-path",
            dir_train,
            "--answers-format",
            "1",
            "--files-path",
            dir_toydata / "filenames.txt",
            "--answers",
            dir_toydata / "answers.json",
            "--label-names",
            dir_toydata / "labelNames.txt",
        ],
        capture_output=True,
        cwd=dir_temp,
    )
    nb_files = len(
        [
            name
            for name in os.listdir(dir_temp)
            # if os.path.isfile(os.path.join(dir_temp, name))
        ]
    )
    assert nb_files == 4
    verify_json(dir_temp / "answers.json")
    shutil.rmtree(dir_temp)  # cleanup

    ############################################################################

    # Test the Peerannot answers format (task->worker), with a test, train and no val set
    os.makedirs(dir_temp)
    result = subprocess.run(
        [
            "peerannot",
            "install",
            Path(__file__).parents[1] / "datasets" / "customDataset.py",
            "--train-path",
            dir_train,
            "--test-path",
            dir_test,
            "--answers-format",
            "1",
            "--answers",
            dir_toydata / "answers.json",
            "--label-names",
            dir_toydata / "labelNames.txt",
            "--files-path",
            dir_toydata / "filenames.txt",
        ],
        capture_output=True,
        cwd=dir_temp,
    )
    verify_json(dir_temp / "answers.json")
    shutil.rmtree(dir_temp)  # cleanup

    ############################################################################

    # Test the Peerannot inverse answers format (worker->task), with no sets
    os.makedirs(dir_temp)
    result = subprocess.run(
        [
            "peerannot",
            "install",
            Path(__file__).parents[1] / "datasets" / "customDataset.py",
            "--answers-format",
            "2",
            "--answers",
            dir_toydata / "answers_inversed.json",
            "--no-task",
        ],
        capture_output=True,
        cwd=dir_temp,
    )
    nb_files = len(
        [
            name
            for name in os.listdir(dir_temp)
            # if os.path.isfile(os.path.join(dir_temp, name))
        ]
    )
    assert nb_files == 1

    verify_json(dir_temp / "answers.json")
    shutil.rmtree(dir_temp)  # cleanup
    # assert True == False
