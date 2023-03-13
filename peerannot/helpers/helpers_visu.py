import json
import numpy as np
from pathlib import Path


def feedback_effort(votes_path):
    votes_path = Path(votes_path)
    assert votes_path.exists(), f"Votes file {votes_path} does not exist"
    with open(votes_path, "r") as votes:
        votes = json.load(votes)
    efforts = [len(aa) for aa in votes.values()]
    return efforts


def working_load(votes_path, metadata_path):
    votes_path = Path(votes_path)
    assert votes_path.exists(), f"Votes file {votes_path} does not exist"
    with open(votes_path, "r") as votes:
        votes = json.load(votes)
    metadata_path = Path(metadata_path)
    assert metadata_path.exists(), f"Votes file {metadata_path} does not exist"
    with open(metadata_path, "r") as metadata:
        metadata = json.load(metadata)
    workerload = np.zeros(metadata["n_workers"])
    for task, labs in votes.items():
        for worker in list(labs.keys()):
            workerload[int(worker)] += 1
    return workerload
