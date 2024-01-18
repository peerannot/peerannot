import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class BirdsAudio:
    def __init__(self):
        self.DIR = Path(__file__).parent.resolve()
        self.filename = (
            self.DIR / "bird_sound_training_data" / "letters" / "annotations.tsv"
        )
        self.user_expertise = self.DIR / "bird_sound_training_data" / "users.tsv"

    def setfolders(self):
        print(f"Loading data folders at {self.DIR}")
        self.get_crowd_labels()
        print(f"Train crowd labels are in {self.DIR / 'answers.json'}")

    def get_crowd_labels(self):
        self.data = pd.read_csv(self.filename, sep="\t")
        self.user_data = pd.read_csv(self.user_expertise, sep="\t")
        self.conv_tasks = {v: k for k, v in enumerate(set(self.data["candidate_id"]))}
        self.conv_workers = {
            v: k + 1 for k, v in enumerate(set(self.user_data["user_id"]))
        }
        is_expert = [
            1 if self.user_data.iloc[i]["birdwatching_activity_level"] == 4 else 0
            for i in range(len(self.user_data))
        ]
        answers = {k: {} for k in self.conv_tasks.values()}
        truth = [-1 for _ in range(len(self.conv_tasks))]
        for index, row in tqdm(self.data.iterrows()):
            answers[self.conv_tasks[row["candidate_id"]]][
                self.conv_workers[row["user_id"]]
            ] = int(row["annotation"])
            if is_expert[self.conv_workers[row["user_id"]] - 1]:
                truth[self.conv_tasks[row["candidate_id"]]] = int(row["annotation"])
        with open(self.DIR / "answers.json", "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=3)
        np.savetxt(self.DIR / "truth.txt", truth, fmt="%1i")
