import json
from pathlib import Path
from download import download
import numpy as np


class LabelMe:
    def __init__(self):
        self.DIR = Path(__file__).parent.resolve()
        download(
            "http://fprodrigues.com/deep_LabelMe.tar.gz",
            self.DIR / "data",
            replace=False,
            kind="tar.gz",
        )

    def setfolders(self):
        print(f"Loading data folders at {self.DIR}")
        train_path = self.DIR / "train"
        test_path = self.DIR / "test"
        valid_path = self.DIR / "val"
        self.DIRdata = self.DIR / "data" / "LabelMe"
        child_dirs = [
            p
            for p in self.DIRdata.iterdir()
            if p.is_dir() and p.name != "prepared"
        ]
        classes = [p.name for p in child_dirs[0].iterdir() if p.is_dir()]
        for path in [train_path, valid_path, test_path]:
            for cl in classes:
                (path / cl).mkdir(exist_ok=True, parents=True)

        # move files from directory
        self.conv_task = {}
        for folder in ["train", "valid", "test"]:
            dst_dir = (
                self.DIR / folder if folder != "valid" else self.DIR / "val"
            )
            for i, file in enumerate((self.DIRdata / folder).glob("*/*")):
                if not file.is_dir():
                    parent = file.parent.name
                    self.conv_task[file.name] = i
                    file.rename(dst_dir / parent / f"{file.stem}-{i}.jpg")

        print("Created:")
        for set, path in zip(
            ("train", "val", "test"), [train_path, valid_path, test_path]
        ):
            print(f"- {set}: {path}")
        self.get_crowd_labels()
        print(f"Train crowd labels are in {self.DIR / 'answers.json'}")

    def get_crowd_labels(self):
        crowdlabels = np.loadtxt(self.DIRdata / "answers.txt")
        orig_name = np.loadtxt(self.DIRdata / "filenames_train.txt", dtype=str)
        convert_labels = {0: 2, 1: 3, 2: 7, 3: 6, 4: 1, 5: 0, 6: 4, 7: 5}
        res_train = {task: {} for task in range(crowdlabels.shape[0])}
        for id_, task in enumerate(crowdlabels):
            where = np.where(task != -1)[0]
            for worker in where:
                res_train[self.conv_task[orig_name[id_]]][
                    int(worker)
                ] = convert_labels[int(task[worker])]

        with open(self.DIR / "answers.json", "w") as answ:
            json.dump(res_train, answ, ensure_ascii=False, indent=3)
