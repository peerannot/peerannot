import json
from pathlib import Path
import zipfile
import pandas as pd
import shutil
from urllib import request
import tarfile


class Music:
    def __init__(self):

        self.DIR = Path(__file__).parent.resolve()
        assert (
            self.DIR / "downloads" / "kaggle_dataset.zip"
        ).exists(), "Please download the archive dataset from kaggle at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download and rename it `kaggle_dataset.zip`"
        filename = self.DIR / "downloads" / "mturk" / "mturk-datasets.tar.gz"
        filename.parent.mkdir(exist_ok=True)
        if not filename.exists() or True:
            with request.urlopen(
                request.Request(
                    "http://fprodrigues.com/mturk-datasets.tar.gz",
                    headers={  # not a bot
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                ),
                timeout=60.0,
            ) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        f.write(response.read())
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=filename.parent)

        with zipfile.ZipFile(
            self.DIR / "downloads" / "kaggle_dataset.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(self.DIR / "downloads")

        self.DIRimages = self.DIR / "downloads" / "Data" / "images_original"
        self.DIRturk = (
            self.DIR / "downloads" / "mturk" / "music_genre_classification"
        )

    def setfolders(self):
        print(f"Loading data folders at {self.DIR}")
        # we will use the test data as validation (but not ideal)
        train_path = self.DIR / "train"
        test_path = self.DIR / "test"
        valid_path = self.DIR / "val"
        self.mturk_answers = pd.read_csv(self.DIRturk / "mturk_answers.csv")
        gold = pd.read_csv(self.DIRturk / "music_genre_gold.csv")
        gold_test = pd.read_csv(self.DIRturk / "music_genre_test.csv")
        gold = gold[["id", "class"]]
        gold_test = gold_test[["id", "class"]]

        child_dirs = [p for p in self.DIRimages.iterdir() if p.is_dir()]
        classes = [p.name for p in child_dirs]
        self.classes = classes
        for path in [train_path, valid_path, test_path]:
            for cl in classes:
                (path / cl).mkdir(exist_ok=True, parents=True)

        # move or copy files from directory
        self.task_conv = {}
        for index, row in gold.iterrows():
            dst_dir = self.DIR / "train"
            genre = row["class"]
            id_ = row["id"]
            file = (
                self.DIRimages
                / genre
                / id_.replace(".", "", 1).replace("mp3", "png")
            )
            self.task_conv[file.stem] = index
            file.rename(dst_dir / genre / f"{file.stem}-{index}.png")
        for index, row in gold_test.iterrows():
            dst_dir = self.DIR / "test"
            genre = row["class"]
            id_ = row["id"]
            file = (
                self.DIRimages
                / genre
                / id_.replace(".", "", 1).replace("mp3", "png")
            )
            if file.name == "jazz00054.png":
                # jazz00054 is known to be a corrupt wav file and thus is not translated as image : https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/discussion/158649
                continue
            file.rename(dst_dir / genre / f"{file.stem}-{index}.png")
            shutil.copy(
                dst_dir / genre / f"{file.stem}-{index}.png",
                self.DIR / "val" / genre / f"{file.stem}-{index}.png",
            )
        print("Created:")
        for set, path in zip(
            ("train", "val", "test"), [train_path, valid_path, test_path]
        ):
            print(f"- {set}: {path}")
        self.get_crowd_labels()
        print(f"Train crowd labels are in {self.DIR / 'answers.json'}")

    def get_crowd_labels(self):
        self.class_to_idx = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }
        res_train = {}
        workers = self.mturk_answers.WorkerID.unique()
        worker_conv = {k: v for k, v in zip(workers, range(len(workers)))}
        for index, task in self.mturk_answers.iterrows():
            file = task["Input.song"].replace(".", "", 1).replace("mp3", "png")
            worker = task["WorkerID"]
            lab = self.class_to_idx[task["Answer.pred_label"]]
            if not res_train.get(self.task_conv[Path(file).stem], None):
                res_train[self.task_conv[Path(file).stem]] = {}

            res_train[self.task_conv[Path(file).stem]][
                worker_conv[worker]
            ] = lab

        with open(self.DIR / "answers.json", "w") as answ:
            json.dump(res_train, answ, ensure_ascii=False, indent=3)
