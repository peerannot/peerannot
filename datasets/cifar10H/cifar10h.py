import json
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import pooch
import zipfile
from pathlib import Path
from tqdm import tqdm


class CIFAR10H:
    def __init__(self):
        self.DIR = Path(__file__).parent.resolve()

    def setfolders(self):
        print(f"Loading data folders at {self.DIR}")
        train_path = self.DIR / "train"
        test_path = self.DIR / "test"
        valid_path = self.DIR / "val"
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        for cl in classes:
            for path in [train_path, test_path, valid_path]:
                (path / cl).mkdir(parents=True, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.CIFAR10(
            root=self.DIR, train=True, download=True, transform=transform
        )
        trainset = torchvision.datasets.CIFAR10(
            root=self.DIR, train=False, download=True, transform=transform
        )
        for i, (img, label) in tqdm(enumerate(trainset), total=len(trainset)):
            if i < 9500:
                save_image(
                    img,
                    train_path
                    / classes[label]
                    / (classes[label] + f"-{i}.png"),
                )
            else:
                save_image(
                    img,
                    valid_path
                    / classes[label]
                    / (classes[label] + f"-{i}.png"),
                )
        for i, (img, label) in tqdm(enumerate(testset), total=len(testset)):
            save_image(
                img,
                test_path / classes[label] / (classes[label] + f"-{i}.png"),
            )
        print("Created:")
        for set, path in zip(
            ("train", "val", "test"), [train_path, valid_path, test_path]
        ):
            print(f"- {set}: {path}")
        self.get_crowd_labels()
        print(f"Train crowd labels are in {self.DIR / 'answers.json'}")
        print(
            f"Train crowd labels (validation set) are in {self.DIR / 'answers_valid.json'}"
        )

    def get_crowd_labels(self):
        url = "https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-raw.zip?raw=true"
        filename = self.DIR / "downloads" / "cifar10h-raw.zip"
        filename.parent.mkdir(exist_ok=True)
        if not filename.exists():
            pooch.retrieve(url=url, known_hash=None, fname=filename)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(self.DIR / "downloads")

        csvfile = "cifar10h-raw.csv"
        df = pd.read_csv(self.DIR / "downloads" / csvfile, na_values="-9999")
        df = df[df.is_attn_check == 0]
        res_train, res_valid = {}, {}
        for t in df.cifar10_test_test_idx.unique():
            tmp = df[df.cifar10_test_test_idx == t]
            if t < 9500:
                res = res_train
            else:
                res = res_valid
            res[str(t)] = {}
            for w in tmp.annotator_id:
                res[str(t)][str(w)] = int(
                    tmp[tmp.annotator_id == w].chosen_label.iloc[0]
                )

        with open(self.DIR / "answers.json", "w") as answ:
            json.dump(res_train, answ)
        with open(self.DIR / "answers_valid.json", "w") as answval:
            json.dump(res_valid, answval)
