import torch
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
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
