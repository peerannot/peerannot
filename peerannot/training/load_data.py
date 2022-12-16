from torchvision import datasets, transforms
import numpy as np


def load_data(path, path_labels=None, path_remove=None, **kwargs):
    img_size = kwargs.get("img_size", 224)
    data_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(path, transform=data_transforms)
    dataset.real_class_to_idx = {
        "plane": 0,
        "car": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    dataset.inv_class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
    if path_labels:
        labs = np.load(path_labels)
        ll = []
        targets = []
        imgs = []
        true_labels = []
        for samp in dataset.samples:
            img, true_label = samp
            true_label = dataset.real_class_to_idx[
                dataset.inv_class_to_idx[true_label]
            ]
            true_labels.append(true_label)
            num = int(img.split("-")[1].split(".")[0])
            ll.append((img, labs[num]))
            imgs.append(img)
            targets.append(labs[num])
        dataset.samples = ll
        dataset.imgs = imgs
        dataset.targets = targets
        dataset.true_labels = true_labels
        dataset.class_to_idx = dataset.real_class_to_idx
    if path_remove:
        rm_idx = np.load(path_remove)
        dataset.samples = [
            samp for i, samp in enumerate(dataset.samples) if i not in rm_idx
        ]
        dataset.targets = [
            tar for i, tar in enumerate(dataset.targets) if i not in rm_idx
        ]
        dataset.imgs = [
            im for i, im in enumerate(dataset.imgs) if i not in rm_idx
        ]
        dataset.true_labels = [
            im for i, im in enumerate(dataset.true_labels) if i not in rm_idx
        ]
    if path_labels:
        acc = (
            np.mean(np.array(dataset.targets) == np.array(dataset.true_labels))
            if np.array(dataset.targets).ndim == 1
            else np.mean(
                np.argmax(np.array(dataset.targets), axis=1)
                == np.array(dataset.true_labels)
            )
        )
        print(f"Accuracy on aggregation: {acc * 100:.3f}%")
    return dataset
