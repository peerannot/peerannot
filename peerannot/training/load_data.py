from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


def load_data(path, path_labels=None, path_remove=None, **kwargs):
    img_size = kwargs.get("img_size", 224)
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if kwargs["data_augmentation"]:
        augmentations = transforms.RandomChoice(
            [
                transforms.RandomAffine(degrees=0, shear=15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomResizedCrop(img_size),
            ]
        )
        data_transforms.insert(0, augmentations)
    data_transforms = transforms.Compose(data_transforms)
    dataset = datasets.ImageFolder(path, transform=data_transforms)
    if "cifar" in str(path).lower():
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
    elif "labelme" in str(path).lower():
        dataset.real_class_to_idx = dataset.class_to_idx
    elif "music" in str(path).lower():
        dataset.real_class_to_idx = dataset.class_to_idx

    dataset.inv_class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
    if path_remove:
        path_remove = Path(path_remove).resolve()
        rm_idx = np.loadtxt(path_remove, dtype=int)[:, 0]
        flag_rm = False
        if rm_idx[0] == -1:  # path_remove comes from an aggregation strat
            rm_idx = np.loadtxt(path_remove, dtype=int)[:, 1]
            flag_rm = True  # flag that column is based on answers.json number
    else:
        rm_idx = []
        flag_rm = False
    if path_labels:
        labs = np.load(path_labels)
        ll = []
        targets = []
        imgs = []
        true_labels = []
        if not flag_rm:
            for i, samp in enumerate(dataset.samples):
                if i not in rm_idx:
                    img, true_label = samp
                    true_label = dataset.real_class_to_idx[
                        dataset.inv_class_to_idx[true_label]
                    ]
                    true_labels.append(true_label)
                    num = int(img.split("-")[-1].split(".")[0])
                    ll.append((img, labs[num]))
                    imgs.append(img)
                    targets.append(labs[num])
        else:
            to_save = []
            for i, samp in enumerate(dataset.samples):
                img, true_label = samp
                true_label = dataset.real_class_to_idx[
                    dataset.inv_class_to_idx[true_label]
                ]
                num = int(img.split("-")[-1].split(".")[0])
                if num not in rm_idx:
                    true_labels.append(true_label)
                    ll.append((img, labs[num]))
                    imgs.append(img)
                    targets.append(labs[num])
                else:
                    to_save.append([i, num])
            # save the matching pairs for too_hard identification
            np.savetxt(path_remove, np.array(to_save), fmt="%1i")
    else:
        labs = dataset.targets
        ll = []
        targets = []
        imgs = []
        true_labels = []
        for i, samp in enumerate(dataset.samples):
            if i not in rm_idx:
                img, true_label = samp
                true_label = dataset.real_class_to_idx[
                    dataset.inv_class_to_idx[true_label]
                ]
                true_labels.append(true_label)
                ll.append((img, true_label))
                imgs.append(img)
                targets.append(true_label)
    dataset.samples = ll
    dataset.imgs = imgs
    dataset.targets = targets
    dataset.true_labels = true_labels
    dataset.class_to_idx_imagefolder = dataset.class_to_idx
    dataset.class_to_idx = dataset.real_class_to_idx
    dataset.inv_class_to_idx = {v: k for k, v in dataset.class_to_idx.items()}
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
