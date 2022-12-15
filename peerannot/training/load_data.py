from torchvision import datasets, transforms


def load_imagefolder(path, **kwargs):
    img_size = kwargs.get("img_size", 224)
    data_transforms = transforms.Compose(
        [
            transforms.resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(path, transform=data_transforms)
    return dataset
