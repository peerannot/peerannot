import torch
import torch.nn as nn
import torchvision


def get_all_models():
    torch_models = torch.hub.list("pytorch/vision")
    all_models = torch_models
    all_models.append("modellabelme")
    return all_models


class Classifier_labelme(nn.Module):
    def __init__(self, dropout, n_units, n_class):
        super().__init__()
        self.dropout = dropout
        self.n_units = n_units
        self.n_class = n_class

        self.backbone = torchvision.models.vgg16_bn(
            weights=torchvision.models.VGG16_BN_Weights.DEFAULT
        )
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.n_units),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_units, self.n_class),
            nn.Softmax(dim=-1),
        )

        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


def networks(name, n_classes, n_params=None, pretrained=False, cifar=False):
    """Load model as pytorch module

    :param name: name of the model: currently one available from the pytorch/vision repository
    :type name: str
    :param n_classes: number of classes
    :type n_classes: int
    :param n_params: number of parameters (needed when using a logistic regression classifier), defaults to None
    :type n_params: int, optional
    :param pretrained: Use pretrained weights on ImageNet, defaults to False
    :type pretrained: bool, optional
    :param cifar: If the model is going to be trained on CIFAR images, the first layer of resnet models need to be modified to avoid any hard downsampling, defaults to False
    :type cifar: bool, optional
    :return: Instanciated pytorch model
    :rtype: pytorch model
    """
    name = name.lower()
    torch_models = get_all_models()
    if name in torch_models and name != "modellabelme":
        if pretrained:
            weights = torch.hub.load(
                "pytorch/vision", "get_model_weights", name=name
            )
            weight = [weight for weight in weights][-1]
        else:
            weight = None
        model = torch.hub.load("pytorch/vision", name, weights=weight)
    elif name == "modellabelme":
        model = Classifier_labelme(0.5, 128, 8)

    if "resnet" in name:
        if model.fc.out_features != n_classes:
            model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif "vgg" in name:
        if model.classifier[6].out_features != n_classes:
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, n_classes
            )
    elif name != "modellabelme":
        raise NotImplementedError("Not implemented yet, sorry")
    print(f"Successfully loaded {name} with n_classes={n_classes}")
    if pretrained:
        print(f"\t with weights {weight}")
    if name.startswith("resnet") and cifar:
        print("Removing initial downsampling")
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=3, bias=False
        )
        model.maxpool = nn.Identity()  # avoid hard downsampling
    return model
