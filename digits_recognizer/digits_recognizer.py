from abc import ABC, abstractmethod
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights, resnet18, ResNet18_Weights, resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
)


MLP_MODEL_TYPE = "mlp"
CNN_MOBILENET2_MODEL_TYPE = "cnn_mobilenet2"
CNN_RESNET_MODEL_TYPE = "cnn_resnet"
CNN_RESNEXT_MODEL_TYPE = "cnn_resnext"


class MLPLayer(nn.Module):
    def __init__(self, units, dropout=None, batchnorm=None, activation="leaky_relu"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout else None
        assert len(units) == 2
        input_units, output_units = units
        self.batchnorm = nn.BatchNorm1d(input_units) if batchnorm else None
        self.dense = nn.Linear(in_features=input_units, out_features=output_units)

        def make_activation(func, **kwargs):
            def act(x):
                return func(input=x, **kwargs)
            return act

        self.activation = make_activation(func=getattr(F, activation))
        if activation == "softmax":
            self.activation = make_activation(func=getattr(F, activation), dim=-1)
    
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ClassficiationHead(nn.Module):
    def __init__(self, units):
        super().__init__()
        assert isinstance(units, (list, tuple))
        layers_amount = len(units) - 1
        assert layers_amount > 0
        assert units[-1] == 10
        self.layers = [
            MLPLayer((units[i], units[i + 1]), dropout=0.1, batchnorm=True) for i in range(layers_amount - 1)
        ]
        self.layers.append(
            MLPLayer((units[-2], units[-1]), dropout=0.1, batchnorm=True, activation="softmax")
        )
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class IDigitsRecognizer(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.fext = self.build_feature_extractor()
        self.classifier = self.build_classifier()

    @abstractmethod
    def build_feature_extractor(self):
        pass

    @abstractmethod
    def flatten_features(self, x):
        pass

    @abstractmethod
    def build_classifier(self):
        pass

    def forward(self, x):
        features = self.fext(x)
        flatten = self.flatten_features(features)
        probs = self.classifier(flatten)
        return probs


class MLPClassifier(IDigitsRecognizer):
    def __init__(self, units=None):
        if units is None:
            units = [28 * 28, 100, 10]
        self.units = units
        super().__init__()

    def build_feature_extractor(self):
        return lambda x: torch.flatten(x, start_dim=1)

    def flatten_features(self, x):
        return torch.flatten(x, start_dim=1)

    def build_classifier(self):
        return ClassficiationHead(self.units)

    def __str__(self) -> str:
        return f"digits_recognizer_mlp:{self.units}"


class ChannelOrderValidator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        assert(len(x.shape) == 4)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x


class MobileNetV2Classifier(IDigitsRecognizer):
    def __init__(self):
        super().__init__()

    def build_feature_extractor(self):
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(
            ChannelOrderValidator(),
            base_model.features[:8],
            nn.MaxPool2d(kernel_size=2)
        )

    def flatten_features(self, x):
        return torch.flatten(x, start_dim=1)

    def build_classifier(self):
        return ClassficiationHead(units=[64, 10])

    def __str__(self) -> str:
        return f"digits_recognizer_mobilenet2"


class ResNetClassifier(IDigitsRecognizer):
    def __init__(self):
        super().__init__()

    def build_feature_extractor(self):
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(
            ChannelOrderValidator(),
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.MaxPool2d(kernel_size=2),
        )

    def flatten_features(self, x):
        return torch.flatten(x, start_dim=1)

    def build_classifier(self):
        return ClassficiationHead(units=[512, 10])

    def __str__(self) -> str:
        return f"digits_recognizer_resnet"


class ResNeXtClassifier(IDigitsRecognizer):
    def __init__(self):
        super().__init__()

    def build_feature_extractor(self):
        base_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(
            ChannelOrderValidator(),
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            nn.MaxPool2d(kernel_size=2),
        )

    def flatten_features(self, x):
        return torch.flatten(x, start_dim=1)

    def build_classifier(self):
        return ClassficiationHead(units=[1024, 10])

    def __str__(self) -> str:
        return f"digits_recognizer_resnet"


class InferenceModel(nn.Module):
    def __init__(self, model, preprocessor, device):
        super().__init__()
        self.core = model
        self.preprocessor = preprocessor
        self.device = device
        self.core.to(self.device)
        self.core.eval()

    def __call__(self, img):
        return self.predict(img)

    def predict(self, img):
        assert len(img.shape) == 3
        tensor = self.preprocessor(img)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.to(self.device)
        scores_tensor = self.core.forward(tensor)
        scores = scores_tensor.detach().to('cpu').numpy()[0]
        return np.argmax(scores)

    def forward(self, x):
        return self.predict(x)

    def __str__(self) -> str:
        return f"InferenceModel:{self.core}"


def build_model(model_type):
    if model_type == MLP_MODEL_TYPE:
        return MLPClassifier()
    elif model_type == CNN_MOBILENET2_MODEL_TYPE:
        return MobileNetV2Classifier()
    elif model_type == CNN_RESNET_MODEL_TYPE:
        return ResNetClassifier()
    elif model_type == CNN_RESNEXT_MODEL_TYPE:
        return ResNeXtClassifier()
    else:
        raise RuntimeError(f'Invalid model type: {model_type}')


if __name__ == "__main__":
    model = ResNeXtClassifier()
    model(torch.rand(1, 28, 28, 1))
