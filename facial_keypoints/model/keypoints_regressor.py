import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights, resnet18, ResNet18_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
)
import torch.nn.functional as F

from common.torch_utils import global_max_pool2d


def get_backbone(backbone_type, pretrained=True):
    weights, backbone = None, None
    if backbone_type == 'mobilenet2':
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = mobilenet_v2(weights)
    elif backbone_type == 'resnet18':
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights)
    elif backbone_type == 'resnext50':
        if pretrained:
            weights = weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        backbone = resnext50_32x4d(weights)
    assert backbone, f'Invalid backbone type: {backbone_type}'
    return backbone


class MLPLayer(nn.Module):
    def __init__(self, units, dropout=None, batchnorm=None, activation='leaky_relu'):
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
        if activation == 'softmax':
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


class RegressionHead(nn.Module):
    def __init__(self, points, dense_structure=(128,), inner_activation='leaky_relu',
                 final_activation='sigmoid'):
        super().__init__()
        dense_structure = [1000,] + list(dense_structure) + [points * 2]
        self.layers = []
        for i in range(len(dense_structure) - 1):
            in_units, out_units = dense_structure[i], dense_structure[i + 1]
            activation = inner_activation
            if i + 1 == len(dense_structure) - 1:
                activation = final_activation
            self.layers.append(
                MLPLayer([in_units, out_units], dropout=0.2, batchnorm=True, activation=activation)
            )
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModelCore(nn.Module):
    def __init__(self, backbone, pretrained=True, points=15, regressor_struct=(128,),
                 regressor_inner_act='leaky_relu', regressor_final_act='sigmoid'):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained)
        self.maxpool = lambda x: global_max_pool2d(x, keep_dims=False)
        self.regression_head = RegressionHead(
            points=points,
            dense_structure=regressor_struct,
            inner_activation=regressor_inner_act,
            final_activation=regressor_final_act,
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.maxpool(features)
        keypoints = self.regression_head(features)
        return keypoints


class Preprocessor(nn.Module):
    def __init__(self, target_resolution=(96, 96)):
        super().__init__()
        self._height, self._width = target_resolution
        self._normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _prepare_single_image(self, tensor_or_array):
        assert len(tensor_or_array.shape) == 3
        if isinstance(tensor_or_array, np.ndarray):
            tensor = torch.from_numpy(tensor_or_array)
        else:
            tensor = tensor_or_array
        if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = torch.permute(tensor, (2, 0, 1))
        _, h, w = tensor.shape
        scale = (1., 1.)
        if (h, w) != (self._height, self._width):
            scale = (h / self._height, w / self._width)
            tensor = torchvision.transforms.Resize(
                size=(self._height, self._width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )(tensor)
        return tensor.float(), scale

    def forward(self, img_or_batch):
        if isinstance(img_or_batch, list) or len(img_or_batch.shape) == 4:
            tensors_and_scales = [self._prepare_single_image(x) for x in img_or_batch]
            tensors, scales = zip(*tensors_and_scales)
            tensors = torch.stack(tensors)
        else:
            tensor, scale = self._prepare_single_image(img_or_batch)
            tensors, scales = torch.unsqueeze(tensor, 0), [scale]
        b, c, h, w = tensors.shape
        assert c == 3 and h == self._height and w == self._width and b > 0, \
            f"b, h, w, c = {b, h, w, c}"
        normalized_tensors = self._normalize(tensors)
        return normalized_tensors, scales


class Postprocessor(nn.Module):
    def __init__(self, keypoints_names, target_resolution=(96, 96)):
        super().__init__()
        self._names = keypoints_names
        self._height, self._width = target_resolution

    def forward(self, raw_prediction, scales, ret_raw=True):
        b, p = raw_prediction.shape
        assert b == len(scales) and len(self._names) * 2 == p
        raw_prediction = torch.reshape(raw_prediction, (b, -1, 2))
        img_scaler = torch.Tensor([self._height, self._width]).to(raw_prediction.device)
        raw_prediction = raw_prediction * img_scaler

        scales = torch.Tensor(scales).to(raw_prediction.device).repeat(1, p // 2).reshape(b, -1, 2)

        raw_prediction = raw_prediction * scales
        if ret_raw:
            return raw_prediction.reshape(b, -1)
        else:
            raw_prediction = raw_prediction.detach().numpy()
            res = []
            for image_prediction in raw_prediction:
                image_result = {name: (y, x) for (y, x), name in zip(image_prediction, self._names)}
                res.append(image_result)
            return res


class KeypointsRegressor(nn.Module):
    def __init__(self, 
                 backbone,
                 keypoints_names,
                 target_resolution=(96, 96),
                 pretrained_backbone=True,
                 regressor_struct=(128,),
                 regressor_inner_act='leaky_relu',
                 regressor_final_act='sigmoid',
                 ):
        super().__init__()
        self.preprocessor = Preprocessor(
            target_resolution=target_resolution,
        )
        self.core = ModelCore(
            backbone=backbone,
            pretrained=pretrained_backbone,
            points=len(keypoints_names),
            regressor_struct=regressor_struct,
            regressor_inner_act=regressor_inner_act,
            regressor_final_act=regressor_final_act,
        )
        self.postprocessor = Postprocessor(
            keypoints_names=keypoints_names,
            target_resolution=target_resolution,
        )

    def forward(self, x, ret_raw=True):
        tensor, scales = self.preprocessor(x)
        prediction = self.core(tensor)
        result = self.postprocessor(prediction, scales, ret_raw=ret_raw)
        return result


def build_model(model_config):
    img_height = model_config['target_resolution']['height']
    img_width = model_config['target_resolution']['width']
    model = KeypointsRegressor(
        backbone=model_config['backbone']['name'],
        pretrained_backbone=model_config['backbone']['pretrained'],
        keypoints_names=model_config['keypoints_names'],
        target_resolution=(img_height, img_width),
        regressor_struct=model_config['regression_head']['dense_struct'],
        regressor_inner_act=model_config['regression_head']['activation'],
        regressor_final_act=model_config['regression_head']['final_activation'],
    )
    return model
