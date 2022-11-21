import torch
import albumentations as A
from albumentations.augmentations.geometric.transforms import HorizontalFlip
from albumentations.augmentations.crops.transforms import RandomResizedCrop 
from albumentations.augmentations.geometric.rotate import Rotate
import numpy as np

from common.torch_utils import normalize_img


class Normalize(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        h, w, _ = image.shape
        keypoints = keypoints.reshape([-1, 2])
        keypoints[:, 0] = keypoints[:, 0] / h
        keypoints[:, 1] = keypoints[:, 1] / w
        return {
            'image': normalize_img(image),
            'keypoints': keypoints.flatten(),
        }


class Denormalize(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        h, w, _ = image.shape
        keypoints = keypoints.reshape([-1, 2])
        keypoints[:, 0] = keypoints[:, 0] * h
        keypoints[:, 1] = keypoints[:, 1] * w
        return {
            'image': ((image + 0.5) * 255.).astype('uint8'),
            'keypoints': keypoints.flatten(),
        }


class AlbumentationsTransform(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints[:, ::-1]  # y, x -> x, y
        sample = {
            'image': image,
            'keypoints': keypoints,
        }

        transformed = self._transform(**sample)

        image, keypoints = transformed['image'], transformed['keypoints']
        keypoints = np.asarray(keypoints)[:, ::-1]  # x, y -> y, x
        keypoints = keypoints.flatten()
        return {
            'image': image,
            'keypoints': keypoints,
        }


class HorizontalFlipTransform(AlbumentationsTransform):
    def __init__(self, p=0.5):
        self._transform = A.Compose([
            HorizontalFlip(p),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        super(HorizontalFlipTransform).__init__()


class RotateTransform(AlbumentationsTransform):
    def __init__(self, p=0.5, limit=15):
        self._transform = A.Compose([
            Rotate(p=p, limit=limit),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        super(RotateTransform).__init__()


class RandomResizedCropTransform(AlbumentationsTransform):
    def __init__(self, height, width, scale=(1., 1.), ratio=(1., 1.), p=0.5):
        self._transform = A.Compose([
            RandomResizedCrop(
                height=height,
                width=width,
                scale=scale,
                ratio=ratio,
                p=p,
            ),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        super(RandomResizedCropTransform).__init__()


class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # image = image.transpose((2, 0, 1))  # HWC -> CHW
        return {
            'image': torch.from_numpy(image).float(),
            'keypoints': torch.from_numpy(keypoints).float(),
        }


class ToNumpy(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # image = torch.permute(image, (1, 2, 0))  # CHW -> HWC
        return {
            'image': image.numpy(),
            'keypoints': keypoints.numpy(),
        }


class CompositeTransform:
    def __init__(self, operators):
        self._ops = operators

    def __call__(self, x):
        for op in self._ops:
            x = op(x)
        return x
