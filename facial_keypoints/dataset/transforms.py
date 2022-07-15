import torch
import albumentations as A
from albumentations.augmentations.geometric import transforms
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


class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self._p = p
        self._transform = A.Compose([
            transforms.HorizontalFlip(p),
        ], keypoint_params=A.KeypointParams(format='xy'))

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints[:, ::-1]  # y, x -> x, y
        sample = {
            'image': image,
            'keypoints': keypoints,
        }
        transformed = self._transform(**sample, keypoint_param=A.KeypointParams(format='xy'))
        image, keypoints = transformed['image'], transformed['keypoints']
        keypoints = np.asarray(keypoints)[:, ::-1]  # x, y -> y, x
        keypoints = keypoints.flatten()
        return {
            'image': image,
            'keypoints': keypoints,
        }


class ToTensor(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # image = image.transpose((2, 0, 1))  # HWC -> CHW
        return {
            'image': torch.from_numpy(image),
            'keypoints': torch.from_numpy(keypoints)
        }


class ToNumpy(object):
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # image = torch.permute(image, (1, 2, 0))  # CHW -> HWC
        return {
            'image': image.numpy(),
            'keypoints': keypoints.numpy(),
        }
