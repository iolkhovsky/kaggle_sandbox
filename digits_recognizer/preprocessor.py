import numpy as np
import torch
from torchvision.transforms import Normalize


IMAGENET_PREPROCESSING_TYPE = "imagenet"


class TorchImagenetPreprocessor:
    def __init__(self) -> None:
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.transform = Normalize(mean=self.mean, std=self.std, inplace=False)
    
    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            tensor = torch.Tensor(tensor)
        if tensor.dtype is not torch.float32:
            tensor.type(torch.float32)
        assert 3 <= len(tensor.shape) <= 4
        if tensor.shape[-1] == 3:
            if len(tensor.shape) == 3:
                tensor = tensor.permute([2, 0, 1])
            else:
                tensor = tensor.permute([0, 3, 1, 2])
        tensor = torch.div(tensor, 255.)
        return self.transform(tensor)
    
    def restore(self, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to('cpu').numpy()
        assert 3 <= len(tensor.shape) <= 4
        if tensor.shape[-3] == 3:
            if len(tensor.shape) == 3:
                tensor = np.transpose(tensor, [1, 2, 0])
            else:
                tensor = np.transpose(tensor, [0, 2, 3, 1])
        tensor = np.multiply(tensor, self.std)
        tensor = np.add(tensor, self.mean)
        tensor *= 255.
        return tensor.astype(np.uint8)
    
    def __str__(self) -> str:
        return 'TorchImagenetPreprocessor'


def build_preprocessor(preprocessor_type):
    if preprocessor_type == IMAGENET_PREPROCESSING_TYPE:
        return TorchImagenetPreprocessor()
    else:
        raise RuntimeError(f'Invalid preprocessor type: {preprocessor_type}')
