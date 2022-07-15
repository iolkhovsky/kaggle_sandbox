import platform
import torch


def get_available_device(verbose=True):
    exec_device = torch.device('cpu')
    if torch.has_mps:
        exec_device = torch.device('mps')
    if torch.has_cuda:
        exec_device = torch.device('cuda')
    if verbose:
        print(f'Platform: {platform.system()}')
        print(f'Release: {platform.release()}')
        print(f'MPS available: {torch.has_mps}')
        print(f'CUDA available: {torch.has_cuda}')
        print(f'Selected device: {exec_device}')
    return exec_device


def global_max_pool2d(tensor, keep_dims=True):
    assert len(tensor.shape) in [2, 4]
    if len(tensor.shape) == 4:
        b, c, h, w = tensor.shape
    else:
        b, c = tensor.shape
    res, _ = torch.max(torch.reshape(tensor, [b, c, -1]), dim=-1)
    if not keep_dims:  # (b, c)
        return res
    else:  # (b, c, 1, 1)
        return torch.reshape(res, [b, c, 1, 1])


def normalize_img(image_tensor):
    return (image_tensor.float() / 255.) - 0.5