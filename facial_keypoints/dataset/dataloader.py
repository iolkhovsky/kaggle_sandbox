import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.sample import Sample


class KeyPointsDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self._df = pd.read_csv(csv_file).dropna()
        self._transform = transform

    def __len__(self):
        return len(self._df.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_sample = Sample.from_series(self._df.iloc[idx])
        sample = {
            'image': data_sample.image,
            'keypoints': Sample.keypoints_to_vector(data_sample.keypoints).astype(np.float32),
        }

        if self._transform:
            sample = self._transform(sample)

        return sample
