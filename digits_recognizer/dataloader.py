import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, csv_path, transform=None, rgb=True):
        self._path = csv_path
        self._df = pd.read_csv(csv_path)
        self._raw_images = self._df.values
        self._labels = None
        if 'label' in self._df:
            self._is_annotated = True
            self._raw_images = self._df.drop(labels=['label'], axis=1).values
            self._labels = self._df['label'].values
        self._transform = transform
        self._rgb = rgb

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, idx):
        img = self._raw_images[idx].reshape([28, 28])
        if self._rgb:
            img = np.expand_dims(img, 0).repeat(3, 0)
        label = None if self._labels is None else self._labels[idx]
        if self._transform:
            img = self._transform(img)
        return img, label
    
    def is_annotated(self):
        return self._labels is not None
    
    def __str__(self):
        return f'MNISTDataset_{len(self)}_{self._path}'
