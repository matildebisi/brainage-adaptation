"""Dataset reading the preprocessed NIfTI volumes listed in metadata."""

import os
import numpy as np
from torch.utils.data import Dataset


class MyCustomDataset(Dataset):
    def __init__(self, transform, loader, dataFrame, img_path_root=None,
                 standardize=False, mean_age=None, std_age=None):
        self.rows = dataFrame.reset_index(drop=True)
        self.transform = transform
        self.loader = loader
        self.img_path_root = img_path_root
        self.standardize = standardize
        self.mean_age = mean_age
        self.std_age = std_age
        if standardize and (std_age is None or std_age <= 0):
            raise ValueError('Age standardization requires a positive training standard deviation.')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        file_path = row.get('img_path')
        if not isinstance(file_path, str) or not file_path:
            file_path = os.path.join(self.img_path_root, str(row['filename']))
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'Missing image: {file_path}')
        age = float(row['AGE'])
        target = ((age - self.mean_age) / self.std_age
                  if self.standardize else age / 100.0)
        image = self.loader(file_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, np.float32(target)
