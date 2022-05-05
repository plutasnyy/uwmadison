import numpy as np
import torch
from torch.utils.data import Dataset

from src.image_utils import load_img, load_msk, load_3c_float_img


class ImageSegmentationDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path_np"].tolist()
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_3c_float_img(img_path)

        msk_path = self.msk_paths[index]
        msk = load_msk(msk_path)

        if self.transforms:
            data = self.transforms(image=img, mask=msk)
            img = data["image"]
            msk = data["mask"]

        img = np.transpose(img, (2, 0, 1))
        msk = np.transpose(msk, (2, 0, 1))

        return torch.tensor(img), torch.tensor(msk)

    def __len__(self):
        return len(self.df)
