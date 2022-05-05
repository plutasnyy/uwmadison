from typing import Dict

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.image_utils import plot_batch
from src.training.config import CFG, DATA_TRANSFORMS
from src.training.constants import LOCAL_BASE_TRAIN_PATH
from src.training.dataset import ImageSegmentationDataset
from src.training.training_utils import fill_df_with_is_empty_entry


class ImageSegmentationDatamodule(pl.LightningDataModule):
    def __init__(
        self, df: pd.DataFrame, batch_size: int, data_transforms: Dict, fold: int = 0
    ):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.data_transforms = data_transforms
        self.fold = fold

    def setup(self, stage=None):
        self.train_df = self.df.query("fold!=@self.fold").reset_index(drop=True)
        self.valid_df = self.df.query("fold==@self.fold").reset_index(drop=True)

        self.train_dataset = ImageSegmentationDataset(
            self.train_df, transforms=self.data_transforms["valid"]
        )

        self.val_dataset = ImageSegmentationDataset(
            self.valid_df, transforms=self.data_transforms["valid"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


if __name__ == "__main__":
    df = pd.read_csv(str(LOCAL_BASE_TRAIN_PATH / "data_processed" / "new_train.csv"))
    df = fill_df_with_is_empty_entry(df)

    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["empty"], groups=df["case"])
    ):
        df.loc[val_idx, "fold"] = fold

    datamodule = ImageSegmentationDatamodule(
        df=df, batch_size=CFG.batch_size, data_transforms=DATA_TRANSFORMS, fold=0
    )

    datamodule.prepare_data()
    datamodule.setup()

    imgs, msks = next(iter(datamodule.train_dataloader()))
    print(imgs.size(), msks.size())
    plot_batch(imgs, msks, size=5)
