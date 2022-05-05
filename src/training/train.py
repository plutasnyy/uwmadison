import os
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from src.training.config import CFG, DATA_TRANSFORMS
from src.training.constants import LOCAL_BASE_TRAIN_PATH
from src.training.datamodule import ImageSegmentationDatamodule
from src.training.lightning_module import ImageSegmentator
from src.training.training_utils import fill_df_with_is_empty_entry

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")
tqdm.pandas()
pd.options.plotting.backend = "plotly"
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

if __name__ == "__main__":

    wandb_logger = None
    if not CFG.debug:
        try:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret("WANDB")
            wandb.login(key=api_key)
            anonymous = None
        except:
            print(
                "To use your W&B account, Go to Add-ons -> Secrets and provide your "
                "W&B access token. Use the Label name as WANDB."
                "Get your W&B access token from here: https://wandb.ai/authorize"
            )
        wandb_logger = WandbLogger(project="uwmadison", name=CFG.exp_name)
        wandb_logger.config = {**CFG}

    pl.seed_everything(CFG.seed)
    df = pd.read_csv(str(LOCAL_BASE_TRAIN_PATH / "train.csv"))
    fill_df_with_is_empty_entry(df)

    skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["empty"], groups=df["case"])
    ):
        df.loc[val_idx, "fold"] = fold

    datamodule = ImageSegmentationDatamodule(
        df=df, batch_size=CFG.batch_size, data_transforms=DATA_TRANSFORMS, fold=0
    )
    model = ImageSegmentator()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        verbose=True,
        monitor="val/loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=CFG.epochs,
        logger=wandb_logger,
        gpus=torch.cuda.is_available(),
        precision=CFG.precision,
        deterministic=True,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule)
