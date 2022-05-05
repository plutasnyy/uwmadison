from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data_processing.constants import KAGGLE_OUTPUT_DATA_PATH, KAGGLE_DATA_PATH
from src.rle import rle_decode

pd.options.plotting.backend = "plotly"


def get_metadata(row):
    data = row["id"].split("_")
    case = int(data[0].replace("case", ""))
    day = int(data[1].replace("day", ""))
    slice_ = int(data[-1])
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def path2info(row):
    path = row["image_path"]
    data = path.split("/")
    slice_ = int(data[-1].split("_")[1])
    case = int(data[-3].split("_")[0].replace("case", ""))
    day = int(data[-3].split("_")[1].replace("day", ""))
    width = int(data[-1].split("_")[2])
    height = int(data[-1].split("_")[3])
    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def id2mask(id_: str, df: pd.DataFrame) -> np.ndarray:
    """
    Filters specific slice in dataset, then for each class creates mask
    Return three-dimensional numpy array of type uint8, max value = 1 which indicates TRUE
    """
    filtered_df = df[df["id"] == id_]
    width_and_height = filtered_df[["height", "width"]].iloc[0]
    shape = (width_and_height.height, width_and_height.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        class_df = filtered_df[filtered_df["class"] == class_]
        rle = class_df.segmentation.squeeze()
        if len(class_df) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def save_mask(
    id_: str, df: pd.DataFrame, save_dir: Path = KAGGLE_OUTPUT_DATA_PATH
) -> None:
    idf = df[df["id"] == id_]
    mask = id2mask(id_, df=df) * 255

    mask_path_png = save_dir / "png" / (id_ + ".png")
    mask_path_png.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(mask_path_png),
        cv2.cvtColor(mask, cv2.COLOR_BGR2RGB),
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )

    mask_path_numpy = save_dir / "np" / (id_ + ".npy")
    mask_path_numpy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(mask_path_numpy), mask)


def prepare_df(data_directory: Path = KAGGLE_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_directory / "train.csv")
    df = df.progress_apply(get_metadata, axis=1)

    paths = data_directory.glob("./train/*/*/*/*")
    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df["image_path"] = path_df["image_path"].astype(str)
    path_df = path_df.progress_apply(path2info, axis=1)
    df = df.merge(path_df, on=["case", "day", "slice"])

    return df
