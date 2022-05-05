import pandas as pd


def fill_df_with_is_empty_entry(df: pd.DataFrame) -> pd.DataFrame:
    df["segmentation"] = df.segmentation.fillna("")
    df["rle_len"] = df.segmentation.map(len)  # length of each rle mask

    df2 = (
        df.groupby(["id"])["segmentation"].agg(list).to_frame().reset_index()
    )  # rle list of each id
    df2 = df2.merge(
        df.groupby(["id"])["rle_len"].agg(sum).to_frame().reset_index()
    )  # total length of all rles of each id

    df = df.drop(columns=["segmentation", "class", "rle_len"])
    df = df.groupby(["id"]).head(1).reset_index(drop=True)
    df = df.merge(df2, on=["id"])
    df["empty"] = df.rle_len == 0  # empty masks
    return df
