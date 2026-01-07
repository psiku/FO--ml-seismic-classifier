import polars as pl
from datasets import load_dataset
from pathlib import Path
from src.constants import (
    DATASET_NAME,
    RAW_DATA_DIR,
    SESMIC_DATA_CSV
    )


def hf_to_polars(ds_split) -> pl.DataFrame:
    """
    Convert a HuggingFace Dataset split -> Polars DataFrame.
    """
    return pl.from_pandas(ds_split.to_pandas())


def download_and_save_all(data_folder: str | Path = RAW_DATA_DIR):
    data_dir = Path(data_folder)
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME)

    # if we have DatasetDict
    if hasattr(ds, "keys"):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]

    sesmic_df = hf_to_polars(ds)
    sesmic_df.write_csv(SESMIC_DATA_CSV)

    return sesmic_df


if __name__ == "__main__":
    sesmic_df = download_and_save_all()
    print("Saved to:", SESMIC_DATA_CSV)
    print("Shape:", sesmic_df.shape)
