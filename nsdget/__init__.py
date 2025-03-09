import wget
import pandas as pd
import os

BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com/"
DATA_DIR = os.path.join(".", "data")
INFO_FILENAME = "nsd_stim_info_merged.parquet"


def download_nsd(filepath: str, outfile: str):
    wget.download(BASE_URL + filepath, out=outfile)


def download_info():
    df = pd.read_pickle(BASE_URL + "nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
    df.to_parquet(os.path.join(DATA_DIR, INFO_FILENAME))


def df_info():
    return pd.read_parquet(os.path.join(DATA_DIR, INFO_FILENAME))


def main():
    download_info()


if __name__ == "__main__":
    main()
