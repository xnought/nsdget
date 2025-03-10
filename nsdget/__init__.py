import wget
import pandas as pd
import os

BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com"
COCO_BASE_URL = "http://images.cocodataset.org"
DATA_DIR = os.path.join(".", "data")
INFO_FILENAME = "nsd_stim_info_merged.parquet"
IMAGES_FILENAME = "nsd_stimuli.hdf5"


def download_nsd(filepath: str, outfile: str):
    wget.download(f"{BASE_URL}/{filepath}", out=outfile)


def download_stimuli_info():
    df = pd.read_pickle(f"{BASE_URL}/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
    df.to_parquet(os.path.join(DATA_DIR, INFO_FILENAME))


def df_stimuli_info():
    filename = os.path.join(DATA_DIR, INFO_FILENAME)
    if not os.path.exists(filename):
        print(f"Downloading to {filename}")
        download_stimuli_info()
    return pd.read_parquet(filename)


def download_images():
    download_nsd(
        "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
        os.path.join(DATA_DIR, IMAGES_FILENAME),
    )


def left_pad_zeros(number: int, pad_to=12):
    number_as_str = str(number)
    num_zeros = pad_to - len(number_as_str)
    assert num_zeros >= 0
    return "0" * num_zeros + number_as_str


def coco_image_links(coco_ids: list[int], splits: list[str]):
    assert len(coco_ids) == len(splits)
    for id, split in zip(coco_ids, splits):
        filename = f"{left_pad_zeros(id, pad_to=12)}.jpg"
        yield f"{COCO_BASE_URL}/{split}/{filename}", filename


def wget_if_not_already_downloaded(url: str, out: str, skip_if_exists: bool):
    if not skip_if_exists or not os.path.exists(out):
        wget.download(url, out=out)


def parallel_wget(urls: list[str], outs: list[str], skip_if_exists=True, **tpe_kwargs):
    from concurrent.futures import ThreadPoolExecutor

    assert len(urls) == len(outs)

    with ThreadPoolExecutor(**tpe_kwargs) as tpe:
        tpe.map(
            lambda d: wget_if_not_already_downloaded(d[0], d[1], skip_if_exists),
            zip(urls, outs),
        )


def download_coco_links(coco_ids: list[int], splits: list[str]):
    # make sure directories exist
    for split in splits:
        dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(dir):
            os.mkdir(dir)

    # download to directories
    links = []
    save_paths = []
    for (link, filename), split in zip(coco_image_links(coco_ids, splits), splits):
        links.append(link)
        save_paths.append(os.path.join(DATA_DIR, split, filename))

    parallel_wget(links, save_paths)


def main():
    # download_stimuli_info()
    # download_images()
    df = df_stimuli_info()
    sub = df[df["shared1000"] == True]
    download_coco_links(sub["cocoId"], sub["cocoSplit"])


if __name__ == "__main__":
    main()
