from __future__ import annotations
import wget
import pandas as pd
import os
from PIL import Image

BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com"
COCO_BASE_URL = "http://images.cocodataset.org"
DATA_DIR = os.path.join(".", "data")
INFO_FILENAME = "nsd_stim_info_merged.parquet"
IMAGES_FILENAME = "nsd_stimuli.hdf5"


def mkdir_if_not_exists(base_dir: str):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)


def download_nsd(filepath: str, outfile: str):
    wget.download(f"{BASE_URL}/{filepath}", out=outfile)


def download_stimuli_info(base_dir: str = DATA_DIR):
    mkdir_if_not_exists(base_dir)

    df = pd.read_pickle(f"{BASE_URL}/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
    df.to_parquet(os.path.join(base_dir, INFO_FILENAME))


def df_stimuli_info(base_dir: str = DATA_DIR):
    filename = os.path.join(base_dir, INFO_FILENAME)
    if not os.path.exists(filename):
        print(f"Downloading to {filename}")
        download_stimuli_info(base_dir)
    return pd.read_parquet(filename)


def left_pad_zeros(number: int, pad_to=12):
    number_as_str = str(number)
    num_zeros = pad_to - len(number_as_str)
    assert num_zeros >= 0
    return "0" * num_zeros + number_as_str


def coco_filename(id: int):
    return f"{left_pad_zeros(id, pad_to=12)}.jpg"


def coco_image_links(coco_ids: list[int], splits: list[str]):
    assert len(coco_ids) == len(splits)
    for id, split in zip(coco_ids, splits):
        filename = coco_filename(id)
        yield f"{COCO_BASE_URL}/{split}/{filename}", filename


def percent_crop_image(im: Image.Image, percent_crop: list[float]) -> Image.Image:
    # percent crop is (top, bottom, left, right)
    [percent_top, percent_bottom, percent_left, percent_right] = percent_crop

    # but PIL.Image().crop takes in (left, top, right, bottom)
    width, height = im.size
    left = int(width * percent_left)
    top = int(height * percent_top)
    right = int(width * (1 - percent_right))
    bottom = int(height * (1 - percent_bottom))

    return im.crop([left, top, right, bottom])


def crop_stimuli_image(im: Image.Image, crop: list[float]):
    # resize based on https://cvnlab.slite.page/p/NKalgWd__F/Experiments
    # sometimes after crop the image is (426, 426) or (427, 427), so further resize to (425, 425)
    im = percent_crop_image(im, crop).resize((425, 425), Image.Resampling.LANCZOS)
    return im


def wget_if_not_already_downloaded(
    url: str, out: str, crop: list[float], skip_if_exists: bool
):
    if not skip_if_exists or not os.path.exists(out):
        wget.download(url, out=out)
        crop_stimuli_image(Image.open(out), crop).save(
            out
        )  # override with cropped version


def parallel_image_download(
    urls: list[str],
    outs: list[str],
    crops: list[list[float]],
    skip_if_exists=True,
    **tpe_kwargs,
):
    from concurrent.futures import ThreadPoolExecutor

    assert len(urls) == len(outs) and len(urls) == len(crops)

    with ThreadPoolExecutor(**tpe_kwargs) as tpe:
        tpe.map(
            lambda d: wget_if_not_already_downloaded(*d, skip_if_exists),
            zip(urls, outs, crops),
        )


def download_stimuli_images(
    coco_ids: list[int],
    splits: list[str],
    crops: list[list[float]],
    base_dir: str = DATA_DIR,
) -> list[str]:
    mkdir_if_not_exists(base_dir)

    # sub directories (ie val2017, train2017) to save to
    for split in splits:
        mkdir_if_not_exists(os.path.join(base_dir, split))

    # links to download
    links = []
    paths = []
    for (link, filename), split in zip(coco_image_links(coco_ids, splits), splits):
        links.append(link)
        paths.append(os.path.join(base_dir, split, filename))

    # download on max possible threads in parallel
    parallel_image_download(links, paths, crops)

    return paths


def df_download_stimuli_images(df: pd.DataFrame, base_dir=DATA_DIR) -> list[str]:
    assert (
        "cocoId" in df.columns and "cocoSplit" in df.columns and "cropBox" in df.columns
    )
    return download_stimuli_images(
        coco_ids=df["cocoId"],
        splits=df["cocoSplit"],
        crops=df["cropBox"],
        base_dir=base_dir,
    )


def open_stimuli_image(
    coco_id: int, coco_split: str, base_dir: str = DATA_DIR
) -> Image.Image:
    filename = coco_filename(coco_id)
    path = os.path.join(base_dir, coco_split, filename)
    assert os.path.exists(path)

    im = Image.open(path).convert("RGB")
    return im


def df_row_open_stimuli_image(row: pd.DataFrame, base_dir: str = DATA_DIR):
    return open_stimuli_image(
        coco_id=row["cocoId"], coco_split=row["cocoSplit"], base_dir=base_dir
    )


def main():
    df = df_stimuli_info()
    sub = df[df["shared1000"] == True].copy()
    sub["img"] = df_download_stimuli_images(sub)
    Image.open(sub.iloc[0]["img"]).show()


if __name__ == "__main__":
    main()
