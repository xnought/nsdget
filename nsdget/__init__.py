import wget
import pandas as pd
import os
from PIL import Image

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


def coco_filename(id: int):
    return f"{left_pad_zeros(id, pad_to=12)}.jpg"


def coco_image_links(coco_ids: list[int], splits: list[str]):
    assert len(coco_ids) == len(splits)
    for id, split in zip(coco_ids, splits):
        filename = coco_filename(id)
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
    relative_paths = []
    for (link, filename), split in zip(coco_image_links(coco_ids, splits), splits):
        links.append(link)
        relative_paths.append(os.path.join(split, filename))

    parallel_wget(links, [os.path.join(DATA_DIR, r) for r in relative_paths])

    return relative_paths


def percent_crop_image(im: Image, percent_crop: list[float]) -> Image.Image:
    # percent crop is (top, bottom, left, right)
    [percent_top, percent_bottom, percent_left, percent_right] = percent_crop

    # but PIL.Image().crop takes in (left, top, right, bottom)
    width, height = im.size
    left = int(width * percent_left)
    top = int(height * percent_top)
    right = int(width * (1 - percent_right))
    bottom = int(height * (1 - percent_bottom))
    return im.crop([left, top, right, bottom])


def load_nsd_coco_image(
    coco_id: int, coco_split: str, crop: list[float] = None
) -> Image.Image:
    filename = coco_filename(coco_id)
    path = os.path.join(DATA_DIR, coco_split, filename)
    assert os.path.exists(path)

    im = Image.open(path).convert("RGB")
    if crop is not None:
        # resize based on https://cvnlab.slite.page/p/NKalgWd__F/Experiments
        # sometimes after crop the image is (426, 426) or (427, 427), so further resize to (425, 425)
        im = percent_crop_image(im, crop).resize((425, 425), Image.Resampling.LANCZOS)
    return im


def main():
    # download_stimuli_info()
    # download_images()
    df = df_stimuli_info()
    sub = df[df["shared1000"] == True]
    imgs = download_coco_links(sub["cocoId"], sub["cocoSplit"])
    row = sub.iloc[0]
    im = load_nsd_coco_image(row["cocoId"], row["cocoSplit"], row["cropBox"])


if __name__ == "__main__":
    main()
