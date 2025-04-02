from __future__ import annotations
import wget
import pandas as pd
import os
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm

BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com"
COCO_BASE_URL = "http://images.cocodataset.org"
DATA_DIR = os.path.join(".", "data")
BETAS_DIR = os.path.join(DATA_DIR, "betas")
INFO_FILENAME = "nsd_stim_info_merged.parquet"
IMAGES_FILENAME = "nsd_stimuli.hdf5"

# took from https://github.com/clane9/NSD-Flat/
NUM_REP = 3  # image repeated at most 3 times per subject
NUM_SUBS = 8
NUM_TRIALS = 30_000
MAX_SESSIONS = 40
NUM_SESSIONS = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}
TRIALS_PER_SESSION = NUM_TRIALS // MAX_SESSIONS


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


def coco_filename(id: int):
    return f"{str(id).zfill(12)}.jpg"


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


def wget_if_not_already_downloaded(url: str, out: str, crop: list[float], skip_if_exists: bool):
    if not skip_if_exists or not os.path.exists(out):
        wget.download(url, out=out)
        crop_stimuli_image(Image.open(out), crop).save(out)  # override with cropped version


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
    assert "cocoId" in df.columns and "cocoSplit" in df.columns and "cropBox" in df.columns
    return download_stimuli_images(
        coco_ids=df["cocoId"],
        splits=df["cocoSplit"],
        crops=df["cropBox"],
        base_dir=base_dir,
    )


def open_stimuli_image(coco_id: int, coco_split: str, base_dir: str = DATA_DIR) -> Image.Image:
    filename = coco_filename(coco_id)
    path = os.path.join(base_dir, coco_split, filename)
    assert os.path.exists(path)

    im = Image.open(path).convert("RGB")
    return im


def df_row_open_stimuli_image(row: pd.DataFrame, base_dir: str = DATA_DIR):
    return open_stimuli_image(coco_id=row["cocoId"], coco_split=row["cocoSplit"], base_dir=base_dir)


def drop_subject_rep_cols(df: pd.DataFrame):
    df.drop([f"subject{subject_idx + 1}_rep{rep_id}" for subject_idx in range(NUM_SUBS) for rep_id in range(3)], inplace=True, axis=1)


def drop_subject_cols(df: pd.DataFrame):
    df.drop([f"subject{subject_idx + 1}" for subject_idx in range(NUM_SUBS)], inplace=True, axis=1)


# copied directly (with some renaming) from https://github.com/clane9/NSD-Flat/blob/main/convert_nsd_annotations.py#L277
def unroll_stimuli_trials(stim_info: pd.DataFrame) -> pd.DataFrame:
    long_stim_info = []

    for ii in tqdm(range(len(stim_info))):
        row = stim_info.iloc[ii].to_dict()
        for subject_idx in range(NUM_SUBS):
            subject_id = subject_idx + 1
            for rep_id in range(NUM_REP):
                trial_id = row[f"subject{subject_id}_rep{rep_id}"]
                if trial_id > 0:
                    long_row = {"subjectId": subject_id, "trialId": trial_id, **row}
                    long_stim_info.append(long_row)

    long_stim_info = pd.DataFrame.from_records(long_stim_info, index=["subjectId", "trialId"])
    long_stim_info = long_stim_info.sort_index()

    drop_subject_rep_cols(long_stim_info)
    drop_subject_cols(long_stim_info)

    return long_stim_info


def str_subject(subject: int):
    return f"subj{str(subject).zfill(2)}"


def str_session(session: int):
    return f"session{str(session).zfill(2)}"


# took from https://github.com/clane9/NSD-Flat/
def download_betas_given_subj_session(
    subject: int,
    session: int,
    betas_type: str = "fsaverage/betas_fithrf_GLMdenoise_RR",
    base_dir: str = DATA_DIR,
) -> dict[str, str]:
    file_paths = []
    for h in ["lh", "rh"]:
        filename = f"{h}.betas_{str_session(session)}.mgh"
        full_path = os.path.join(base_dir, filename)
        url = f"{BASE_URL}/nsddata_betas/ppdata/{str_subject(subject)}/{betas_type}/{filename}"
        if not os.path.exists(full_path):
            wget.download(url, out=base_dir)
        file_paths.append(full_path)
    return file_paths


# took from https://github.com/clane9/NSD-Flat/
def load_betas_given_subj_session(
    subject: int,
    session: int,
    betas_type: str = "fsaverage/betas_fithrf_GLMdenoise_RR",
    base_dir: str = DATA_DIR,
    dtype=np.float32,
):
    file_paths = download_betas_given_subj_session(subject, session, betas_type, base_dir)

    hemispheres = []
    for fp in file_paths:
        img = nib.load(fp)
        print(img.header.get_data_shape())
        fdata: np.ndarray = img.get_fdata()  # (BETAS, 1, 1, TRIALS)
        fdata = np.squeeze(fdata)  # (BETAS, TRIALS)
        fdata = np.ascontiguousarray(fdata.T)  # (TRIALS, BETAS)
        hemispheres.append(fdata)
        print(fdata.shape)

    # stitch together both hemispheres
    return np.concat(hemispheres, axis=1, dtype=dtype)  # (TRIALS, BETAS*2)


def download_vol_betas_subject_session(subject_id: str, session_id: str, base_dir: str = BETAS_DIR):
    subject_dir = os.path.join(base_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    filename = f"betas_{session_id}.nii.gz"
    download_to = os.path.join(base_dir, subject_id, filename)
    if os.path.exists(download_to):
        print(f"Already downloaded at {download_to}")
        return download_to

    print(f"Downloading to {download_to}")
    link = f"nsddata_betas/ppdata/{subject_id}/func1pt8mm/betas_fithrf_GLMdenoise_RR/{filename}"
    download_nsd(link, download_to)

    return download_to


def load_vol_betas_subject_session(subject_id: str, session_id: str, base_dir: str = BETAS_DIR):
    filename = download_vol_betas_subject_session(subject_id, session_id, base_dir)
    d = nib.load(filename)
    voxels = d.get_fdata()
    return voxels


def download_all_session_betas(subject_id: str, base_dir: str = BETAS_DIR):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as tpe:
        tpe.map(lambda i: download_vol_betas_subject_session(subject_id, str_session(i), base_dir), range(1, NUM_SESSIONS[subject_id] + 1))


def get_shape_data(filename):
    return nib.load(filename).header.get_data_shape()


def main():
    # df = df_stimuli_info()
    # print(df.columns)
    # sub = df[df["shared1000"]].copy()
    # sub["img"] = df_download_stimuli_images(sub)
    # Image.open(sub.iloc[0]["img"]).show()
    session = 1
    subject = 1

    # betas = load_vol_betas_subject_session(session_id=str_session(session), subject_id=str_subject(subject))
    # print(betas.shape)

    for subject_idx in range(NUM_SUBS):
        subject_id = str_subject(subject_idx + 1)  # since subjects are 1 indexed from 1 to 8
        download_all_session_betas(subject_id)  # download all at once for speed
        for session_idx in range(NUM_SESSIONS[subject_id]):
            session_id = str_session(session_idx + 1)
            # for trial_idx in range(NUM_TRIALS):
            #     print(trial_idx)


if __name__ == "__main__":
    main()
