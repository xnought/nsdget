from __future__ import annotations
import pandas as pd
import os
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from colored import Fore, Back, Style

NiiImage = nib.nifti1.Nifti1Image

BASE_URL = "https://natural-scenes-dataset.s3.amazonaws.com"
COCO_BASE_URL = "http://images.cocodataset.org"
DATA_DIR = os.path.join(".", "nsdata")
BETAS_DIR = os.path.join(DATA_DIR, "betas")
INFO_FILENAME = "nsd_stim_info_merged.parquet"
IMAGES_FILENAME = "nsd_stimuli.hdf5"
TRIALS_FILENAME = "trials.parquet"

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


def parallel_map(func: Callable, args_per_call: list, **tpe_kwargs):
    with ThreadPoolExecutor(**tpe_kwargs) as tpe:
        futures = [tpe.submit(func, *args) for args in args_per_call]

        # Initialize progress bar
        with tqdm(total=len(futures)) as pbar:
            # Update progress as each future completes
            for _ in as_completed(futures):
                pbar.update(1)


def wget(url: str, filename: str):
    urlretrieve(url, filename)


def download_nsd(filepath: str, outfile: str):
    wget(f"{BASE_URL}/{filepath}", outfile)


def assert_data_dir():
    assert os.path.exists(DATA_DIR), "Data dir must be set"


def download_stimuli_info():
    assert_data_dir()

    df = pd.read_pickle(f"{BASE_URL}/nsddata/experiments/nsd/nsd_stim_info_merged.pkl")
    df.to_parquet(os.path.join(DATA_DIR, INFO_FILENAME))


def df_stimuli_info():
    assert_data_dir()

    filename = os.path.join(DATA_DIR, INFO_FILENAME)
    if not os.path.exists(filename):
        print(f"Downloading to {filename}")
        download_stimuli_info()
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
        wget(url, out)
        crop_stimuli_image(Image.open(out), crop).save(out)  # override with cropped version


def parallel_image_download(
    urls: list[str],
    outs: list[str],
    crops: list[list[float]],
    skip_if_exists=True,
    **tpe_kwargs,
):
    assert len(urls) == len(outs) and len(urls) == len(crops)
    parallel_map(wget_if_not_already_downloaded, [(*d, skip_if_exists) for d in zip(urls, outs, crops)], **tpe_kwargs)


def download_stimuli_images(
    coco_ids: list[int],
    splits: list[str],
    crops: list[list[float]],
) -> list[str]:
    assert_data_dir()

    # sub directories (ie val2017, train2017) to save to
    for split in splits:
        os.makedirs(os.path.join(DATA_DIR, split), exist_ok=True)

    # links to download
    links = []
    paths = []
    for (link, filename), split in zip(coco_image_links(coco_ids, splits), splits):
        links.append(link)
        paths.append(os.path.join(DATA_DIR, split, filename))

    # download on max possible threads in parallel
    parallel_image_download(links, paths, crops)

    return paths


def df_download_stimuli_images(df: pd.DataFrame) -> list[str]:
    assert "cocoId" in df.columns and "cocoSplit" in df.columns and "cropBox" in df.columns
    return download_stimuli_images(
        coco_ids=df["cocoId"],
        splits=df["cocoSplit"],
        crops=df["cropBox"],
    )


def open_stimuli_image(coco_id: int, coco_split: str) -> Image.Image:
    filename = coco_filename(coco_id)
    path = os.path.join(DATA_DIR, coco_split, filename)
    assert os.path.exists(path)

    im = Image.open(path).convert("RGB")
    return im


def df_row_open_stimuli_image(row: pd.DataFrame):
    return open_stimuli_image(coco_id=row["cocoId"], coco_split=row["cocoSplit"])


def str_subject(subject: int):
    return f"subj{str(subject).zfill(2)}"


def str_session(session: int):
    return f"session{str(session).zfill(2)}"


def betas_dir():
    return os.path.join(DATA_DIR, "betas")


def download_vol_betas_subject_session(subject_id: str, session_id: str):
    subject_dir = os.path.join(betas_dir(), subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    filename = f"betas_{session_id}.nii.gz"
    download_to = os.path.join(betas_dir(), subject_id, filename)
    if os.path.exists(download_to):
        return download_to

    link = f"nsddata_betas/ppdata/{subject_id}/func1pt8mm/betas_fithrf_GLMdenoise_RR/{filename}"
    download_nsd(link, download_to)

    return download_to


def download_all_session_betas(subject_id: str):
    parallel_map(download_vol_betas_subject_session, [(subject_id, str_session(i)) for i in range(1, NUM_SESSIONS[subject_id] + 1)])


def iter_subject_sessions():
    for subject_idx in range(NUM_SUBS):
        subject_id = str_subject(subject_idx + 1)
        for session_idx in range(NUM_SESSIONS[subject_id]):
            session_id = str_session(session_idx + 1)
            yield subject_id, session_id


def download_all_betas():
    assert_data_dir()
    parallel_map(download_vol_betas_subject_session, list(iter_subject_sessions()))


def uncompress_nii_betas(img: NiiImage) -> NiiImage:
    ndarray = undo_betas_compression(img)
    new_header = img.header.copy()
    new_header.set_data_dtype(np.float32)
    unzipped_img = nib.nifti1.Nifti1Image(ndarray, affine=img.affine, header=new_header)
    return unzipped_img


def get_shape_data(filename):
    return nib.load(filename).header.get_data_shape()


def drop_subject_rep_cols(df: pd.DataFrame):
    df.drop([f"subject{subject_idx + 1}_rep{rep_id}" for subject_idx in range(NUM_SUBS) for rep_id in range(3)], inplace=True, axis=1)


def drop_subject_cols(df: pd.DataFrame):
    df.drop([f"subject{subject_idx + 1}" for subject_idx in range(NUM_SUBS)], inplace=True, axis=1)


# copied directly (with some renaming) from https://github.com/clane9/NSD-Flat/blob/main/convert_nsd_annotations.py#L277
def unroll_stimuli_trials(stim_info: pd.DataFrame) -> pd.DataFrame:
    long_stim_info = []

    for i in tqdm(range(len(stim_info))):
        row = stim_info.iloc[i].to_dict()
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


def add_session_and_local_trial_info(df: pd.DataFrame):
    df["i"] = range(len(df))
    session_id = [None] * len(df)
    session_trial_id = [None] * len(df)

    for subject_idx in tqdm(range(NUM_SUBS)):
        subject_id = str_subject(subject_idx + 1)
        for session_idx in range(NUM_SESSIONS[subject_id]):
            for trial_idx in range(TRIALS_PER_SESSION):
                global_trial_idx = session_idx * TRIALS_PER_SESSION + trial_idx
                df_idx = df.loc[subject_idx + 1, global_trial_idx + 1]["i"]
                session_id[df_idx] = session_idx + 1
                session_trial_id[df_idx] = trial_idx + 1
    df.drop("i", axis=1, inplace=True)
    df["sessionId"] = session_id
    df["sessionTrialId"] = session_id


def mask_unran_trials(df: pd.DataFrame):
    df["i"] = range(len(df))
    mask = [False] * len(df)

    for subject_idx in tqdm(range(NUM_SUBS)):
        subject_id = str_subject(subject_idx + 1)
        for session_idx in range(NUM_SESSIONS[subject_id]):
            for trial_idx in range(TRIALS_PER_SESSION):
                global_trial_idx = session_idx * TRIALS_PER_SESSION + trial_idx
                mask[df.loc[subject_idx + 1, global_trial_idx + 1]["i"]] = True

    df.drop("i", axis=1, inplace=True)
    return mask


def trials_path():
    return os.path.join(DATA_DIR, "trials.parquet")


def df_trials():
    assert_data_dir()

    # if already computed, just read from cache
    cache_path = trials_path()
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # otherwise process the info
    df = unroll_stimuli_trials(df_stimuli_info())
    mask = mask_unran_trials(df)
    df = df[mask].copy()
    add_session_and_local_trial_info(df)
    df.reset_index(inplace=True)

    # save to cache for next time I call
    df.to_parquet(cache_path)
    return df


def undo_betas_compression(img: NiiImage) -> np.ndarray:
    """
    On https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD they say
    that you need to convert back to float, then divide by 300
    """
    return img.get_fdata(dtype=np.float32) / 300.0


def load_session_betas_nii(subject_id: str, session_id: str, session_trial_id: int) -> NiiImage:
    filename = os.path.join(betas_dir(), subject_id, f"betas_{session_id}.nii.gz")
    img: NiiImage = nib.load(filename)
    trial_data = img.slicer[..., session_trial_id - 1]
    return uncompress_nii_betas(trial_data)


def init_data_dir(data_dir: str):
    # used for subsequent calls
    global DATA_DIR
    DATA_DIR = data_dir
    os.makedirs(DATA_DIR, exist_ok=True)


def title_print(s: str):
    print(f"{Fore.white}{Back.magenta}> Data in {s}{Style.reset}")


def subtitle_print(s: str):
    print(f"{Fore.white}{Back.green}>> {s}{Style.reset}")
