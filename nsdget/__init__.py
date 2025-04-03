from __future__ import annotations
import pandas as pd
from .core import (
    title_print,
    subtitle_print,
    str_session,
    str_subject,
    open_stimuli_image,
    download_all_betas,
    df_download_stimuli_images,
    df_stimuli_info,
    df_trials,
    init_data_dir,
    load_session_betas_nii,
    DATA_DIR,
    NiiImage,
)


def nsd_single_trial_betas(row: pd.Series) -> NiiImage:
    """
    numpy=True will return a np.ndarray
    numpy=False will return a NiiImage slice (like what you get from nib.load())
    """
    cols = row.keys()
    assert "subjectId" in cols
    assert "sessionId" in cols
    assert "sessionTrialId" in cols

    subject_id = str_subject(row["subjectId"])  # between [1, 8]
    session_id = str_session(row["sessionId"])  # between [1, max_sessions for the subject]
    session_trial_id = row["sessionId"]  # local trial id between [1, 750]

    return load_session_betas_nii(subject_id, session_id, session_trial_id)


def nsd_coco_image(row: pd.Series):
    cols = row.keys()
    assert "cocoId" in cols
    assert "cocoSplit" in cols

    coco_id = row["cocoId"]
    coco_split = row["cocoSplit"]

    return open_stimuli_image(coco_id, coco_split)


def nsd_betas_images_trials(save_to: str = DATA_DIR) -> pd.DataFrame:
    title_print(f"Data points to {save_to}")
    init_data_dir(save_to)

    subtitle_print("Downloading All Betas (will take some time >1hr)")
    download_all_betas()

    subtitle_print("Downloading 73k COCO Images")
    df_download_stimuli_images(df_stimuli_info())

    subtitle_print("Unrolling stimulu info into dataframe")
    return df_trials()


def main():
    df = nsd_betas_images_trials(save_to="./nsdata/")
    print(df.head())


if __name__ == "__main__":
    main()
