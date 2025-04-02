from __future__ import annotations
import pandas as pd
from .core import (
    str_session,
    str_subject,
    load_session_betas,
    open_stimuli_image,
    download_all_betas,
    df_download_stimuli_images,
    df_stimuli_info,
    df_trials,
    init_data_dir,
    DATA_DIR,
)

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()


def nsd_single_trial_betas(row: pd.Series):
    cols = row.keys()
    assert "subjectId" in cols
    assert "sessionId" in cols
    assert "sessionTrialId" in cols

    subject_id = str_subject(row["subjectId"])  # between [1, 8]
    session_id = str_session(row["sessionId"])  # between [1, max_sessions for the subject]
    session_trial_id = row["sessionId"]  # local trial id between [1, 750]
    d = load_session_betas(subject_id, session_id, session_trial_id)
    return d


def nsd_coco_image(row: pd.Series):
    cols = row.keys()
    assert "cocoId" in cols
    assert "cocoSplit" in cols

    coco_id = row["cocoId"]
    coco_split = row["cocoSplit"]

    return open_stimuli_image(coco_id, coco_split)


def nsd_betas_images_trials(save_to: str = DATA_DIR) -> pd.DataFrame:
    print(f"{Fore.CYAN}> Data in {DATA_DIR}{Style.RESET_ALL}")
    init_data_dir(save_to)

    print(f"{Fore.MAGENTA}>> Downloading all Betas{Style.RESET_ALL}")
    download_all_betas()

    print(f"{Fore.MAGENTA}>> Downloading all COCO Images{Style.RESET_ALL}")
    df_download_stimuli_images(df_stimuli_info())

    print(f"{Fore.MAGENTA}>> Unrolling stimulu info into dataframe{Style.RESET_ALL}")
    return df_trials()


def main():
    df = nsd_betas_images_trials(save_to="./nsdata/")
    print(nsd_single_trial_betas(df.iloc[0]).shape)
    nsd_coco_image(df.iloc[0]).show()


if __name__ == "__main__":
    main()
