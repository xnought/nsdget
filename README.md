# 🧠🏞️ nsdget

[![PyPI - Version](https://img.shields.io/pypi/v/nsdget.svg)](https://pypi.org/project/nsdget) 

**nsdget: Download Natural Scenes Dataset images and fMRI without downloading the entire dataset.**

Quickly get a subset of stimuli (images shown to subject) and corresponding fMRI data from the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

Not affiliated with Natural Scenes Dataset. I just wanted an easier and quicker way to download the data I needed from them (hence this package).

## Usage

To use the data, first submit a form to the NSD people: [NSD Data Access Agreement](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform?usp=send_form). Then,

**Install**

```bash
uv add nsdget
```

or

```bash
pip install nsdget
```

**API**

Simple API, all there is:

```python
from nsdget import nsd_betas_images_trials, nsd_coco_image, nsd_single_trial_betas

# download and use data
df: pd.DataFrame = nsd_betas_images_trials(save_to="./nsdata/")
betas: np.ndarray = sd_single_trial_betas(df.iloc[0]) # 1.8mm res fmri single trial from NSD for the given row (index 0 here) 
image0: PIL.Image = nsd_coco_image(df.iloc[0]) # crops how NSD did and gives you the PIL image (can easily be converted to numpy too)
```

## Development

```bash
uv sync
uv run nsdget
```

## References

- https://naturalscenesdataset.org/ (Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).)
- https://cocodataset.org/#home
- https://github.com/clane9/NSD-Flat/blob/b6851300ea3778eae7e4dbb88a85d71ce18cb9a5/generate_dataset.py#L157
