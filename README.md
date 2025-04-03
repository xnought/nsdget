# 🧠🏞️ nsdget

[![PyPI - Version](https://img.shields.io/pypi/v/nsdget.svg)](https://pypi.org/project/nsdget) 

**nsdget: easily download and use the single trial betas 1.8mm and coco images from the Natural Scenes Dataset**

Note: I'm not affiliated with [Natural Scenes Dataset](https://naturalscenesdataset.org/). I just wanted an easier and quicker way to download the data I needed from them (hence this package).

Shoutout to https://github.com/clane9/NSD-Flat/ since I reused some of the functions from there. Thank you!

## Usage

To use the data, please fill out the [NSD Data Access Agreement](https://docs.google.com/forms/d/e/1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform?usp=send_form) first. Then download the `nsdget` python package:

**Install**

```bash
uv add nsdget
```

or

```bash
pip install nsdget
```

Then, to download the betas for all 8 subjects and 73k coco images do


```python
from nsdget import nsd_betas_images_trials, nsd_coco_image, nsd_single_trial_betas

df = nsd_betas_images_trials(save_to="./nsdata/") # 213k trials in a Pandas DataFrame
betas = nsd_single_trial_betas(df.iloc[0]) # first betas trial as Nifty Image 
image = nsd_coco_image(df.iloc[0])  # first trial stimulus as COCO PIL Image
```

Example above ran in [`example.ipynb`](./notebooks/example.ipynb).

Note that download happens only once. After the first slow run, subsequent runs will be very fast.

If you want the betas in numpy, just convert from nibabel image to numpy like

```python
nd_betas = betas.get_fdata() 
```

If you want the COCO image in numpy just convert the PIL Image to numpy like

```python
nd_image = np.asarray(image)
```

## Development

**Dev run**

```bash
uv sync
make run
```

**Deployment PyPi**

```bash
uv sync
TOKEN=... make publish # insert your PyPi token where ...
```

## References

- https://naturalscenesdataset.org/ (Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).)
- https://cocodataset.org/#home
- https://github.com/clane9/NSD-Flat/blob/b6851300ea3778eae7e4dbb88a85d71ce18cb9a5/generate_dataset.py#L157
