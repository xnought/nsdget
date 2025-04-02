# 🧠🏞️ nsdget

[![PyPI - Version](https://img.shields.io/pypi/v/nsdget.svg)](https://pypi.org/project/nsdget) 

**nsdget: Download Natural Scenes Dataset images and fMRI without downloading the entire dataset.**

Quickly get a subset of stimuli (images shown to subject) and corresponding fMRI data from the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

Not affiliated with Natural Scenes Dataset. I just wanted an easier and quicker way to download the data I needed from them (hence this package).

**Roadmap**

- [x] Create a dataframe with every trial
- [x] Have the dataframe point to images downloaded
- [x] Create way to get betas given dataframe row

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

No API reference yet. See examples of usage:

- Download stimuli images (coco data) example in [`view_info.ipynb`](./notebooks/view_info.ipynb)


## Development

```bash
uv sync
make
```

## References

- https://naturalscenesdataset.org/ (Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, Charest, Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).)
- https://cocodataset.org/#home
- https://github.com/clane9/NSD-Flat/blob/b6851300ea3778eae7e4dbb88a85d71ce18cb9a5/generate_dataset.py#L157
