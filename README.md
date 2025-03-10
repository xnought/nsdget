# 🧠🏞️ nsdget

[![PyPI - Version](https://img.shields.io/pypi/v/nsdget.svg)](https://pypi.org/project/nsdget) 

**nsdget: Download Natural Scenes Dataset images and fMRI without downloading the entire dataset.**

Quickly get a subset of stimuli (images shown to subject) and corresponding fMRI data from the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

Not affiliated with Natural Scenes Dataset. I just wanted an easier and quicker way to download the data I needed from them (hence this package).

**Roadmap**

- [x] Functions to download stimuli (images) w/out downloading the 40gb total image object. Just download images directly from COCO.
- [ ] Function to download fMRI data  
- [ ] Document functions better
- [x] Publish to PyPi

## Usage

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

- https://naturalscenesdataset.org/
- https://cocodataset.org/#home
