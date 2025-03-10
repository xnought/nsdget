# 🧠🏞️ nsdget

**nsdget: Natural Scenes Dataset Getter via Python API.**

Quickly get a subset of stimuli (images shown to subject) and corresponding fMRI data from the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

Not affiliated with Natural Scenes Dataset. I just wanted an easier and quicker way to download the data I needed from them (hence this package).

**Roadmap**

- [x] Functions to download stimuli (images) w/out downloading the 40gb total image object. Just download images directly from COCO.
- [ ] Function to download fMRI data  
- [ ] Document functions better
- [ ] Publish to PyPi

## Usage

- Download stimuli images (coco data) example in [`view_info.ipynb`](./notebooks/view_info.ipynb)

## Development

```bash
uv sync
make
```

## References

- https://naturalscenesdataset.org/
- https://cocodataset.org/#home
