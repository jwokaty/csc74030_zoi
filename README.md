# csc74030 - Zone of Inhibition

## Project Background

Gram negative bacteria, like E. coli, have a strong barrier that makes them resistant to
antibiotics [3]. A common two-step process first evaluates the minimal inhibitory
concentrations (MICs) that depict inhibition of bacterial growth and can be used to
classify a specimen as susceptible (responsive to treatment) or resistant [2]. Resistance is
confirmed via a secondary diagnostic [2]. Kirby-Baurer disk diffusion is a common method
used to initially screen for resistance; however, “reading and interpretation” are difficult
to reproduce [2].

This project aims to develop a system to classify antibiotic resistance observed on agar
plates depicting the minimal inhibitory concentrations (MICs) of bacteria to antibiotics
labeled by discs. The system will employ computer vision techniques, such as
enhancement, edge detection and object segmentation, to preprocess regions of interest
of images taken by a SIRweb/SIRscan system. It will also use a pretrained CNN to analyze
the MICs before classifying resistance as susceptible, intermediate, or resistant.

The pipeline has four stages: preprocessing (denoising and ROI extraction),
image pyramids and thresholding, feature extraction, and a classifier.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for the Python environment
- [Quarto](https://quarto.org/docs/get-started/) (optional) to render slides. It is installed
  separately and is not a Python dependency.
- Git, to clone FBCNN (see below)

## Setup

```
uv sync
```

This creates `.venv` from `pyproject.toml` and `uv.lock`. It installs PyTorch,
so the first run takes a while.

### FBCNN

Denoising uses [FBCNN](https://github.com/jiaxi-jiang/FBCNN). It is not part of
this repository. Clone it into `external/`:

```
git clone https://github.com/jiaxi-jiang/FBCNN.git external/FBCNN
```

The model weights are downloaded automatically the first time they are needed.

### Data

The data is not stored in this repository. After cloning, copy it into `data/`,
which should contain the top-level folders `images_original` and
`images_measured`, along with the other data files.

Denoised images are written to `data/processed/`.

`data/roi_labels.csv` is tracked in this repository.

## Configuration

Data and external paths are set in `config.toml`. To use different paths on
your own computer, copy `config.local.example.toml` to `config.local.toml` and
change only the paths you need. The local file is not tracked by git.

Notebooks must run with the project environment. In Jupyter, choose the
`.venv` kernel created by `uv sync`.

## Project layout

| Path | Contents |
| --- | --- |
| `src/csc74030_zoi/paths.py` | Locations of data and external files |
| `src/csc74030_zoi/preprocessing.py` | Image and label loading, denoising |
| `src/csc74030_zoi/pyramids.py` | Gaussian and Laplacian pyramids, blending |
| `src/csc74030_zoi/thresholding.py` | Global and local thresholding |
| `src/csc74030_zoi/features.py` | Texture and shape features |
| `src/csc74030_zoi/model.py` | Classifier and cross-validation |
| `src/csc74030_zoi/plots.py` | Plotting helpers |
| `notebooks/` | One numbered notebook per homework |
| `data/` | Copied in after cloning; denoised images go in `data/processed/` |
| `slides/` | Rendered slides (not tracked by git) |

## Rendering slides

```
uv run quarto render notebooks/<notebook>.ipynb
```

Quarto uses the outputs saved in the notebook. Settings are in `_quarto.yml`.

## Citations

1. Giske, C. G., Bressan, M., Fiechter, F., Hinic, V., Mancini, S., Nolte, O., & Egli, A. (2024).
Image dataset of disk diffusion assay scanned with the SIRscan system [Dataset]. Dryad.
https://doi.org/10.5061/dryad.5dv41nsfj
2. Giske, C. G., Bressan, M., Fiechter, F., Hinic, V., Mancini, S., Nolte, O., & Egli, A. (2024).
GPT-4 based AI agents - the new expert system for detection of antimicrobial resistance
mechanisms? Journal of Clinical Microbiology. https://doi.org/10.1128/jcm.00689-24
3. Lab Tests Guide. (2023). Gram positive vs gram negative bacteria.
https://www.labtestsguide.com/gram-positive-vs-gram-negative-bacteria
4. FBCNN: Jiang, J., Zhang, K., and Timofte, R. "Towards Flexible Blind JPEG 
Artifacts Removal." ICCV 2021. Code: <https://github.com/jiaxi-jiang/FBCNN>
(Apache License 2.0). The denoising code is adapted from `main_test_fbcnn_gray.py`.

## License

Apache 2.0
