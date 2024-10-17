# NeXtQSM

A complete *pre-trained* deep learning pipeline for data-consistent quantitative susceptibility mapping trained with hybrid data.

Cognolato, F., O'Brien, K., Jin, J., Robinson, S., Laun, F. B., Barth, M., & Bollmann, S. (2023). NeXtQSM—A complete deep learning pipeline for data-consistent Quantitative Susceptibility Mapping trained with hybrid data. *Medical Image Analysis*, 84, 102700. https://doi.org/10.1016/j.media.2022.102700.

## Installation

Create a conda environment. Python version **3.8** is recommended, higher versions might be incompatible:  
```bash
conda create -n nextqsm python=3.8 
conda activate nextqsm 
```

NeXtQSM is available via pip:

```bash
pip install nextqsm
```

## Downloading weights

NeXtQSM requires a set of training weights for inference (~150 MB) which are automatically downloaded when you run NeXtQSM on a dataset.

You can also manually download the weights using:

```bash
nextqsm --download_weights
```

## Usage

Run NeXtQSM using the following command, providing an unwrapped frequency map (unitless and scaled to ppm) and brain mask as inputs in the NIfTI file format:

```bash
nextqsm [phase_file] [mask_file] [out_file]
```

