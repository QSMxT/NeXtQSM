# NeXtQSM
A complete deep learning pipeline for data-consistent quantitative susceptibility mapping trained with hybrid data

https://doi.org/10.48550/arXiv.2107.07752

## Installation

NeXtQSM requires `tensorflow` and `packaging` as a dependencies. It also requires a set of training weights for inference (~150 MB).

Example setup using conda:

```bash
git clone https://github.com/QSMxT/NeXtQSM.git nextqsm
cd nextqsm/
conda create --name nextqsm python=3.8
conda activate nextqsm
conda install tensorflow packaging
```

### Download weights

Run the following from inside the repository folder to add the weights:

```bash
pip install cloudstor
python -c "import cloudstor; cloudstor.cloudstor(url='https://cloudstor.aarnet.edu.au/plus/s/5OehmoRrTr9XlS5', password='').download('', 'nextqsm-weights.tar')"
tar xf nextqsm-weights.tar -C checkpoints/
rm nextqsm-weights.tar
```

