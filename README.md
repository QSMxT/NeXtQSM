# NeXtQSM
A complete deep learning pipeline for data-consistent quantitative susceptibility mapping trained with hybrid data

https://doi.org/10.48550/arXiv.2107.07752

## Download weights

```bash
pip install cloudstor
python -c "import cloudstor; cloudstor.cloudstor(url='https://cloudstor.aarnet.edu.au/plus/s/5OehmoRrTr9XlS5', password='').download('', 'nextqsm-weights.tar')"
tar xf nextqsm-weights.tar -C checkpoints/
rm nextqsm-weights.tar
```