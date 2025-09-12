# PigRGP

A deep learning model for pig genome prediction based on residual network.

## Getting Started

You can refer to the follow command to install `PigRGP`

```shell
conda create -n PigRGP python=3.12
conda activate PigRGP
cd PigRGP
pip install -r requirements.txt
```

## Usage

`--input` The first column of the required input file is the phenotypic value, and the second and subsequent columns are the PCs or numerical type genotype information
`--batch_size` The number of training samples processed together in one forward/backward pass during model training
`--dropout` The proportion of neurons randomly dropped during training
`--epoch` The number of complete passes through the entire training dataset during model training
`--lr` Learning rate
`n_trials` The number of hyperparameter searches
`--seed` Random seed

```shell

python jiada_val230.py --input ${tra}.pc${i}.input --seed 42 --batch_size 128 256 --dropout 0.4 0.6 --epoch 32 --lr 0.001 0.01 --n_trials 10


```

