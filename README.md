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

`--input` The first column of the required input file is the phenotypic value, and the second and subsequent columns are the PCs or numerical type genotype information<br>
`--batch_size` The number of training samples processed together in one forward/backward pass during model training<br>
`--dropout` The proportion of neurons randomly dropped during training<br>
`--epoch` The number of complete passes through the entire training dataset during model training<br>
`--lr` Learning rate<br>
`n_trials` The number of hyperparameter searches<br>
`--seed` Random seed<br>

```shell

python PigRGP.py --input ${tra}.pc${i}.input --seed 42 --batch_size 128 256 --dropout 0.4 0.6 --epoch 32 --lr 0.001 0.01 --n_trials 10


```

