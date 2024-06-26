## Overview

This is the PyTorch implementation of paper [M-Net: A Lightweight Network Based on Multilayer Perceptron for Massive MIMO CSI Feedback](https://xplorestaging.ieee.org/document/10464906).

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task.

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── M-Net  # The cloned M-Net repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── in_04.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Train M-Net from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start advanced scheme aided M-Net training from scratch. Change scenario using `--scenario` and change compression ratio with `--cr`.

``` bash
python /home/M-Net/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 1000 \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --scheduler cosine \
  --seed 196 \
  --expansion 1 \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction

The main results reported in our paper are presented as follows. All the listed results can be found in Table1 of our paper, you can easily reproduce the following results.


Scenario | Compression Ratio | NMSE | Flops | Parameters
:--: | :--: | :--: | :--: | :--:
indoor | 1/4 | -31.48 | 3.93M | 2150K
indoor | 1/8 | -21.00 | 2.88M | 1101K
indoor | 1/16 | -15.13 | 2.36M | 576K
indoor | 1/32 | -11.29 | 2.10M | 314K
indoor | 1/64 | -8.41 | 1.97M | 153K
outdoor | 1/4 | -11.90 | 3.93M | 2150K
outdoor | 1/8 | -8.23 | 2.88M | 1101K
outdoor | 1/16 | -5.67 | 2.36M | 576K
outdoor | 1/32 | -3.68 | 2.10M | 314K
outdoor | 1/64 | -2.39 | 1.97M | 153K



## Acknowledgment

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 
