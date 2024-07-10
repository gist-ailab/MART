# MART

This repo is the official implementation of "***MART: MultiscAle Relational Transformer Networks for Trajectory Prediction***"

The source code will be released soon!

## Model Architecture
![model](./figures/main_model.png)

## Getting Started

### Environment Setup


1. Set up a python environment
```
conda create -n mart python=3.8
conda activate mart
```

2. Install requirements using the following command.
```
pip install -r requirements.txt
```

## Train & Evaluation

* Trained and evaluated on NVIDIA GeForce RTX 3090 with python 3.8.

* The dataset is included in ```./datasets/nba/```

* For reproducibility, we set the seed with 1.

### Train MART on the NBA dataset

```
python main_nba.py --config mart_nba.ini --gpu $GPU_IDs
```

### Test MART on the NBA dataset after training
```
python main_nba.py --config mart_nba.ini --gpu $GPU_IDs --test
```

## Main Results
### NBA dataset
* minADE_20: 0.727 [m]
* minFDE_20: 0.903 [m]

### How to reproduce results

* The checkpoint is included in ```./checkpoints/mart_nba_reproduce/```

```
python main_nba.py --config mart_nba_reproduce.ini --gpu $GPU_IDs --test
```

## Acknowledgement
* The part of the code about the feature initialization is adapted from ([GroupNet](https://github.com/MediaBrain-SJTU/GroupNet)).
* Thanks for sharing the preprocessed NBA dataset and dataloader ([LED](https://github.com/MediaBrain-SJTU/LED)).
* Thanks for providing the code of the Relational Transformer ([RT](https://github.com/CameronDiao/relational-transformer)). We implemented the RT from ```jax``` to ```pytorch```.
