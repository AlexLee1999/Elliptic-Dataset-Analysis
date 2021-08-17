# NML Final Bitcoin Antimoney laundry

## Table of content
- [Introduction](#introduction)
- [File tree](#file-tree)
- [Learning Model](#learning-model)
- [Usage](#usage)
- [Requirements](#requirements)

## Introduction
- Reproduce the experiment of the [paper](https://arxiv.org/abs/1908.02591)
- Add new features and data analysis for the elliptic data set
- Add new ML method to rerun the experiment

## File tree
```bash
.
├── elliptic_bitcoin_dataset
|   ├── full_data.csv
|   ├── Result.csv
|   ├── elliptic_txs_edgelist.csv
|   ├── elliptic_txs_features.csv
|   └── elliptic_txs_classes.csv
├── NML_Final-antimoney-laundry
└── txs
```
## Learning Model
- Linear Regression
- Logistic Regression
- MLP
- SVM
- Random Forest
- Logistic Regression with Cross Validation

## Usage

```bash
python3 -m pip install -r requirements.txt
./start.sh <options>
```
* -all | -A : run all script
* -raw | -R : run replication of the expriment
* -modified | -M : run with modified features
* -pca | -P : run with pca modified features
* -pcaf | -PF : run with pca modified features with feature selection
* -corrf | -CF : run with modified features with feature selection
* -stat | -S : run stat analysis


Requirements
-----------------------------
- python >=3.6
- numpy
- pandas
- scikit-learn
- seaborn
- statsmodels


