# NML Final Bitcoin Antimoney laundry

## Table of content
- [Introduction](#introduction)
- [File tree](#file-tree)
- [Usage](#usage)
- [Requirements](#requirements)

## Introduction

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

## Usage

```bash
pip install -r requirements.txt
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


