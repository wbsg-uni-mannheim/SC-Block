#!/bin/bash

export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)

python src/finetuning/open_book/contrastive_pretraining/src/processing/preprocess/preprocess-deepmatcher-datasets.py
python src/finetuning/open_book/contrastive_pretraining/src/processing/contrastive/prepare-data-deepmatcher.py