#!/bin/bash

export DATA_DIR=/home/alebrink/development/SCBlock/data/
export PYTHONPATH=/home/alebrink/development/SCBlock

python src/finetuning/open_book/contrastive_pretraining/src/processing/preprocess/preprocess-deepmatcher-datasets.py
python src/finetuning/open_book/contrastive_pretraining/src/processing/contrastive/prepare-data-deepmatcher.py