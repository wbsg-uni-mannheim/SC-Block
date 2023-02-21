#!/bin/bash

export DATA_DIR=/home/alebrink/development/SCBlock/data/
export PYTHONPATH=/home/alebrink/development/SCBlock
export DATASET=abt-buy

python src/data/deepmatcher/convert_table_to_query_table.py --dataset=$DATASET
python src/data/deepmatcher/convert_table_to_table_format.py --dataset=$DATASET
