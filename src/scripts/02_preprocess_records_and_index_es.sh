#!/bin/bash

export ES_INSTANCE=# ES_Instance
export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)

# List of datasets
datasets=("abt-buy" "amazon-google" "walmart-amazon_1")
#datasets=("wdcproducts80cc20rnd050un_block_s_train_l" "wdcproducts80cc20rnd050un_block_m_train_l" "wdcproducts80cc20rnd050un_block_l_train_l")

for DATASET in "${datasets[@]}"
do
    python src/strategy/indexing/preprocess_records_and_index_es.py --dataset=$DATASET
    #python src/strategy/indexing/preprocess_records_and_index_es.py --dataset=$DATASET --switched=True
done