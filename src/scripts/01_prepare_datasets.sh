#!/bin/bash

export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)

echo $PYTHONPATH

# List of datasets
datasets=("abt-buy" "amazon-google" "walmart-amazon_1")
#datasets=("wdcproducts80cc20rnd050un_block_s_train_l" "wdcproducts80cc20rnd050un_block_m_train_l" "wdcproducts80cc20rnd050un_block_l_train_l")

for DATASET in "${datasets[@]}"
do
    python src/data/deepmatcher/convert_table_to_query_table.py --dataset=$DATASET --table_name=tableA
    python src/data/deepmatcher/convert_table_to_table_format.py --dataset=$DATASET --table_name=tableB

    # Switch table names
    #python src/data/deepmatcher/convert_table_to_query_table.py --dataset=$DATASET --table_name=tableB
    #python src/data/deepmatcher/convert_table_to_table_format.py --dataset=$DATASET --table_name=tableA
done