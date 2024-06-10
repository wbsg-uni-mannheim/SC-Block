#!/bin/bash

export ES_INSTANCE= # ES_Instance
export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=5


python src/strategy/run_strategy.py --path_to_config=$(pwd)'/config/experiments/abt-buy/blocking_experiments_abt-buy.yml' --worker=2
#python src/strategy/run_strategy.py --path_to_config=$(pwd)'/config/experiments/walmart-amazon_1/blocking_experiments_walmart-amazon_1.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config=$(pwd)'config/experiments/amazon-google/blocking_experiments_amazon-google.yml' --worker=2
#python src/strategy/run_strategy.py --path_to_config=$(pwd)'config/experiments/wdcproducts80cc20rnd050un_block_s_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_s_train_l.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config=$(pwd)'sc-block/config/experiments/wdcproducts80cc20rnd050un_block_m_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_m_train_l.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config=$(pwd)'/config/experiments/wdcproducts80cc20rnd050un_block_l_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_l_train_l.yml' --worker=4
