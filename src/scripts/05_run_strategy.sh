#!/bin/bash

export ES_INSTANCE=wifo5-33.informatik.uni-mannheim.de
export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0


python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/abt-buy/blocking_experiments_abt-buy.yml' --worker=2
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/walmart-amazon_1/blocking_experiments_walmart-amazon_1.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/amazon-google/blocking_experiments_amazon-google.yml' --worker=2
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_s_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_s_train_l.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_m_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_m_train_l.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_l_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_l_train_l.yml' --worker=4


#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/abt-buy/blocking_experiments_abt-buy_switched.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/amazon-google/blocking_experiments_amazon-google_switched.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/walmart-amazon_1/blocking_experiments_walmart-amazon_1_switched.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_s_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_s_train_l_switched.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_m_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_m_train_l_switched.yml' --worker=4
#python src/strategy/run_strategy.py --path_to_config='/home/alebrink/development/sc-block/config/experiments/wdcproducts80cc20rnd050un_block_l_train_l/blocking_experiments_wdcproducts80cc20rnd050un_block_l_train_l_switched.yml' --worker=4


