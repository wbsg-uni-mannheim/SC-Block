#!/bin/bash

export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)
export ES_INSTANCE= # ES_Instance
export CUDA_VISIBLE_DEVICES=5


# Load SC-Block model for Abt-Buy
export DATASET=abt-buy
export MODELNAME=$(pwd)/reports/contrastive/abtbuy-clean-1024-5e-5-0.07-20-roberta-base-

python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
      --model_name=$MODELNAME \
      --base_model='roberta-base' --with_projection=False --dimensions=768

