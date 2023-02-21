#!/bin/bash

export ES_INSTANCE=wifo5-33.informatik.uni-mannheim.de
export DATA_DIR=/home/alebrink/development/SCBlock/data/
export PYTHONPATH=/home/alebrink/development/SCBlock
export DATASET=abt-buy

python src/strategy/open_book/indexing/index_es_entity.py --dataset=$DATASET