#!/bin/bash

export ES_INSTANCE=wifo5-33.informatik.uni-mannheim.de
export DATA_DIR=/home/alebrink/development/SCBlock/data/
export PYTHONPATH=/home/alebrink/development/SCBlock
export CUDA_VISIBLE_DEVICES=0

python src/strategy/run_strategy --path_to_config={path_to_config} --worker=4
