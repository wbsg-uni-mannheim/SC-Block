#!/bin/bash

export DATA_DIR=/home/alebrink/development/SCBlock/data/
export PYTHONPATH=/home/alebrink/development/SCBlock
export ES_INSTANCE=wifo5-33.informatik.uni-mannheim.de
export CUDA_VISIBLE_DEVICES=0


export MODELNAME={path/to/selected_model}
python src/strategy/open_book/indexing/index_faiss_entity.py --dataset={data_set} --bi_encoder_name='supcon_bi_encoder'\
			--model_name=$MODELNAME \
			--base_model='roberta-base' --with_projection=False --dimensions=768
