#!/bin/bash

export DATA_DIR=$(pwd)/data
export PYTHONPATH=$(pwd)
export ES_INSTANCE=wifo5-33.informatik.uni-mannheim.de
export CUDA_VISIBLE_DEVICES=6

#epochs=(5 10 30)

#for EPOCHS in "${epochs[@]}"
#do
#    export DATASET=abt-buy
#    export MODELNAME=/ceph/alebrink/contrastive-product-matching/reports/contrastive/abtbuy-clean-$BATCHSIZE-5e-5-0.07-20-roberta-base
#
#    python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#          --model_name=$MODELNAME \
#          --base_model='roberta-base' --with_projection=False --dimensions=768
#
#export DATASET=abt-buy
#export MODELNAME="text-embedding-3-small"
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='openai_bi_encoder'\
#      --model_name=$MODELNAME --dimensions=1536 \

#export DATASET=amazon-google
#export MODELNAME="text-embedding-3-small"
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='openai_bi_encoder'\
#      --model_name=$MODELNAME --dimensions=1536 \

#export DATASET=walmart-amazon_1
#export MODELNAME="text-embedding-3-small"
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='openai_bi_encoder'\
#    --model_name=$MODELNAME --dimensions=1536 \

export DATASET=wdcproducts80cc20rnd050un_block_s_train_l
export MODELNAME="text-embedding-3-small"
python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='openai_bi_encoder'\
    --model_name=$MODELNAME --dimensions=1536 \

export DATASET=wdcproducts80cc20rnd050un_block_m_train_l
export MODELNAME="text-embedding-3-small"
python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='openai_bi_encoder'\
    --model_name=$MODELNAME --dimensions=1536 \
#
#    export DATASET=walmart-amazon_1
#    export MODELNAME=/ceph/alebrink/contrastive-product-matching/reports/contrastive/walmart-amazon-clean-$BATCHSIZE-5e-5-0.07-20-roberta-base
#    python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#          --model_name=$MODELNAME \
#          --base_model='roberta-base' --with_projection=False --dimensions=768

#export DATASET=wdcproducts80cc20rnd050un_block_s_train_l
#export MODELNAME=/ceph/alebrink/contrastive-product-matching/reports/contrastive/wdcproducts80cc20rnd050un-simclr-del1024-5e-5-0.07-20-roberta-base

#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768

#export DATASET=wdcproducts80cc20rnd050un_block_m_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768

#export DATASET=wdcproducts80cc20rnd050un_block_l_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768
#
## Index using SBERT Model
#export DATASET=wdcproducts80cc20rnd050un_block_s_train_l
#export MODELNAME=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/wdcproducts80cc20rnd000un_10_train_valid_cosine
#
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768
#
#export DATASET=wdcproducts80cc20rnd050un_block_m_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768
#
#export DATASET=wdcproducts80cc20rnd050un_block_l_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='supcon_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=1024 \
#      --base_model='roberta-base' --with_projection=False --dimensions=768

#done

#export MODELNAME=/ceph/alebrink/tableAugmentation/data/embedding/cc.en.300.bin
#export DATASET=abt-buy
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME \
#      --base_model='fasttext' --with_projection=False --dimensions=300

#export DATASET=amazon-google
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME \
#      --base_model='fasttext' --with_projection=False --dimensions=300
#
#export DATASET=walmart-amazon_1
#
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME \
#      --base_model='fasttext' --with_projection=False --dimensions=300
#
#export DATASET=wdcproducts80cc20rnd050un_block_s_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=128 \
#      --base_model='fasttext' --with_projection=False --dimensions=300
#
#export DATASET=wdcproducts80cc20rnd050un_block_m_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=128 \
#      --base_model='fasttext' --with_projection=False --dimensions=300
#
#export DATASET=wdcproducts80cc20rnd050un_block_l_train_l
#python src/strategy/indexing/index_faiss_entity.py --dataset=$DATASET --bi_encoder_name='fasttext_bi_encoder'\
#      --model_name=$MODELNAME --batch_size=128 \
#      --base_model='fasttext' --with_projection=False --dimensions=300


