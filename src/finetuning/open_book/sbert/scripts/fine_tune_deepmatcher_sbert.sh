#!/bin/bash

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
export DATA_DIR=/ceph/alebrink/tableAugmentation/data
export CUDA_VISIBLE_DEVICES=5

python sbert_fine-tuning_deepmatcher.py \
	--model_pretrained_checkpoint="roberta-base" \
  --dataset=abt-buy \
  --loss=cosine \
  --epochs=10 \
  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/abt-buy_10_train_valid_cosine

python sbert_fine-tuning_deepmatcher.py \
	--model_pretrained_checkpoint="roberta-base" \
  --dataset=amazon-google \
  --loss=cosine \
  --epochs=10 \
  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/amazon-google_10_train_valid_cosine

python sbert_fine-tuning_deepmatcher.py \
	--model_pretrained_checkpoint="roberta-base" \
  --dataset=dblp-googlescholar_1 \
  --loss=cosine \
  --epochs=10 \
  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/dblp-googlescholar_1_10_train_valid_cosine

python sbert_fine-tuning_deepmatcher.py \
	--model_pretrained_checkpoint="roberta-base" \
  --dataset=walmart-amazon_1 \
  --loss=cosine \
  --epochs=10 \
  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/walmart-amazon_1_10_train_valid_cosine

python sbert_fine-tuning_deepmatcher.py \
	--model_pretrained_checkpoint="roberta-base" \
  --dataset=wdcproducts80cc20rnd000un \
  --loss=cosine \
  --epochs=10 \
  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/wdcproducts80cc20rnd000un_10_train_valid_cosine

#python sbert_fine-tuning_deepmatcher.py \
#	--model_pretrained_checkpoint="roberta-base" \
#  --dataset=abt-buy \
#  --loss=cosine \
#  --epochs=20 \
#  --output_dir=/ceph/alebrink/tableAugmentation/data/models/open_book/sbert/abt-buy_20_cosine