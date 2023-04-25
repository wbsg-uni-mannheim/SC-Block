#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
EPOCHS=$3

AUG="del"

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
#export CUDA_VISIBLE_DEVICES=0

python run_pretraining_barlow_deepmatcher.py \
    --do_train \
	--dataset_name=abt-buy \
	--clean=True \
    --train_file /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/processed/abt-buy/contrastive/abt-buy-train.pkl.gz \
	--id_deduction_set /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/interim/abt-buy/abt-buy-train.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /ceph/alebrink/contrastive-product-matching/reports/contrastive/abtbuy-barlow-$AUG$BATCH-$LR-$TEMP-$EPOCHS-roberta-base/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \
