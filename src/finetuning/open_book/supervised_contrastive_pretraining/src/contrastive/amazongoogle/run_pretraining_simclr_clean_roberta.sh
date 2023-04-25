#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
TEMP=$3
EPOCHS=$4

AUG="del"

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
#export CUDA_VISIBLE_DEVICES=3

python run_pretraining_simclr_deepmatcher.py \
    --do_train \
	--dataset_name=amazon-google \
	--clean=True \
    --train_file /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/processed/amazon-google/contrastive/amazon-google-train.pkl.gz \
	--id_deduction_set /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/interim/amazon-google/amazon-google-train.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /ceph/alebrink/contrastive-product-matching/reports/contrastive/amazon-google-simclr-$AUG$BATCH-$LR-$TEMP-$EPOCHS-roberta-base/ \
	--temperature=$TEMP \
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
