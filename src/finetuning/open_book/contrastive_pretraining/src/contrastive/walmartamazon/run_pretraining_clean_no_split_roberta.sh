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
AUG=$5

export PYTHONPATH=$(pwd)
#export CUDA_VISIBLE_DEVICES=0

python run_pretraining_deepmatcher.py \
    --do_train \
	--dataset_name=walmart-amazon \
	--clean=True \
	--split=False \
    --train_file $(pwd)/src/finetuning/open_book/contrastive_pretraining/data/processed/walmart-amazon/contrastive/walmart-amazon-train.pkl.gz \
	--id_deduction_set $(pwd)/src/finetuning/open_book/contrastive_pretraining/data/interim/walmart-amazon/walmart-amazon-train.json.gz \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir $(pwd)/reports/contrastive/walmart-amazon-clean-no-split-$AUG$BATCH-$LR-$TEMP-$EPOCHS-roberta-base/ \
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
