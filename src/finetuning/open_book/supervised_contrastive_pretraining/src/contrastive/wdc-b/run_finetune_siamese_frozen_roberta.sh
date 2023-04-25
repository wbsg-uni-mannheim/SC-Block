#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
PREBATCH=$2
LR=$3
TEMP=$4
EPOCHS=$5
PREEPOCHS=$6
AUG=$7
PREAUG=$8

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
#export CUDA_VISIBLE_DEVICES=7

python run_finetune_siamese.py \
	--model_pretrained_checkpoint /ceph/alebrink/contrastive-product-matching/reports/contrastive/wdcproducts80cc20rnd050un-clean-$AUG$PREBATCH-$LR-$TEMP-$PREEPOCHS-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name=wdcproducts80cc20rnd050un \
    --train_file /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/interim/wdcproducts80cc20rnd050un/wdcproducts80cc20rnd050un-train.json.gz \
	--validation_file /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/interim/wdcproducts80cc20rnd050un/wdcproducts80cc20rnd050un-train.json.gz \
	--test_file /home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/interim/wdcproducts80cc20rnd050un/wdcproducts80cc20rnd050un-gs.json.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir /ceph/alebrink/contrastive-product-matching/reports/contrastive-ft-siamese/wdcproducts80cc20rnd050un-clean-$AUG$PREBATCH-$PREAUG$LR-$TEMP-$PREEPOCHS-$EPOCHS-frozen-roberta-base/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \