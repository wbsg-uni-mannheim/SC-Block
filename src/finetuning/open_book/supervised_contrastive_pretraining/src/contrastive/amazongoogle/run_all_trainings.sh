#!/bin/bash

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
export CUDA_VISIBLE_DEVICES=0

chmod +x amazongoogle/run_pretraining_simclr_clean_roberta.sh
./amazongoogle/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
chmod +x amazongoogle/run_pretraining_barlow_clean_roberta.sh
./amazongoogle/run_pretraining_barlow_clean_roberta.sh 64 5e-5 20

#chmod +x amazongoogle/run_pretraining_clean_roberta.sh
#./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 20
#./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 200

#chmod +x amazongoogle/run_finetune_siamese_frozen_roberta.sh
#./amazongoogle/run_finetune_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 20
#./amazongoogle/run_finetune_siamese_frozen_roberta.sh 64 1024 5e-5 0.07 50 200

#chmod +x amazongoogle/run_finetune_cross_encoder_roberta.sh
#./amazongoogle/run_finetune_cross_encoder_roberta.sh 64 5e-5 50