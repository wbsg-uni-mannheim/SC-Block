#!/bin/bash

export PYTHONPATH=/home/alebrink/development/table-augmentation-framework/
export CUDA_VISIBLE_DEVICES=3

chmod +x abtbuy/run_pretraining_simclr_clean_roberta.sh
./abtbuy/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
chmod +x abtbuy/run_pretraining_barlow_clean_roberta.sh
./abtbuy/run_pretraining_barlow_clean_roberta.sh 64 5e-5 0.07 20


chmod +x amazongoogle/run_pretraining_simclr_clean_roberta.sh
./amazongoogle/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
chmod +x amazongoogle/run_pretraining_barlow_clean_roberta.sh
./amazongoogle/run_pretraining_barlow_clean_roberta.sh 64 5e-5 0.07 20


chmod +x dblpgooglescholar/run_pretraining_simclr_clean_roberta.sh
./dblpgooglescholar/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20
chmod +x dblpgooglescholar/run_pretraining_barlow_clean_roberta.sh
./dblpgooglescholar/run_pretraining_barlow_clean_roberta.sh 64 5e-5 0.07 20



chmod +x walmartamazon/run_pretraining_simclr_clean_roberta.sh
./walmartamazon/run_pretraining_simclr_clean_roberta.sh 1024 5e-5 0.07 20

chmod +x walmartamazon/run_pretraining_barlow_clean_roberta.sh
./walmartamazon/run_pretraining_barlow_clean_roberta.sh 64 5e-5 0.07 20
