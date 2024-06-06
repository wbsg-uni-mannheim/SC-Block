#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# Walmart Amazon Models
./walmartamazon/run_pretraining_clean_roberta.sh 512 5e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 256 5e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 128 5e-5 0.07 20

./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.05 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.06 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.09 20

./walmartamazon/run_pretraining_clean_roberta.sh 1024 1e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 3e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 7e-5 0.07 20

# Ablation Study Gamma

./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.05 20
./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.06 20
./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.09 20

./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.05 20
./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.06 20
./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.09 20

./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.05 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.06 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.09 20

./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.05 20
./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.06 20
./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.09 20
