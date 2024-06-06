#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

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
