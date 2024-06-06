#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

./abtbuy/run_pretraining_clean_roberta.sh 1024 1e-5 0.07 20
./abtbuy/run_pretraining_clean_roberta.sh 1024 3e-5 0.07 20
./abtbuy/run_pretraining_clean_roberta.sh 1024 7e-5 0.07 20

./amazongoogle/run_pretraining_clean_roberta.sh 1024 1e-5 0.07 20
./amazongoogle/run_pretraining_clean_roberta.sh 1024 3e-5 0.07 20
./amazongoogle/run_pretraining_clean_roberta.sh 1024 7e-5 0.07 20

./walmartamazon/run_pretraining_clean_roberta.sh 1024 1e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 3e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 1024 7e-5 0.07 20

./wdc-b/run_pretraining_clean_roberta.sh 1024 1e-5 0.07 20
./wdc-b/run_pretraining_clean_roberta.sh 1024 3e-5 0.07 20
./wdc-b/run_pretraining_clean_roberta.sh 1024 7e-5 0.07 20
