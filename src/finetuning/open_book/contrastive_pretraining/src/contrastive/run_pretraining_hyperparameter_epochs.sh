#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 5
./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 10
./abtbuy/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 30

./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 5
./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 10
./amazongoogle/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 30

./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 5
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 10
./walmartamazon/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 30

./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 5
./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 10
./wdc-b/run_pretraining_clean_roberta.sh 1024 5e-5 0.07 30
