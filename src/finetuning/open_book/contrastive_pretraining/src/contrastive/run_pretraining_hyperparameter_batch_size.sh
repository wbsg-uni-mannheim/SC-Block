#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

./abtbuy/run_pretraining_clean_roberta.sh 512 5e-5 0.07 20
./abtbuy/run_pretraining_clean_roberta.sh 256 5e-5 0.07 20
./abtbuy/run_pretraining_clean_roberta.sh 128 5e-5 0.07 20

./amazongoogle/run_pretraining_clean_roberta.sh 512 5e-5 0.07 20
./amazongoogle/run_pretraining_clean_roberta.sh 256 5e-5 0.07 20
./amazongoogle/run_pretraining_clean_roberta.sh 128 5e-5 0.07 20

./walmartamazon/run_pretraining_clean_roberta.sh 512 5e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 256 5e-5 0.07 20
./walmartamazon/run_pretraining_clean_roberta.sh 128 5e-5 0.07 20

./wdc-b/run_pretraining_clean_roberta.sh 512 5e-5 0.07 20
./wdc-b/run_pretraining_clean_roberta.sh 256 5e-5 0.07 20
./wdc-b/run_pretraining_clean_roberta.sh 128 5e-5 0.07 20
