# Contrastive Pre-Training

This folder contains scripts to train contrastive models for blocking. 
We provide training procedures with the following loss functions:
- Supervised Contrastive Loss
- SimCLR
- Barlow Twins

The training scripts for each dataset can be found in the folder `src/contrastive`.

## Overview of Pre-training Scripts
|Dataset | Loss Function | Script |
|---|---|---|
|Abt-Buy|Supervised Contrastive Loss|`src/contrastive/abtbuy/run_pretraining_supervised_contrastive_clean_roberta.sh`|
|Abt-Buy|SimCLR|`src/contrastive/abtbuy/run_pretraining_simclr_clean_roberta.sh`|
|Abt-Buy|Barlow Twins|`src/contrastive/abtbuy/run_pretraining_barlow_twins_clean_roberta.sh`|
|Amazon-Google|Supervised Contrastive Loss|`src/contrastive/amazongoogle/run_pretraining_supervised_contrastive_clean_roberta.sh`|
|Amazon-Google|SimCLR|`src/contrastive/amazongoogle/run_pretraining_simclr_clean_roberta.sh`|
|Amazon-Google|Barlow Twins|`src/contrastive/amazongoogle/run_pretraining_barlow_twins_clean_roberta.sh`|
|Walmart-Amazon|Supervised Contrastive Loss|`src/contrastive/walmartamazon/run_pretraining_supervised_contrastive_clean_roberta.sh`|
|Walmart-Amazon|SimCLR|`src/contrastive/walmartamazon/run_pretraining_simclr_clean_roberta.sh`|
|Walmart-Amazon|Barlow Twins|`src/contrastive/walmartamazon/run_pretraining_barlow_twins_clean_roberta.sh`|
|WDC Block|Supervised Contrastive Loss|`src/contrastive/wdcproducts80cc20rnd050un/run_pretraining_supervised_contrastive_clean_roberta.sh`|
|WDC Block|SimCLR|`src/contrastive/wdcproducts80cc20rnd050un/run_pretraining_simclr_clean_roberta.sh`|
|WDC Block|Barlow Twins|`src/contrastive/wdcproducts80cc20rnd050un/run_pretraining_barlow_twins_clean_roberta.sh`|

For convenience, we provide for each dataset a script to run all pre-training scripts with the configuration used in the paper.
The scripts can be found in the respective folder for the dataset.
For instance, to run all pre-training scripts for the dataset Abt-Buy, you can use the script `src/finetuning/open_book/contrastive_pretraining/src/contrastive/abtbuy/run_all_pretrainings.sh`.
Please run the scripts from the root directory of the repository.

After training the models can be used to embed records and build a blocking index.
You can find the trained models in the folder `reports/contrastive`.

The code in this repository is based on the code of the paper "Supervised Contrastive Learning for Product Matching" by Ralph Peeters and Christian Bizer. The code can be found [here](https://github.com/wbsg-uni-mannheim/contrastive-product-matching).
