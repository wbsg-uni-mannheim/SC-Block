import gzip
import json
import os

import pandas as pd
import pickle

from tqdm import tqdm


def convert_table_corpus_to_table_b(path_to_table_corpus):

    final_dict = {'row_id': [], 'brand': [], 'name': [],  'description': [], 'price': [], 'pricecurrency': [], 'spectablecontent': [], 'cluster_id': [], }
    for file in tqdm(os.listdir(path_to_table_corpus)):
        filename = os.fsdecode(file)
        if 'tableB' in filename:
            print('Found table B!')
            continue

        file_path = '{}/{}'.format(path_to_table_corpus, filename)
        with gzip.open(file_path, 'rt') as f:
            for line in f:
                json_dict = json.loads(line)
                for key in final_dict:
                    if key in json_dict:
                        final_dict[key].append(json_dict[key])
                    else:
                        final_dict[key].append(None)

    df = pd.DataFrame(final_dict)
    df = df.rename(columns={"row_id": "id", "name": "title"})
    df['description'] = df['description'].str[:200]
    print(df.columns)
    df = df[['id', 'brand', 'title', 'description', 'price', 'pricecurrency', 'spectablecontent', 'cluster_id']]

    return df

# Load the pickled pandas DataFrame
df_train = pd.read_pickle("/home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/processed/wdc-b/contrastive/wdc-b-train.pkl.gz")

df_additional_data = convert_table_corpus_to_table_b('/ceph/alebrink/tableAugmentation/data/corpus/wdc-b')

df_additional_data['id'] = 'tableb_' + df_additional_data['id'].astype(str)

df_train = pd.concat([df_train, df_additional_data])

df_train.to_pickle("/home/alebrink/development/table-augmentation-framework/src/finetuning/open_book/contrastive_product_matching/data/processed/wdc-b/contrastive/wdc-b-additionaldata-train.pkl.gz")
