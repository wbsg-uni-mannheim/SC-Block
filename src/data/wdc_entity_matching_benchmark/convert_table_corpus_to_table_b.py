import gzip
import json
import logging
import os

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option('--path_to_table_corpus', default=None)
def convert_table_corpus_to_table_b(path_to_table_corpus):

    final_dict = {'row_id': [], 'brand': [], 'name': [],  'description': [], 'price': [], 'pricecurrency': [], 'spectablecontent': [], 'cluster_id': [], }
    for file in tqdm(os.listdir(path_to_table_corpus)):
        filename = os.fsdecode(file)
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
    output_path = '/ceph/alebrink/tableAugmentation/data/deepmatcher/wdcproducts80cc20rnd050un/tableB.csv'
    df.to_csv(path_or_buf=output_path, sep=',', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_corpus_to_table_b()