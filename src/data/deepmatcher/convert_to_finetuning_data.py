import itertools
import logging
import os
import random
import re
from collections import defaultdict

import click
import pandas as pd

#from src.finetuning.open_book.contrastive.data.datasets import Augmenter
from tqdm import tqdm

from src.strategy.open_book.entity_serialization import EntitySerializer


@click.command()
@click.option('--schema_org_class')
def convert_to_finetuning_data(schema_org_class):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param schema_org_class string org class represents the dataset name"""

    random.seed(42)

    path_to_table_a = '{}/deepmatcher/{}/tableA.csv'.format(
        os.environ['DATA_DIR'], schema_org_class)
    path_to_table_b = '{}/deepmatcher/{}/tableB_small.csv'.format(
        os.environ['DATA_DIR'], schema_org_class)
    df_table_a = pd.read_csv(path_to_table_a, index_col='id', encoding='utf-8')
    df_table_b = pd.read_csv(path_to_table_b, index_col='id', encoding='utf-8')

    entity_serializer = EntitySerializer(schema_org_class)

    splits = ['train', 'valid', 'test']
    split_dfs = {}
    for split in splits:

        # Load split
        path_to_split = '{}/deepmatcher/{}/{}.csv'.format(os.environ['DATA_DIR'], schema_org_class, split)
        df_split = pd.read_csv(path_to_split, encoding='utf-8')

        # Prepare data
        df_split = df_split.apply(apply_serialization, args=(df_table_a, df_table_b, entity_serializer), axis=1)
        df_split = df_split[['features_left', 'features_right', 'label']].rename(columns={"label": "labels"})

        path_to_fine_tuning_split = '{}/finetuning/open_book/{}/{}_finetuning_{}'.format(
            os.environ['DATA_DIR'],
            schema_org_class, schema_org_class, split)

        # Save split
        df_split.to_csv('{}.csv'.format(path_to_fine_tuning_split), index=None, encoding='utf-8')
        df_split.to_pickle('{}.pkl.gz'.format(path_to_fine_tuning_split), compression='gzip')
        logging.info('Converted and Saved {} split for data set {}'.format(split, schema_org_class))

        # Collect splits
        split_dfs[split] = df_split

    # Merge train + validation
    df_train_val = pd.concat([split_dfs['train'], split_dfs['valid']])
    path_to_fine_tuning_split = '{}/finetuning/open_book/{}/{}_finetuning_train_valid'.format(os.environ['DATA_DIR'],
                                                                                              schema_org_class,
                                                                                              schema_org_class)

    df_train_val.to_csv('{}.csv'.format(path_to_fine_tuning_split), index=None, encoding='utf-8')
    df_train_val.to_pickle('{}.pkl.gz'.format(path_to_fine_tuning_split), compression='gzip')
    logging.info('Saved train + valid')


def apply_serialization(row, df_table_a, df_table_b, entity_serializer):
    row['features_left'] = entity_serializer.convert_to_str_representation(df_table_a.iloc[row['ltable_id']].to_dict())
    row['features_right'] = entity_serializer.convert_to_str_representation(df_table_b.iloc[row['rtable_id']].to_dict())
    return row


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_to_finetuning_data()
    