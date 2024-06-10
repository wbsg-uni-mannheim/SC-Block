import logging
import os
import random

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option('--dataset')
@click.option('--testset')
@click.option('--size', default='large')
@click.option('--modify_data', default=False)
def convert_table_to_query_table(dataset, testset, size, modify_data):

    random.seed(42)

    splits = ['train', 'valid', 'test']
    ds_sets = {}
    for split in splits:
        if split == 'test':
            path = '{}/wdc_product_entity_matching_benchmark/80pair/{}{}_{}.json.gz'.format(os.environ['DATA_DIR'], dataset, testset, split)
        else:
            path = '{}/wdc_product_entity_matching_benchmark/80pair/{}000un_{}_{}.json.gz'.format(os.environ['DATA_DIR'], dataset,
                                                                                  split, size)

        print(path)
        df_split = pd.read_json(path, lines=True)
        ds_sets[split] = df_split.sort_values(by=['label'], ascending=False)

    if modify_data:
        # Determine unique cluster ids - all clusters appear left and right
        cluster_ids = [ds_sets[split]['cluster_id_left'].unique() for split in ds_sets]
        cluster_ids = set([cluster_id for cluster_id_list in cluster_ids for cluster_id in cluster_id_list])
        not_test_cluster_ids = [ds_sets[split]['cluster_id_left'].unique() for split in ds_sets if split in ['train', 'valid']]
        not_test_cluster_ids = set([cluster_id for cluster_id_list in not_test_cluster_ids for cluster_id in cluster_id_list])

        # Make sure that seen records are really seen during training! --> One pair from the cluster
        cluster_ids_to_record_ids = {}

        unique_ids = set()
        for split in ds_sets:
            print(len(ds_sets[split]))
            unique_ids.update(ds_sets[split]['id_left'].unique())

        print(len(unique_ids))
        print(ds_sets['test'].columns)

        swap_columns = ['id', 'brand', 'title', 'description', 'price', 'priceCurrency', 'cluster_id']

        # Determine left offer ids per cluster id
        for cluster_id in tqdm(cluster_ids):
            # Get Offer ids for train & valid
            offer_ids = [ds_sets[split].loc[(ds_sets[split]['cluster_id_left'] == cluster_id)]['id_left'].unique()
                          for split in ds_sets]
            offer_ids = list(set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list]))
            left_offer_id = offer_ids.pop()  # Leading offer id

            # train_valid_offer_ids = [ds_sets[split].loc[(ds_sets[split]['cluster_id_left'] == cluster_id)]['id_left'].unique()
            #              for split in ['train', 'valid']]
            # train_valid_offer_ids = list(set([offer_id for offer_id_list in train_valid_offer_ids
            #                                   for offer_id in offer_id_list]))
            # test_offer_ids = list(ds_sets['test'].loc[(ds_sets['test']['cluster_id_left'] == cluster_id)]['id_left'].unique())
            #
            # if len(train_valid_offer_ids) > 0:
            #     # Determine lead offer id from train/valid
            #     train_valid_offer_ids.sort()
            #     left_offer_id = train_valid_offer_ids.pop() # Leading offer id
            # else:
            #     # Determine lead offer id from test if the cluster is only present in the test set
            #     test_offer_ids.sort()
            #     left_offer_id = test_offer_ids.pop()
            #
            # offer_ids = set(train_valid_offer_ids + test_offer_ids)

            #cluster_ids_to_record_ids[cluster_id] = {'left_offer_id': left_offer_id, 'offer_ids': list(offer_ids)}

            for split in ds_sets:
                # # Swap leading offer ids if they appear on the right hand site of a pair
                # swap_rows = ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id]
                # for column in swap_columns:
                #     swap_rows['{}_temp'.format(column)] = swap_rows['{}_right'.format(column)]
                #     swap_rows['{}_right'.format(column)] = swap_rows['{}_left'.format(column)]
                #     swap_rows['{}_left'.format(column)] = swap_rows['{}_temp'.format(column)]

                # Replace left hand side offer ids by leading offer id
                for offer_id in offer_ids:
                    ds_sets[split].loc[ds_sets[split]['id_left'] == offer_id, 'id_left'] = left_offer_id

                # Replace right hand side leading offer ids with random offer id from the same cluster
                ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(list(offer_ids))
                # if left_offer_id in ds_sets[split]['id_right']:
                #     if split == 'test':
                #         ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(test_offer_ids)
                #     else:
                #         ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id, 'id_right'] = random.choice(
                #             train_valid_offer_ids)
                #ds_sets[split] = ds_sets[split].loc[ds_sets[split]['id_right'] == left_offer_id]

                ds_sets[split] = ds_sets[split].drop_duplicates()

                # Delete all other offer ids from the left hand sight
                # swap_rows = ds_sets[split].loc[ds_sets[split]['id_left'].isin(offer_ids)]
                # for column in swap_columns:
                #     swap_rows['{}_temp'.format(column)] = swap_rows['{}_right'.format(column)]
                #     swap_rows['{}_right'.format(column)] = swap_rows['{}_left'.format(column)]
                #     swap_rows['{}_left'.format(column)] = swap_rows['{}_temp'.format(column)]
                #ds_sets[split] = ds_sets[split].loc[~ds_sets[split]['id_left'].isin(offer_ids)]
                #ds_sets[split] = pd.concat([ds_sets[split], swap_rows])

        # Remove pairs where both records are supposed to be in the query table
        unique_left_ids = set()
        for split in ds_sets:
            #print(len(ds_sets[split]))
            unique_left_ids.update(ds_sets[split]['id_left'].unique())

        for split in ds_sets:
            ds_sets[split] = ds_sets[split].loc[~ds_sets[split]['id_right'].isin(unique_left_ids)]


        #unique_ids = set()
        for cluster_id in tqdm(cluster_ids):
            offer_ids = [ds_sets[split].loc[(ds_sets[split]['cluster_id_left'] == cluster_id) & (
                        ds_sets[split]['cluster_id_right'] == cluster_id)]['id_left'].unique() for split in ds_sets]
            offer_ids = set([offer_id for offer_id_list in offer_ids for offer_id in offer_id_list])

        for split in splits:
            print(len(ds_sets[split]))
            print(len(ds_sets[split].loc[ds_sets[split]['label'] == 1]))

    table_A_records = []
    table_B_records = []
    split_info = {'train': [], 'valid': [], 'test': []}
    print(ds_sets['test'].columns)
    for split in splits:
        for index, row in ds_sets[split].iterrows():
            left_record = {'id': row['id_left'], 'brand': row['brand_left'], 'title': row['title_left'],
                           'description': row['description_left'], 'price': row['price_left'],
                           'pricecurrency': row['priceCurrency_left'], 'specTableContent': row['specTableContent_left'],
                           'cluster_id': row['cluster_id_left']}
            table_A_records.append(left_record)

            right_record = {'id': row['id_right'],'brand': row['brand_right'], 'title': row['title_right'],
                           'description': row['description_right'],'price': row['price_right'],
                           'pricecurrency': row['priceCurrency_right'], 'spectablecontent': row['specTableContent_right'],
                           'cluster_id': row['cluster_id_right']}
            table_B_records.append(right_record)

            matching_info = {'ltable_id': row['id_left'], 'rtable_id': row['id_right'], 'label': row['label'], 'cluster_id': row['cluster_id_left']}
            split_info[split].append(matching_info)

    # Create Data Frames
    df_table_a = pd.DataFrame(table_A_records)
    df_table_b = pd.DataFrame(table_B_records)

    # #Add matches to train/ validation to make sure that each cluster is seen at least once.
    # for cluster_id in not_test_cluster_ids:
    #     cluster_id_to_record_ids = cluster_ids_to_record_ids[cluster_id]
    #     for _ in range(2):
    #         if len(offer_ids) > 0:
    #             matching_info = {'ltable_id': cluster_id_to_record_ids['left_offer_id'],
    #                              'rtable_id': cluster_id_to_record_ids['offer_ids'].pop(), 'label': 1, 'cluster_id': cluster_id}
    #             if matching_info not in split_info['train'] and matching_info not in split_info['valid'] \
    #                 and matching_info not in split_info['test']:
    #                 if random.randint(1, 4) == 1:
    #                     split_info['valid'].append(matching_info)
    #                 else:
    #                     split_info['train'].append(matching_info)

    df_train = pd.DataFrame(split_info['train']).drop(columns=['cluster_id'])
    df_valid = pd.DataFrame(split_info['valid']).drop(columns=['cluster_id'])
    df_test = pd.DataFrame(split_info['test']).drop(columns=['cluster_id'])

    # Drop duplicates from data tables
    df_table_a = df_table_a.drop_duplicates(subset=['id'])
    df_table_b = df_table_b.drop_duplicates(subset=['id'])

    # Save Data Frames
    path = '{}/deepmatcher/wdcproducts80cc20rnd{}'.format(os.environ['DATA_DIR'], testset)

    if not os.path.exists(path):
        os.makedirs(path)

    df_table_a = df_table_a.set_index('id')
    df_table_a.to_csv(path_or_buf='{}/tableA.csv'.format(path), sep=',')

    df_table_b = df_table_b.set_index('id')
    df_table_b.to_csv(path_or_buf='{}/tableB_small.csv'.format(path), sep=',')

    df_train.to_csv(path_or_buf='{}/train.csv'.format(path), sep=',', index=False)
    df_valid.to_csv(path_or_buf='{}/valid.csv'.format(path), sep=',', index=False)
    df_test.to_csv(path_or_buf='{}/test.csv'.format(path), sep=',', index=False)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_query_table()