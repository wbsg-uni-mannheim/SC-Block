import logging

import pandas as pd


def convert_ids_of_datasets():
    # Load the dataframe
    df_tab_a = pd.read_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_block_s_train_l/tableA.csv')
    df_tab_b = pd.read_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_block_s_train_l/tableB.csv')

    # Create a new column with increasing numbers starting at one
    df_tab_a['new_id'] = range(0, len(df_tab_a))
    df_tab_b['new_id'] = range(0, len(df_tab_b))

    # Save the mapping between the old and the new id in a dictionary
    id_mapping_a = dict(zip(df_tab_a['id'], df_tab_a['new_id']))
    id_mapping_b = dict(zip(df_tab_b['id'], df_tab_b['new_id']))

    # Update the dataframe
    df_tab_a.drop(['id'], axis=1, inplace=True)
    df_tab_a = df_tab_a.rename(columns={'new_id': 'id'})

    df_tab_b.drop(['id'], axis=1, inplace=True)
    df_tab_b = df_tab_b.rename(columns={'new_id': 'id'})

    # Update mapping in train, validation & test
    # Update the ltable_id column using the id_mapping dictionary
    df_train = pd.read_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_block_s_train_l/train.csv')
    df_train['ltable_id'] = df_train['ltable_id'].map(id_mapping_a)
    df_train['rtable_id'] = df_train['rtable_id'].map(id_mapping_b)


    # Save the updated dataframe
    df_train.to_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_new_mapping/train.csv', index=False)

    df_valid = pd.read_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_block_s_train_l/valid.csv')
    df_valid['ltable_id'] = df_valid['ltable_id'].map(id_mapping_a)
    df_valid['rtable_id'] = df_valid['rtable_id'].map(id_mapping_b)

    # Save the updated dataframe
    df_valid.to_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_new_mapping/valid.csv', index=False)

    df_test = pd.read_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_block_s_train_l/test.csv')
    df_test['ltable_id'] = df_test['ltable_id'].map(id_mapping_a)
    df_test['rtable_id'] = df_test['rtable_id'].map(id_mapping_b)

    # Save the updated dataframe
    df_test.to_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_new_mapping/test.csv', index=False)

    # Save the updated dataframe
    cols = df_tab_a.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_tab_a = df_tab_a[cols]
    df_tab_a.to_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_new_mapping/tableA.csv', index=False)

    cols = df_tab_b.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_tab_b = df_tab_b[cols]
    df_tab_b.to_csv('/ceph/alebrink/deepblocker/data/wdcproducts80cc20rnd050un_new_mapping/tableB.csv', index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_ids_of_datasets()