import os
from collections import defaultdict

import pandas as pd


def convert_synthetic_data_to_clustered_data(df_dataset, dataset_name):
    #df_dataset = pd.read_csv(path_to_ds, sep=';')

    #print(len(df_dataset))
    df_dataset = df_dataset[df_dataset['labels'] == 1]
    supcon_records = []
    record_id = 1
    unique_tuples = defaultdict(list)
    for index, row in df_dataset.iterrows():
        for identifier in ['left_tuples', 'right_tuples']:
            if row[identifier] not in unique_tuples[row['cluster_id']]:
                new_record = {'cluster_id': row['cluster_id'],
                              'row_id': record_id,
                              'source': row['source'],
                              'features': row[identifier]}
                record_id += 1
                supcon_records.append(new_record)
                unique_tuples[row['cluster_id']].append(row[identifier])


    path_to_fine_tuning_split = '{}data/{}/{}_fine-tuning_supcon_dl_block_augmentation.pkl.gz'.format(
        os.environ['DATA_DIR'],
        dataset_name, dataset_name)
    path_to_fine_tuning_split_csv = '{}data/{}/{}_fine-tuning_supcon_dl_block_augmentation.csv'.format(
        os.environ['DATA_DIR'],
        dataset_name, dataset_name)

    pd.DataFrame(supcon_records) \
        .sort_values(['cluster_id', 'row_id']) \
        .to_pickle(path_to_fine_tuning_split, compression='gzip')

    pd.DataFrame(supcon_records) \
        .sort_values(['cluster_id', 'row_id']) \
        .to_csv(path_to_fine_tuning_split_csv)

    print('Saved Supcon Data Set for {} in {}'.format(dataset_name, path_to_fine_tuning_split))



def convert_synthetic_data(path_to_ds, dataset_name):
    df_dataset = pd.read_csv(path_to_ds, sep=';')
    print(df_dataset.head(5))

    # Split DS by source
    identifier_to_features = defaultdict(str)
    pair_dict = defaultdict(list)

    identifier = 0
    for index, row in df_dataset.iterrows():
        identifier_1 = '{}-{}'.format(row['source'], str(identifier))
        identifier_to_features[identifier_1] = row['left_tuples']
        identifier += 1

        identifier_2 = '{}-{}'.format(row['source'], str(identifier))
        identifier_to_features[identifier_2] = row['right_tuples']
        identifier += 1

        pair_dict[identifier_1].append({'right_identifier': identifier_2, 'label': row['labels']})

    # Determine cluster - left to right
    clusters = []
    negative_pair_ids = set()
    for left_identifier in pair_dict:
        cluster = [left_identifier]
        for record in pair_dict[left_identifier]:
            right_identifier = record['right_identifier']
            if record['label'] == 1:
                cluster.append(right_identifier)
            else:
                negative_pair_ids.add(left_identifier)
                negative_pair_ids.add(right_identifier)

        clusters.append(cluster)

    # Merge cluster right to left
    changed = True
    while changed:
        new_clusters = []
        matches = []
        for i in range(0, len(clusters)):
            if i not in matches:
                new_cluster = set(clusters[i])
                for j in range(i + 1, len(clusters)):
                    if len(list(new_cluster & set(clusters[j]))) > 0:
                        matches.append(j)
                        new_cluster.update(clusters[j])
                new_clusters.append(list(new_cluster))

        changed = len(clusters) != len(new_clusters)
        clusters = list(new_clusters)

    # Assign ids to cluster
    id_2_cluster = {}
    cluster_id = 1
    supcon_records = []
    for cluster in clusters:
        record_id = 1
        for record in cluster:
            new_record ={'cluster_id': cluster_id,
                         'row_id': record_id,
                         'source': record.split(',')[0],
                         'features': identifier_to_features[record]}
            supcon_records.append(new_record)
            record_id += 1
            if record in negative_pair_ids:
                negative_pair_ids.remove(record)

        cluster_id += 1

    path_to_fine_tuning_split = '{}data/{}/{}_fine-tuning_supcon_dl_block_augmentation.pkl.gz'.format(os.environ['DATA_DIR'],
                                                                                            dataset_name, dataset_name)
    path_to_fine_tuning_split_csv = '{}data/{}/{}_fine-tuning_supcon_dl_block_augmentation.csv'.format(
        os.environ['DATA_DIR'],
        dataset_name, dataset_name)

    pd.DataFrame(supcon_records) \
        .sort_values(['cluster_id', 'row_id']) \
        .to_pickle(path_to_fine_tuning_split, compression='gzip')

    pd.DataFrame(supcon_records) \
        .sort_values(['cluster_id', 'row_id']) \
        .to_csv(path_to_fine_tuning_split_csv)

    print('Saved Supcon Data Set for {} in {}'.format(dataset_name, path_to_fine_tuning_split))


if __name__ == "__main__":
    path_to_ds = '../../../../../../DeepBlocker/synthetic_training_data_2022-08-01_13-30-44_abt-buy.csv'
    dataset_name = 'abt-buy' # MAKE SURE THAT dataset_name & path_to_ds MATCH!


    convert_synthetic_data_to_clustered_data(path_to_ds, dataset_name)