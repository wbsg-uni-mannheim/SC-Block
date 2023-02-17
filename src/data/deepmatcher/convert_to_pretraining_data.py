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
@click.option('--augmentation', type=int)
def convert_to_pretraining_data(schema_org_class, augmentation):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param schema_org_class string org class represents the dataset name"""

    random.seed(42)

    path_to_table_a = '{}/deepmatcher/{}/tableA.csv'.format(
        os.environ['DATA_DIR'], schema_org_class)
    path_to_table_b = '{}/deepmatcher/{}/tableB_small.csv'.format(
        os.environ['DATA_DIR'], schema_org_class)
    path_to_train_set = '{}/deepmatcher/{}/train.csv'.format(os.environ['DATA_DIR'], schema_org_class)
    path_to_valid_set = '{}/deepmatcher/{}/valid.csv'.format(os.environ['DATA_DIR'], schema_org_class)
    #path_to_test_set = '{}/deepmatcher/{}/test.csv'.format(os.environ['DATA_DIR'], schema_org_class)

    for percentage in [0, 5, 80, 100]:
    #for percentage in [100]:
        # Read Train Set
        print('Prepare DS with {} percentage of train data'.format(percentage))
        pair_dict = defaultdict(list)
        pair_list = []
        if percentage > 0:
            add_pairs_to_pair_dict([path_to_train_set, path_to_valid_set], pair_dict, pair_list, percentage)
        #add_pairs_to_pair_dict(path_to_valid_set, pair_dict)

        # Determine cluster - left to right
        clusters = []
        negative_pair_ids = set()
        for left_row_id in pair_dict:
            left_identifier = 'table_a-{}'.format(left_row_id)
            cluster = [left_identifier]
            for record in pair_dict[left_row_id]:
                right_identifier = 'table_b-{}'.format(record['row_id'])
                if record['label'] == '1':
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
                    for j in range(i+1, len(clusters)):
                        if len(list(new_cluster & set(clusters[j]))) > 0:
                            matches.append(j)
                            new_cluster.update(clusters[j])
                    new_clusters.append(list(new_cluster))

            changed = len(clusters) != len(new_clusters)
            clusters = list(new_clusters)

        # Merge clusters with matching serializations


        #Assign ids to cluster
        id_2_cluster = {}
        cluster_id = 1
        for cluster in clusters:
            record_id = 1
            for record in cluster:
                id_2_cluster[record] = {'cluster_id': cluster_id, 'record_id': record_id}
                record_id += 1
                if record in negative_pair_ids:
                    negative_pair_ids.remove(record)

            cluster_id += 1

        entity_serializer = EntitySerializer(schema_org_class)

        # Build supcon training data
        supcon_records = []
        new_supcon_record_a, cluster_id = add_supcon_records(path_to_table_a, 'table_a', id_2_cluster, entity_serializer,
                                                 cluster_id, negative_pair_ids, schema_org_class, augmentation)
        new_supcon_record_b, cluster_id = add_supcon_records(path_to_table_b, 'table_b', id_2_cluster, entity_serializer,
                                                 cluster_id, negative_pair_ids, schema_org_class, augmentation)

        supcon_records.extend(new_supcon_record_a)
        supcon_records.extend(new_supcon_record_b)

        # # Merge based on token containment
        # for i in tqdm(range(0, len(supcon_records))):
        #     supcon_records[i]['preprocessed'] = re.sub('[^0-9a-zA-Z]+', ' ', supcon_records[i]['features'].lower())
        #
        # clusters = []
        # matches = []
        # for i in tqdm(range(0, len(supcon_records))):
        #     if i not in matches:
        #         new_cluster = {supcon_records[i]['cluster_id']}
        #         entity_1_token_set = set(supcon_records[i]['preprocessed'].split(' '))
        #         for j in range(i+1, len(supcon_records)):
        #             entity_2_token_set = set(supcon_records[j]['preprocessed'].split(' '))
        #             if len(entity_1_token_set.intersection(entity_2_token_set)) == min(len(entity_1_token_set), len(entity_2_token_set)):
        #                 matches.append(j)
        #                 new_cluster.add(supcon_records[j]['cluster_id'])
        #         clusters.append(list(new_cluster))


        # Merge clusters with the same serialization
        serialization_to_cluster = defaultdict(list)
        for record in supcon_records:
            serialization_to_cluster[record['features']].append(record['cluster_id'])

        clusters = list(serialization_to_cluster.values())
        changed = True
        while changed:
            new_clusters = []
            matches = []
            for i in range(0, len(clusters)):
                if i not in matches:
                    new_cluster = set(clusters[i])
                    for j in range(i+1, len(clusters)):
                        if len(list(new_cluster & set(clusters[j]))) > 0:
                            matches.append(j)
                            new_cluster.update(clusters[j])
                    new_clusters.append(list(new_cluster))

            changed = len(clusters) != len(new_clusters)
            clusters = list(new_clusters)


        original_2_merged_cluster_id = defaultdict(int)
        for cluster_of_clusters in clusters:
            min_cluster_id = min(cluster_of_clusters)
            for cluster_id in cluster_of_clusters:
                original_2_merged_cluster_id[cluster_id] = min_cluster_id

        if not os.path.exists('{}/finetuning/open_book/{}/'.format(os.environ['DATA_DIR'], schema_org_class)):
            os.makedirs('{}/finetuning/open_book/{}/'.format(os.environ['DATA_DIR'], schema_org_class))

        path_to_fine_tuning_split = '{}/finetuning/open_book/{}/{}_fine-tuning_supcon_{}_all_pairs_{}_dlblock_augmentation_preprocessed_surface_form_deduplication.pkl.gz'.format(os.environ['DATA_DIR'],
                                                                                            schema_org_class, schema_org_class, percentage, augmentation)
        path_to_fine_tuning_split_csv = '{}/finetuning/open_book/{}/{}_fine-tuning_supcon_{}_all_pairs_{}_dlblock_augmentation_preprocessed_surface_form_deduplication.csv'.format(os.environ['DATA_DIR'],
                                                                                            schema_org_class, schema_org_class, percentage, augmentation)

        df_subcon = pd.DataFrame(supcon_records)\
            .sort_values(['cluster_id', 'id'])

        #df_subcon['cluster_id'] = df_subcon['cluster_id'].map(original_2_merged_cluster_id)

        #df_subcon = df_subcon.drop_duplicates(subset=['cluster_id', 'features', 'source'])

        df_subcon.to_pickle(path_to_fine_tuning_split, compression='gzip')

        df_subcon.to_csv(path_to_fine_tuning_split_csv)
        logging.info('Saved Supcon Data Set for {} in {}'.format(schema_org_class, path_to_fine_tuning_split))
        #
        # # Build SBERT Training Data
        # sbert_records = add_sbert_records(path_to_table_a, path_to_table_b, pair_list, entity_serializer, schema_org_class)
        #
        # path_to_fine_tuning_split_csv = '{}/finetuning/open_book/{}/{}_fine-tuning_sbert_{}_subset_pairs.csv'.format(
        #     os.environ['DATA_DIR'],
        #     schema_org_class, schema_org_class, percentage)
        #
        # pd.DataFrame(sbert_records).to_csv(path_to_fine_tuning_split_csv, sep=';')
        # logging.info('Saved SBERT Data Set for {} in {}'.format(schema_org_class, path_to_fine_tuning_split_csv))


def add_pairs_to_pair_dict(files, pair_dict, pair_list, percentage):
    lines = []
    for file in files:
        new_lines = []
        with open(file, 'r') as f:
            new_lines.extend(f.readlines())
        lines.extend(random.sample(new_lines, int(len(new_lines) * (percentage / 100))))

    for line in lines:
        line = line.replace('\n', '')
        line_values = line.split(',')
        if line_values[0] == 'ltable_id':
            # Skip first line
            continue
        pair_dict[line_values[0]].append({'row_id': line_values[1], 'label': line_values[2]})
        pair_list.append(tuple([int(value) for value in line_values]))


def add_supcon_records(path_to_table, table_name, id_2_cluster, entity_serializer, new_cluster_id, negative_pair_ids, schema_org_class, augmentation):
    new_records = []

    #augmenter = Augmenter('all', schema_org_class)
    df_table = pd.read_csv(path_to_table)

    for index, row in df_table.iterrows():
        entity = row.to_dict()
        entity = dict((k.lower(), v) for k, v in entity.items())

        entity['row_id'] = entity['id']
        del entity['id']

        if schema_org_class in ['amazon-google', 'dblp-acm_1', 'dblp-acm_2',
                                'dblp-googlescholar_1', 'dblp-googlescholar_2',
                                'walmart-amazon_1', 'walmart-amazon_2']:
            entity['name'] = entity['title']
            del entity['title']
        elif schema_org_class in ['itunes-amazon_1', 'itunes-amazon_2']:
            entity['name'] = entity['song_name']
            del entity['song_name']

        entity_identifier = '{}-{}'.format(table_name, entity['row_id'])
        if entity_identifier in id_2_cluster:
            new_record = {'id': id_2_cluster[entity_identifier]['record_id'],
                          'cluster_id': id_2_cluster[entity_identifier]['cluster_id'],
                          'source': table_name,
                          'row_id': entity['row_id'],
                          'features': entity_serializer.convert_to_str_representation(entity)}
            new_records.append(new_record)

            for i in range(0,augmentation):
                entity_copy = entity.copy()
                for attr in entity_copy:
                    if len(str(entity_copy[attr])) > 0:
                        try:
                            entity_copy[attr] = augment_like_dlblock(str(entity_copy[attr]))
                        except:
                            logging.getLogger().info('Not able to augmentation: {}'.format(entity_copy[attr]))

                new_record = {'id': id_2_cluster[entity_identifier]['record_id'] * 10,
                              'cluster_id': id_2_cluster[entity_identifier]['cluster_id'],
                              'source': table_name,
                              'row_id': entity['row_id'],
                              'features': entity_serializer.convert_to_str_representation(entity_copy)}
                new_records.append(new_record)
        else:
            # Add all records to training data
            new_record = {'id': 1,
                          'cluster_id': new_cluster_id,
                          'source': table_name,
                          'row_id': entity['row_id'],
                          'features': entity_serializer.convert_to_str_representation(entity)}
            new_records.append(new_record)

            for i in range(0, augmentation):
                entity_copy = entity.copy()
                for attr in entity_copy:
                    if len(str(entity_copy[attr])) > 0:
                        try:
                            entity_copy[attr] = augment_like_dlblock(str(entity_copy[attr]))
                        except:
                            logging.getLogger().info('Not able to augmentation: {}'.format(entity_copy[attr]))
                new_record = {'id': i + 1,
                              'cluster_id': new_cluster_id,
                              'source': table_name,
                              'row_id': entity['row_id'],
                              'features': entity_serializer.convert_to_str_representation(entity_copy)}
                new_records.append(new_record)

            new_cluster_id += 1
    return new_records, new_cluster_id


def add_sbert_records(path_to_table_a, path_to_table_b, pair_list, entity_serializer, schema_org_class):
    new_records = []

    df_table_a = pd.read_csv(path_to_table_a)
    df_table_b = pd.read_csv(path_to_table_b)

    for pair in pair_list:
        record_a = df_table_a.loc[df_table_a['id'] == pair[0]].iloc[0]
        record_b = df_table_b.loc[df_table_b['id'] == pair[1]].iloc[0]

        entities = []
        for record in [record_a, record_b]:
            entity = record.to_dict()
            entity = dict((k.lower(), v) for k, v in entity.items())

            entity['row_id'] = entity['id']
            del entity['id']

            if schema_org_class in ['amazon-google', 'dblp-acm_1', 'dblp-acm_2',
                                    'dblp-googlescholar_1', 'dblp-googlescholar_2',
                                    'walmart-amazon_1', 'walmart-amazon_2']:
                entity['name'] = entity['title']
                del entity['title']
            elif schema_org_class in ['itunes-amazon_1', 'itunes-amazon_2']:
                entity['name'] = entity['song_name']
                del entity['song_name']

            entities.append(entity_serializer.convert_to_str_representation(entity))

        record = {'cluster': 0, 'score': pair[2], 'split': 'train', 'entity1': entities[0], 'entity2': entities[1]}
        new_records.append(record)

    return new_records


def augment_like_dlblock(string_representation, max_perturbation=0.4):
    """Method is replicated from DL-Block Paper - generate_synthetic_training_data
        To-Do: Pass parameter through stack"""
    string_representation_values = str(string_representation).split(' ')
    max_tokens_to_remove = int(len(string_representation_values) * max_perturbation)

    string_representation_values_copy = string_representation_values[:]
    num_tokens_to_remove = random.randint(0, max_tokens_to_remove)
    for _ in range(num_tokens_to_remove):
        # randint is inclusive. so randint(0, 5) can return 5 also
        string_representation_values_copy.pop(random.randint(0, len(string_representation_values_copy) - 1))

    return ' '.join(string_representation_values_copy)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_to_pretraining_data()
