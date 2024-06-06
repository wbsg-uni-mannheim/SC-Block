import logging
import os
from collections import defaultdict

import click
import pandas as pd

from src.model.evidence import RetrievalEvidence
from src.model.querytable import RetrievalQueryTable


@click.command()
@click.option('--dataset')
@click.option('--table_name', default='tableA')
def convert_table_to_query_table(dataset, table_name):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param dataset string org class represents the dataset name"""

    switched = False if table_name == 'tableA' else True  # Switched table A and B
    if switched:
        logging.info('Switched table A and B')

    path_to_table_a = '{}/deepmatcher/{}/{}.csv'.format(
        os.environ['DATA_DIR'], dataset, table_name)
    path_to_test_set = '{}/deepmatcher/{}/test.csv'.format(os.environ['DATA_DIR'], dataset)
    path_to_train_set = '{}/deepmatcher/{}/train.csv'.format(os.environ['DATA_DIR'], dataset)
    path_to_valid_set = '{}/deepmatcher/{}/valid.csv'.format(os.environ['DATA_DIR'], dataset)

    # Add all records as evidences to query table
    test_record_dict = defaultdict(list)
    for split in ['train', 'valid', 'test']:
        path = '{}/deepmatcher/{}/{}.csv'.format(os.environ['DATA_DIR'], dataset, split)
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_values = line.split(',')
                if line_values[0] == 'ltable_id':
                    # Skip first line
                    continue

                if table_name == 'tableA':
                    test_record_dict[line_values[0]].append({'row_id': line_values[1], 'label': line_values[2], 'split': split})
                if table_name == 'tableB':
                    test_record_dict[line_values[1]].append({'row_id': line_values[0], 'label': line_values[2], 'split': split})

    # Extract seen records
    seen_evidences_records = set()
    seen_entity_records = set()
    #seen_pair = []
    for path in [path_to_train_set, path_to_valid_set]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_values = line.split(',')
                if line_values[0] == 'ltable_id':
                    continue
                if line_values[2] == '0':
                    if table_name == 'tableA':
                        # Take seen record from table B
                        #seen_pair.append(f'{line_values[0]}#{line_values[1]}')
                        seen_evidences_records.add(line_values[1])
                        seen_entity_records.add(line_values[0])
                    elif table_name == 'tableB':
                        # Take seen evidence record from table A
                        #seen_pair.append(f'{line_values[1]}#{line_values[0]}')
                        seen_evidences_records.add(line_values[0])
                        seen_entity_records.add(line_values[1])



    # Build Query Table
    query_table_ids = {'abt-buy': 10000,
                       'amazon-google': 11000,
                       'dblp-acm_1': 12000,
                       'dblp-acm_2': 13000,
                       'dblp-googlescholar_1': 14000,
                       'dblp-googlescholar_2': 15000,
                       'itunes-amazon_1': 16000,
                       'itunes-amazon_2': 17000,
                       'walmart-amazon_1': 18000,
                       'walmart-amazon_2': 19000,
                       'wdcproducts80cc20rnd050un': 20000,
                       'wdcproducts80cc20rnd000un': 21000,
                       'wdcproducts80cc20rnd100un': 22000,
                       'wdcproducts80cc20rnd050un_block_s_train_l': 23000,
                       'wdcproducts80cc20rnd050un_block_m_train_l': 24000,
                       'wdcproducts80cc20rnd050un_block_l_train_l': 25000}

    qt_id = query_table_ids[dataset]
    assembling_strategy = 'Test {} of data set {}'.format(table_name, dataset)
    gt_table = dataset


    verified_evidences = []
    table = []
    evidence_id = 1

    df_table = pd.read_csv(path_to_table_a).fillna('')
    context_attributes = df_table.columns.to_list()[1:]
    if 'wdcproducts' in dataset:
        context_attributes = list(map(lambda x: x.replace('title', 'name'), context_attributes))

    seen_counter = {'both_seen': 0, 'left_seen': 0, 'right_seen': 0, 'none_seen': 0}

    for index, row in df_table.iterrows():
        if len(table) >= 100:
            # Check data sets into query tables with 100 records
            query_table = RetrievalQueryTable(qt_id, 'retrieval', assembling_strategy,
                                              gt_table, dataset,
                                              context_attributes,  # Exclude id
                                              table, verified_evidences)
            query_table.switched = switched
            query_table.save(with_evidence_context=False)

            # Initialize variables for new query table
            verified_evidences = []
            table = []
            evidence_id = 1
            qt_id += 1

        entity = row.to_dict()
        entity = dict((k.lower(), v) for k, v in entity.items())

        entity['entityId'] = entity['id']
        del entity['id']

        if dataset in ['amazon-google', 'dblp-acm_1', 'dblp-acm_2', 'dblp-googlescholar_1',
                                'dblp-googlescholar_2', 'walmart-amazon_1', 'walmart-amazon_2']:
            entity['name'] = entity['title']
            del entity['title']
        elif 'wdcproducts' in dataset:
            entity['name'] = entity['title']
            del entity['title']
            del entity['cluster_id']
            context_attributes = list(map(lambda x: x.replace('Pant', 'Ishan'), context_attributes))
        elif dataset in ['itunes-amazon_1', 'itunes-amazon_2']:
            entity['name'] = entity['song_name']
            del entity['song_name']

        #added_pairs = []
        for reference in test_record_dict[str(entity['entityId'])]:
            #pair_id = '{}#{}'.format(entity['entityId'], reference['row_id'])
            #if pair_id not in added_pairs:
            table_identifier = '{}_{}.json.gz'.format(dataset, 'tableA' if table_name == 'tableB' else 'tableB').lower()
            evidence = RetrievalEvidence(evidence_id, qt_id, entity['entityId'],
                                         table_identifier, reference['row_id'], None, reference['split'])
            evidence.scale = int(reference['label'])
            evidence.signal = int(reference['label']) == 1

            if str(entity['entityId']) in seen_entity_records and reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['both_seen'] += 1
            elif str(entity['entityId']) in seen_entity_records:
                evidence.seen_training = 'left_seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['left_seen'] += 1
            elif reference['row_id'] in seen_evidences_records:
                evidence.seen_training = 'right_seen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['right_seen'] += 1
            else:
                evidence.seen_training = 'unseen'
                if int(reference['label']) == 1 and reference['split'] == 'test':
                    seen_counter['none_seen'] += 1

            # = str(entity['entityId']) in seen_entity_records or reference['row_id'] in seen_evidences_records

            #reference['row_id'] in seen_evidences_records or entity['entityId'] in seen_entity_records

            verified_evidences.append(evidence)
                #added_pairs.append(pair_id)
            evidence_id += 1

        #if len(test_record_dict[str(entity['entityId'])]) > 0:
        # Add all entities of table a to query table
        table.append(entity)

    # Save final query table
    query_table = RetrievalQueryTable(qt_id, 'retrieval', assembling_strategy,
                                      gt_table, dataset,
                                      context_attributes,  # Exclude id
                                      table, verified_evidences)
    query_table.switched = switched
    query_table.save(with_evidence_context=False)
    logging.info('Converted {} of {} to query table'.format(table_name, dataset))

    print(seen_counter)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_query_table()
