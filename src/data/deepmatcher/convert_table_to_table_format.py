import gzip
import json
import logging
import os
from collections import defaultdict

import click
import pandas as pd

from src.model.evidence import RetrievalEvidence
from src.model.querytable import RetrievalQueryTable


@click.command()
@click.option('--dataset')
@click.option('--table_name', default='tableB')
@click.option('--path_to_additional_records', default=None)
@click.option('--index_size', default=200000)
def convert_table_to_table_format(dataset, table_name, path_to_additional_records, index_size):
    """ Convert Test set Table A of deepmatcher benchmark to query table
    :param dataset string org class represents the dataset name"""

    path_to_table_b = '{}/deepmatcher/{}/{}.csv'.format(
        os.environ['DATA_DIR'], dataset, table_name)
    path_to_corpus_table = '{}/corpus/{}'.format(
        os.environ['DATA_DIR'], dataset.lower())

    formatted_lines = []

    df_table_b = pd.read_csv(path_to_table_b).fillna('')

    for index, row in df_table_b.iterrows():
        #print(row['price'])
        formatted_line = row.to_dict()
        formatted_line = dict((k.lower(), v) for k, v in formatted_line.items())

        formatted_line['row_id'] = formatted_line['id']
        del formatted_line['id']

        if dataset in ['amazon-google', 'dblp-acm_1', 'dblp-acm_2',
                                'dblp-googlescholar_1', 'dblp-googlescholar_2',
                                'walmart-amazon_1', 'walmart-amazon_2']:
            formatted_line['name'] = formatted_line['title']
            del formatted_line['title']
        elif 'wdcproducts' in dataset:
            formatted_line['name'] = formatted_line['title']
            del formatted_line['title']
        elif dataset in ['itunes-amazon_1', 'itunes-amazon_2']:
            formatted_line['name'] = formatted_line['song_name']
            del formatted_line['song_name']

        formatted_line['page_url'] = '{}_{}'.format(dataset, table_name)
        formatted_lines.append(formatted_line)

    # Make sure that path exists
    if not os.path.exists(path_to_corpus_table):
        os.makedirs(path_to_corpus_table)

    # Add file name
    path_to_corpus_table_file = path_to_corpus_table + '/{}_{}.json.gz'.format(dataset, table_name)
    with gzip.open(path_to_corpus_table_file, 'wb') as output_file:
        for formatted_line in formatted_lines:
            new_line = json.dumps(formatted_line) + '\n'
            output_file.write(new_line.encode())


    logging.info('Converted {} of {} to table format'.format(table_name, dataset))

    if path_to_additional_records is not None and 'wdcproducts' in dataset:
        # Determine original clusters
        unqiue_cluster_ids = set([formatted_line['cluster_id'] for formatted_line in formatted_lines])
        print(len(unqiue_cluster_ids))
        no_additional_records = index_size - len(formatted_lines)
        no_additional_record_files = 1
        collected_additional_records = []

        with gzip.open(path_to_additional_records, 'rb') as file:
            # 1. Fill index with normalized entities
            for line in file:
                additional_record = json.loads(line)
                if additional_record['cluster_id'] not in unqiue_cluster_ids:
                    additional_record['name'] = additional_record['title']
                    del additional_record['title']
                    additional_record['pricecurrency'] = additional_record['priceCurrency']
                    del additional_record['priceCurrency']
                    additional_record['row_id'] = additional_record['id']
                    del additional_record['id']

                    additional_record['page_url'] = '{}_{}'.format(dataset, 'additional_rows')

                    collected_additional_records.append(additional_record)

                    no_additional_records -= 1
                    if no_additional_records == 0:
                        break

                    if len(collected_additional_records) >= 5000:
                        save_additional_records(collected_additional_records, path_to_corpus_table, dataset, no_additional_record_files)
                        no_additional_record_files += 1
                        collected_additional_records = []
                else:
                    logging.debug('Skipped additional record from cluster {}'.format(additional_record['cluster_id']))
                    logging.debug(additional_record)

            save_additional_records(collected_additional_records, path_to_corpus_table, dataset,
                                    no_additional_record_files)

    logging.info('Added additional rows to index table such that it has a complete size of {}!'.format(str(index_size)))


def save_additional_records(collected_additional_records, path_to_corpus_table, dataset, additional_record_files):
    path_to_corpus_table_file = path_to_corpus_table + '/{}_additional_records_{}.json.gz'.format(dataset,
                                                                                                  str(additional_record_files))
    with gzip.open(path_to_corpus_table_file, 'wb') as output_file:
        for collected_additional_record in collected_additional_records:
            new_line = json.dumps(collected_additional_record) + '\n'
            output_file.write(new_line.encode())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_table_format()