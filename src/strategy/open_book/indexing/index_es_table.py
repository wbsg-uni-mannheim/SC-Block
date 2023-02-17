import gzip
import json
import os
import time

import click
from elasticsearch import Elasticsearch, helpers
import logging

from multiprocessing import Pool


@click.command()
@click.option('--schema_org_class')
@click.option('--worker', help='Number of workers', type=int)
def load_data(schema_org_class, worker):
    logger = logging.getLogger()

    path_to_data_dir = os.environ['DATA_DIR']
    elasticsearch_host = os.environ['ES_INSTANCE']
    # Connect to Elasticsearch
    _es = Elasticsearch([{'host': elasticsearch_host, 'port': 9200}])
    _es.ping()

    # Prepare parallel processing
    results = []
    process_count = 0

    # Load data into one:
    #  1. Index with a fixed schema (Schema is based on Schema.org)
    doc_index_number = 1
    table_index_name = '{}_{}'.format('table_index', schema_org_class)

    mapping = '''{"mappings": {
                "date_detection": false,
                "properties": {
                "content": {
                    "type": "text" 
            },
                "boolean_content": {
                    "type": "text",
                    "similarity": "boolean" 
            }}}}'''
    _es.indices.delete(table_index_name, ignore=[404])
    _es.indices.create(table_index_name, body=mapping)

    # Collect statistics about added/ not added entities & tables
    index_statistics = {}
    index_statistics['tables_added'] = 0
    index_statistics['tables_not_added'] = 0
    index_statistics['entities_added'] = 0
    index_statistics['entities_not_added'] = 0

    directory = '{}/corpus/{}'.format(path_to_data_dir, schema_org_class)
    pool = Pool(worker)
    for filename in os.listdir(directory):
        results.append(pool.apply_async(create_table_index_action, (directory, filename, table_index_name,)))

        process_count += 1
        logger.info('Submitted {} tasks!'.format(process_count))

        # Bulk upload current results
        while True:
            results, doc_index_number, index_statistics = \
                send_actions_to_elastic(_es, results, doc_index_number, index_statistics)

            if len(results) < worker:
                break

    logger.info('Wait for all tasks to finish!')

    while True:
        if len(results) == 0:
            break
        else:
            time.sleep(5)

        results, doc_index_number, index_statistics = \
            send_actions_to_elastic(_es, results, doc_index_number, index_statistics)

    pool.close()
    pool.join()

    # Report statistics about indexing
    logger.info('Added entities: {}'.format(index_statistics['entities_added']))
    logger.info('Not added entities: {}'.format(index_statistics['entities_not_added']))
    logger.info('Added tables: {}'.format(index_statistics['tables_added']))
    logger.info('Not added tables: {}'.format(index_statistics['tables_not_added']))


def send_actions_to_elastic(_es, results, doc_index_number, index_statistics):
    """Send actions to elastic and update statistics"""
    logger = logging.getLogger()

    collected_results = []
    actions = []
    for result in results:
        if result.ready():
            new_actions, new_statistics = result.get()
            logger.debug('Retrieved {} actions'.format(len(new_actions)))
            for action in new_actions:
                action['_id'] = doc_index_number
                actions.append(action)
                doc_index_number += 1

            index_statistics['entities_added'] += new_statistics['entities_added']
            index_statistics['entities_not_added'] += new_statistics['entities_not_added']
            index_statistics['tables_added'] += new_statistics['tables_added']
            index_statistics['tables_not_added'] += new_statistics['tables_not_added']
            collected_results.append(result)

    if len(actions) > 0:
        helpers.bulk(client=_es, actions=actions, chunk_size=1000)

    # Remove collected results from list of results
    results = [result for result in results if result not in collected_results]

    return results, doc_index_number, index_statistics


def create_table_index_action(directory, filename, table_index_name):
    log_format = '%(asctime)s - subprocess - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    logger = logging.getLogger()
    actions = []
    table_added = False
    index_statistics = {'entities_added': 0, 'entities_not_added': 0, 'tables_added': 0, 'tables_not_added': 0}
    
    file_path = '{}/{}'.format(directory, filename)
    # Use 'entity' to find entity indices
    try:
        with gzip.open(file_path, 'rb') as file:

            # 1. Fill Entities Index - Index on demand (!) --> Not necessary to index all data upfront
            table_name_column = {'table_name': filename, 'content': '', 'boolean_content': '', 'schema': ''}
            schema = set()
            
            for line in file.readlines():

                # Extract entity name
                raw_entity = json.loads(line)

                if 'name' in raw_entity:
                    # Collect Table Name Column for Tables Index
                    if type(raw_entity['name']) is list:
                        # Check if all values are of type string
                        raw_entity['name'] = ' '.join([name for name in raw_entity['name'] if type(name) is str])

                    if type(raw_entity['name']) is str and len(raw_entity['name']) > 0:
                        # Update schema
                        schema.union(set(raw_entity.keys()))
                        table_name_column['content'] = ' '.join([table_name_column['content'], raw_entity['name']])
                        table_name_column['boolean_content'] = table_name_column['content']

                        # The following code block is relevant if the text type keyword is used!

                        # if (len(table_name_column['content']) + len(raw_entity['name'])) < 256:
                        #     # Only include tables if their content is not empty
                        #     table_name_column['content'] = ' '.join([table_name_column['content'], raw_entity['name']])
                        #     table_name_column['boolean_content'] = table_name_column['content']
                        # else:
                        #     # Deal with long tables --> Split into multiple documents
                        #     # Elastic search ignores strings longer than 256 characters
                        #     table_name_column['schema'] = ', '.join(schema)
                        #     action = {'_index': table_index_name, '_source': table_name_column.copy()}
                        #     actions.append(action)
                        #     table_added = True
                        #
                        #     schema = set(raw_entity.keys())
                        #     table_name_column['content'] = raw_entity['name']
                        #     table_name_column['boolean_content'] = table_name_column['content']

                        index_statistics['entities_added'] += 1

                    else:
                        logger.warning(
                            'ENTITY NAME ERROR - Entity not added to index {} - Entity name is not a string value: {}'
                                .format(filename, raw_entity))
                        index_statistics['entities_not_added'] += 1

                else:
                    logger.warning(
                        'TABLE INDEX ERROR - Entity does not have a name attribute: {} - not added to index: {}'
                            .format(str(raw_entity), filename))
                    index_statistics['entities_not_added'] += 1

    except gzip.BadGzipFile as e:
        logger.warning('{} - Cannot open file {}'.format(e, filename))

    if len(table_name_column['content']) > 0:
        table_name_column['schema'] = ', '.join(schema)
        action = {'_index': table_index_name, '_source': table_name_column.copy()}
        actions.append(action)
        table_added = True

    if table_added:
        index_statistics['tables_added'] += 1
    else:
        index_statistics['tables_not_added'] += 1

    logger.debug('Added {} actions'.format(len(actions)))

    return actions, index_statistics


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_data()
