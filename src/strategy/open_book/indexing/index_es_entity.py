import gzip
import json
import os
import time

import click
from elasticsearch import Elasticsearch, helpers
import logging

from multiprocessing import Pool

from tqdm import tqdm

from src.preprocessing.entity_extraction import extract_entity
from src.strategy.open_book.entity_serialization import EntitySerializer
from src.strategy.open_book.es_helper import determine_es_index_name
from src.preprocessing.language_detection import LanguageDetector
from src.strategy.open_book.retrieval.retrieval_strategy import load_es_index_configuration


@click.command()
@click.option('--dataset')
@click.option('--worker', help='Number of workers', type=int, default=1)
@click.option('--tokenizer', help='Tokenizer for ES Index', default=None)
@click.option('--no-test/--test', default=True)
@click.option('--with-language-detection/--without-language-detection', default=False)
@click.option('--duplicate-check/--no-duplicate-check', default=False)
@click.option('--entity-length-check/--no-entity-length-check', default=False)
def load_data(dataset, worker, tokenizer, no_test, with_language_detection, duplicate_check, entity_length_check):
    logger = logging.getLogger()

    # Connect to Elasticsearch
    _es = Elasticsearch([{'host': os.environ['ES_INSTANCE'], 'port': 9200}], timeout=15, max_retries=3, retry_on_timeout=True)

    if not _es.ping():
        raise ValueError("Connection failed")

    start = time.time()
    clusters = None

    # Load data into one index:
    #  1. Index with a fixed schema (Schema is based on Schema.org)
    entity_index = 0
    mapping = load_es_index_configuration(tokenizer)
    entity_index_name = determine_es_index_name(dataset, tokenizer=tokenizer)

    #time.sleep(5)
    if no_test:
        _es.indices.delete(entity_index_name, ignore=[404])

        _es.indices.create(entity_index_name, body=mapping)

    # Collect statistics about added/ not added entities & tables
    index_statistics = {'tables_added': 0, 'tables_not_added': 0, 'entities_added': 0, 'entities_not_added': 0}

    directory = '{}/corpus/{}'.format(os.environ['DATA_DIR'], dataset)

    # Prepare parallel processing
    results = []
    if worker > 0:
        pool = Pool(worker)
    collected_filenames = []

    for filename in os.listdir(directory):

        if clusters is not None:
            #Check if table connected to file has clustered records
            file_table = filename.lower().split('_')[1]
            if file_table[:3] not in clusters:
                continue
            if file_table not in clusters[file_table[:3]]:
                continue

        collected_filenames.append(filename)
        if len(collected_filenames) > 50:
            if worker == 0:
                results.append(create_table_index_action(directory, collected_filenames, entity_index_name, dataset,
                                                         clusters, with_language_detection, duplicate_check, entity_length_check))
            else:
                results.append(
                    pool.apply_async(create_table_index_action, (directory, collected_filenames, entity_index_name,
                                                                 dataset, clusters, with_language_detection,
                                                                 duplicate_check, entity_length_check,)))
            collected_filenames = []

    if len(collected_filenames) > 0:
        if worker == 0:
            results.append(create_table_index_action(directory, collected_filenames, entity_index_name, dataset,
                                                     clusters, with_language_detection, duplicate_check, entity_length_check))
        else:
            results.append(
                 pool.apply_async(create_table_index_action, (directory, collected_filenames, entity_index_name,
                                                              dataset, clusters, with_language_detection,
                                                              duplicate_check, entity_length_check,)))

    pbar = tqdm(total=len(results))
    logger.info('Wait for all tasks to finish!')

    while True:
        if len(results) == 0:
            break

        results, entity_index = send_actions_to_elastic(_es, results, entity_index, index_statistics, pbar, no_test, worker)

    pbar.close()

    if worker > 0:
        pool.close()
        pool.join()

    execution_time = time.time() - start
    # Report statistics about indexing
    logger.info('Added entities: {}'.format(index_statistics['entities_added']))
    logger.info('Not added entities: {}'.format(index_statistics['entities_not_added']))
    logger.info('Added tables: {}'.format(index_statistics['tables_added']))
    logger.info('Not added tables: {}'.format(index_statistics['tables_not_added']))

    logger.info('Indexing time: {} sec'.format(execution_time))




def send_actions_to_elastic(_es, results, entity_index, index_statistics, pbar, no_test, worker):
    """Send actions to elastic and update statistics"""
    logger = logging.getLogger()

    collected_results = []
    actions = []

    for result in results:
        new_actions, new_statistics = None, None
        if worker == 0:
            new_actions, new_statistics = result
        elif result.ready():
            new_actions, new_statistics = result.get()

        if new_actions is not None and new_statistics is not None:
            logger.debug('Retrieved {} actions'.format(len(new_actions)))
            for action in new_actions:
                action['_id'] = entity_index
                actions.append(action)

                entity_index += 1

            index_statistics['entities_added'] += new_statistics['entities_added']
            index_statistics['entities_not_added'] += new_statistics['entities_not_added']
            collected_results.append(result)
            pbar.update(1)

    if len(actions) > 0 and no_test:
        # Add entities to ES
        helpers.bulk(client=_es, actions=actions, chunk_size=1000, request_timeout=60)

    # Remove collected results from list of results
    results = [result for result in results if result not in collected_results]

    return results, entity_index


def create_table_index_action(directory, files, entity_index, dataset, clusters, with_language_detection, duplicate_check, entity_length_check):
    """Creates entity document that will be index to elastic search"""
    log_format = '%(asctime)s - subprocess - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()
    entity_serializer = EntitySerializer(dataset)
    if with_language_detection:
        ld = LanguageDetector()

    actions = []
    index_statistics = {'entities_added': 0, 'entities_not_added': 0}

    for filename in files:
        file_path = '{}/{}'.format(directory, filename)

        if clusters is not None:
            #Check if table connected to file has clustered records
            file_table = filename.lower().split('_')[1]
            if file_table[:3] not in clusters:
                continue
            if file_table not in clusters[file_table[:3]]:
                continue
        else:
            file_table = None

        # Use 'entity' to find entity indices
        found_entities = []
        try:
            with gzip.open(file_path, 'rb') as file:
                # 1. Fill index with normalized entities
                for line in file.readlines():
                    # Extract entity name
                    raw_entity = json.loads(line)
                    if clusters is not None and file_table is not None:
                        if raw_entity['row_id'] not in clusters[file_table[:3]][file_table]:
                            index_statistics['entities_not_added'] += 1
                            continue

                    if 'name' in raw_entity \
                            and raw_entity['name'] is not None \
                            and len(raw_entity['name']) > 0:
                        # Detect language of entity
                        if with_language_detection and ld.check_language_is_not_english(raw_entity):
                            logger.debug('ENTITY INDEX ERROR - Language of entity {} is not english.'
                                         .format(raw_entity['name']))
                            index_statistics['entities_not_added'] += 1

                        else:
                            entity = extract_entity(raw_entity, dataset)
                            # Determine duplicates based on entity values without description
                            entity_wo_description = entity.copy()
                            if 'description' in entity_wo_description:
                                del entity_wo_description['description']

                            if 'name' in entity and \
                                (not entity_length_check or len(entity.keys()) > 1):
                                if duplicate_check:
                                    if entity_wo_description not in found_entities:
                                        found_entities.append(entity_wo_description)

                                        entity['table'] = filename.lower()
                                        entity['row_id'] = raw_entity['row_id']
                                        entity['page_url'] = raw_entity['page_url']
                                        entity['all_attributes'] = entity_serializer.convert_to_str_representation(entity)

                                        actions.append({'_index': entity_index, '_source': entity})
                                        index_statistics['entities_added'] += 1
                                    else:
                                        index_statistics['entities_not_added'] += 1
                                else:
                                    entity['table'] = filename.lower()
                                    entity['row_id'] = raw_entity['row_id']
                                    entity['page_url'] = raw_entity['page_url']
                                    entity['all_attributes'] = entity_serializer.convert_to_str_representation(entity)

                                    actions.append({'_index': entity_index, '_source': entity})
                                    index_statistics['entities_added'] += 1
                            else:
                                #print(entity)
                                index_statistics['entities_not_added'] += 1

                    else:
                        logger.debug(
                            'TABLE INDEX ERROR - Entity does not have a name attribute: {} - not added to index: {}'
                                .format(str(raw_entity), filename))
                        index_statistics['entities_not_added'] += 1

        except gzip.BadGzipFile as e:
            logger.warning('{} - Cannot open file {}'.format(e, filename))

    logger.debug('Added {} actions'.format(index_statistics['entities_added']))

    return actions, index_statistics


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_data()
