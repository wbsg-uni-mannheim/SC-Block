import gzip
import json
import logging

from elasticsearch import helpers

from src.preprocessing.value_normalizer import normalize_value, get_datatype


def determine_es_index_name(schema_org_class, table=False, tokenizer=None, clusters=False):
    if table:
        index_name = '{}_{}'.format(schema_org_class, 'table_index')
    else:
        index_name = '{}_{}'.format(schema_org_class, 'entity_index')

    if tokenizer is not None:
        # If a special tokenizer is used, the information is append to the index name
        index_name = '{}_{}'.format(index_name, tokenizer)

    if clusters:
        index_name = '{}_{}'.format(index_name, 'only_clustered')

    logging.getLogger().info('Index name {}'.format(index_name))

    return index_name


def index_table(path_to_table_corpus, schema_org_class, table_file_name, table_index_name, es):
    logger = logging.getLogger()
    # Index table
    file_path = '{}{}/{}'.format(path_to_table_corpus, schema_org_class, table_file_name)
    actions = []
    with gzip.open(file_path, 'rb') as file:
        entity_index_number = 0
        for line in file.readlines():
            entity_index_number += 1
            # Extract entity
            raw_entity = json.loads(line)

            if 'name' in raw_entity:
                entity = {'table': table_index_name, 'row_id': raw_entity['row_id'],
                          'page_url': raw_entity['page_url']}

                # Normalize/ unpack raw_entity if necessary
                for key in raw_entity.keys():
                    if type(raw_entity[key]) is str:
                        entity[key] = normalize_value(raw_entity[key], get_datatype(key), raw_entity)
                    elif type(raw_entity[key]) is dict:
                        # First case: property has name sub-property --> lift name
                        if 'name' in raw_entity[key]:
                            entity[key] = normalize_value(raw_entity[key]['name'], get_datatype(key), raw_entity)
                        # Second case: lift all values by sub-property name
                        else:
                            for property_key in raw_entity[key].keys():
                                entity[property_key] = normalize_value(raw_entity[key][property_key], get_datatype(key), raw_entity)
                    elif type(raw_entity[key]) is list:
                        # Check if element type is str
                        if all(type(element) is str for element in raw_entity[key]):
                            entity[key] = [normalize_value(element, get_datatype(key), raw_entity) for element in raw_entity[key]]
                        # Check if nested object has name attribute
                        elif all(type(element) is dict for element in raw_entity[key]) \
                                and all('name' in element for element in raw_entity[key]):
                            entity[key] = [normalize_value(element['name'], get_datatype(key), raw_entity) for element in
                                           raw_entity[key]]

                actions.append({'_index': table_index_name, '_source': entity})

            else:
                logger.warning(
                    'TABLE INDEX ERROR - Entity does not have a name attribute: {} - not added to index: {}'
                        .format(str(raw_entity), table_index_name))

    mapping = {"settings": {"number_of_shards": 1}, "mappings": {"date_detection": False}}
    es.indices.create(index=table_index_name, ignore=400, body=json.dumps(mapping))
    try:
        helpers.bulk(client=es, actions=actions, request_timeout=30)
    except helpers.BulkIndexError as e:
        logger.warning(e)
    logger.info('Table {} indexed'.format(table_file_name))

