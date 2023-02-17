import gzip
import logging
import os

import click
from elasticsearch import Elasticsearch
from tqdm import tqdm

from src.strategy.open_book.es_helper import determine_es_index_name
from src.strategy.open_book.ranking.source.source_re_ranker import extract_host
from src.strategy.open_book.retrieval.query_by_entity import QueryByEntity


@click.command()
@click.option('--schema_org_class')
@click.option('--step_size', help='Number of entities processed at once', type=int, default=9000)
@click.option('--level', help='host/domain',  default='domain')
def collect_relevant_page_ranks(schema_org_class, step_size, level):
    logger = logging.getLogger()

    strategy = QueryByEntity(schema_org_class)

    _es = Elasticsearch([{'host': os.environ['ES_INSTANCE'], 'port': 9200}])
    entity_index_name = determine_es_index_name(schema_org_class)
    no_entities = int(_es.cat.count(entity_index_name, params={"format": "json"})[0]['count'])
    final_step = int(no_entities / step_size) + 1

    # Collect table names
    table_names = set()
    for i in tqdm(range(final_step)):
        # Determine retrieval range
        start = i * step_size
        end = start + step_size
        if end > no_entities:
            end = no_entities

        # Retrieve entities
        entities = strategy.query_tables_index_by_id(range(start, end), entity_index_name)
        if len(entities['hits']['hits']) != end - start:
            logger.warning('Did not receive all entities from {}!'.format(entity_index_name))
        start += step_size

        for hit in entities['hits']['hits']:
            table_names.add(hit['_source']['table'])

    # Collect hostnames
    hosts = set()
    for table_name in tqdm(table_names):
        # Extract host from table_name
        hosts.add(extract_host(table_name))

    logger.info('Found {} hosts!'.format(len(hosts)))

    # Initialize dict with all hosts
    host_dict = { host: None for host in hosts}

    logger.info('Extract page ranks from common crawl file.')
    # Load Initial Page rank from Common Crawl file
    file_path = '{}/ranking/cc-main-2020-jul-aug-sep-{}-ranks.txt.gz'.format(os.environ['DATA_DIR'], level)
    found_values = 0
    with gzip.open(file_path, 'r') as file:
        for line in tqdm(file):
            decoded_line = line.decode('utf-8')
            line_values = decoded_line.split('\t')
            if level == 'host':
                assert len(line_values) == 5
            elif level == 'domain':
                assert len(line_values) == 6
            host = line_values[4].replace('\n', '')
            if host in host_dict:
                host_dict[host] = decoded_line
                found_values += 1
                if found_values == len(host_dict):
                    # Run as long as a value is found for every host!
                    break

    logger.info('Write Page Rankes of hosts to file!')
    output_file_path = '{}/ranking/cc-main-2020-jul-aug-sep-relevant-tc-{}-page-ranks-{}.txt'\
        .format(os.environ['DATA_DIR'], level, schema_org_class)

    not_found = 0
    with open(output_file_path, "w") as outfile:
        if level == 'host':
            outfile.write('harmonic centrality rank\thc value\tpage rank\treserved hostname')
        elif level == 'domain':
            outfile.write('harmonic centrality rank\thc value\tpage rank\treserved hostname\tx')
        for host in host_dict.keys():
            if host_dict[host] is None:
                logger.warning('No ranking information found for host {}'.format(host))
                not_found += 1
            else:
                outfile.write(host_dict[host])

    logger.info('Did not find page ranks for {} hosts.'.format(not_found))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    collect_relevant_page_ranks()