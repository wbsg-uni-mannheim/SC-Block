import logging
import os
import time
from multiprocessing import Process, Queue

import click
import yaml
from elasticsearch import Elasticsearch
from tqdm import tqdm

from src.strategy.open_book.es_helper import determine_es_index_name
from src.strategy.open_book.indexing.faiss_collector import FaissIndexCollector
from src.strategy.open_book.retrieval.encoding.bi_encoder_factory import select_bi_encoder
from src.strategy.open_book.retrieval.query_by_entity import QueryByEntity


@click.command()
@click.option('--path_to_config')
@click.option('--dataset')
@click.option('--bi_encoder_name')
@click.option('--model_name')
@click.option('--base_model')
@click.option('--with_projection', default=False)
@click.option('--pooling', default='mean')
@click.option('--similarity_measure', default='cos')
@click.option('--dimensions', type=int)
@click.option('--clusters', default=False)
@click.option('--batch_size', default= 512)
def load_data(path_to_config, dataset, bi_encoder_name, model_name, base_model, with_projection, pooling, similarity_measure, dimensions, clusters, batch_size):
    logger = logging.getLogger()

    if path_to_config is not None:
        # Load yaml configuration
        with open(path_to_config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # validate_configuration(config) --> To-Do: Implement proper validation
        dataset = config['general']['dataset']
        model_name = config['bi_encoder_configuration']['model_name']
        batch_size = config['general']['indexing_batch_size']
        pooling = config['bi_encoder_configuration']['pooling']
        similarity_measure = config['bi_encoder_configuration'][
            'similarity_measure']  # Normalisation needed if similarity measure is cos
        bi_encoder_configuration = config['bi_encoder_configuration']
        dimensions = config['bi_encoder_configuration']['dimensions']
        clusters =  config['general']['clusters']
    else:
        bi_encoder_configuration = {'name': bi_encoder_name, 'model_name': model_name, 'base_model': base_model,
                                    'with_projection': with_projection, 'projection': dimensions, 'pooling': pooling,
                                    'normalize': True, similarity_measure: similarity_measure,
                                    'dimensions': dimensions }

    logger.info('Chosen Model {} for indexing schema org class {}'.format(model_name, dataset))

    start_time = time.time()

    strategy = QueryByEntity(dataset)

    _es = Elasticsearch([{'host': os.environ['ES_INSTANCE'], 'port': 9200}])
    entity_index_name = determine_es_index_name(dataset, clusters=clusters)
    logger.info('Create FAISS index for ES index {}'.format(entity_index_name))
    no_entities = int(_es.cat.count(entity_index_name, params={"format": "json"})[0]['count'])
    final_step = int(no_entities / batch_size) + 1

    # Initialize Faiss collector - TO-DO: Move configuration to yaml file!
    faiss_collector = FaissIndexCollector(dataset, model_name, pooling, similarity_measure, final_step,
                                          dimensions, clusters)

    input_q = Queue()
    output_q = Queue()

    processes = []

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        worker = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        worker = [1]

    for gpu_n in worker:
        for i in range(0,2):
            # Start multiple processes - 1 per GPU
            p = Process(target=generate_embeddings,
                        args=(bi_encoder_configuration, dataset, input_q, output_q, gpu_n,))
            p.start()
            processes.append(p)

    for i in tqdm(range(final_step)):
        # Determine retrieval range
        start = i * batch_size
        end = start + batch_size
        if end > no_entities:
            end = no_entities

        # Retrieve entities
        entities = strategy.query_tables_index_by_id(range(start, end), entity_index_name)
        if len(entities['hits']['hits']) != end - start:
            logger.warning('Did not receive all entities from {}!'.format(entity_index_name))
        start += batch_size

        # Encode entities sequentially
        entities = [entity['_source'] for entity in entities['hits']['hits']]

        # logger.info('Start batch processing for entities!')
        # generate_embeddings(model_name, dataset, entities)
        input_q.put((i, entities))

        # Collect results
        # Wait and persist results every 100th iteration
        collect = True
        wait_and_persist_indices = i % 50000 == 0
        if input_q.qsize() > 80 or output_q.qsize() > 20:
        #if input_q.qsize() > 300 or output_q.qsize() > 80:

            faiss_collector = collect_faiss_entities(output_q, faiss_collector, wait_and_persist_indices,
                                                         wait_and_persist_indices)
            logger.info('Input Size: {} - Output Size: {}'.format(input_q.qsize(), output_q.qsize()))

        if input_q.qsize() > 80:
            logger.info('{} Configurations available for processing'.format(input_q.qsize()))
            time.sleep(input_q.qsize() * 0.01)

    while True:
        faiss_collector = collect_faiss_entities(output_q, faiss_collector, False, False)

        if faiss_collector.next_representation == final_step:
            faiss_collector = collect_faiss_entities(output_q, faiss_collector, True, True)
            break

    for p in processes:
        p.terminate()
        p.join()
        p.close()

    input_q.close()
    output_q.close()

    indexing_time = time.time() - start_time

    logger.info('Added {} entities to FAISS indices'.format(faiss_collector.collected_entities))
    logger.info('Indexing time: {}'.format(indexing_time))


def generate_embeddings(bi_encoder_configuration, dataset, input_q, output_q, gpu_n):
    """Generate embeddings of entities"""

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_n)

    # Initialize Bi-Encoder
    bi_encoder = select_bi_encoder(bi_encoder_configuration, dataset)
    if bi_encoder_configuration['normalize']:
        collection_identifier = 'entity_vector_{}_norm'.format(bi_encoder_configuration['pooling'])
    else:
        collection_identifier = 'entity_vector_{}'.format(bi_encoder_configuration['pooling'])

    while True:

        i, entities = input_q.get()

        pooled_outputs = bi_encoder.encode_entities_and_return_pooled_outputs(entities)

        if len(entities) == 1:
            # Cover special case if only one entity was selected
            entities[0][collection_identifier] = pooled_outputs
        else:
            for entity, pooled_output in zip(entities, pooled_outputs):
                entity[collection_identifier] = pooled_output

        output_q.put((i, entities))


def generate_dummy_embeddings(model_name, dataset, input_q, output_q):
    """Generate dummy embeddings of for testing purposes"""

    # Initialize Bi-Encoder

    while True:

        i, entities = input_q.get()

        entities_cls = [[0,0,0]]*len(entities)
        entities_cls_norm = [[0,0,0]]*len(entities)
        entities_mean = [[0,0,0]]*len(entities)
        entities_mean_norm = [[0,0,0]]*len(entities)

        if len(entities) == 1:
            # Cover special case if only one entity was selected
            entities[0]['entity_vector_cls'] = [0,0,0]
            entities[0]['entity_vector_cls_norm'] = [0,0,0]
            entities[0]['entity_vector_mean'] = [0,0,0]
            entities[0]['entity_vector_mean_norm'] = [0,0,0]
        else:
            for entity, entity_cls, entity_cls_norm, entity_mean, entity_mean_norm in zip(entities, entities_cls,
                                                                                          entities_cls_norm,
                                                                                          entities_mean,
                                                                                          entities_mean_norm):
                entity['entity_vector_cls'] = entity_cls
                entity['entity_vector_cls_norm'] = entity_cls_norm
                entity['entity_vector_mean'] = entity_mean
                entity['entity_vector_mean_norm'] = entity_mean_norm

        output_q.put((i, entities))


def collect_faiss_entities(output_q, faiss_collector, wait_for_encodings, persist_indices):
    """Collect entities in faiss indicies and return number of collected entities"""
    logger = logging.getLogger()

    if not output_q.empty():
        faiss_collector.unsaved_representations.append(output_q.get(wait_for_encodings))
        while output_q.qsize() > 10:
            #Empty queue if necessary
            if not output_q.empty():
                faiss_collector.unsaved_representations.append(output_q.get(wait_for_encodings))

            logger.debug('Output size during collection: {}'.format(output_q.qsize()))

        entities = faiss_collector.next_savable_entities()

        while entities is not None:
            faiss_collector.collected_entities += len(entities)

            faiss_collector.initialize_entity_representations()
            for entity in entities:
                faiss_collector.collect_entity_representation(entity)

            faiss_collector.add_entity_representations_to_indices()

            entities = faiss_collector.next_savable_entities()

    if persist_indices:
        faiss_collector.save_indices()

    return faiss_collector

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_data()
