#!/usr/bin/env python3

import logging
import time
from datetime import datetime

import numpy as np
from torch.multiprocessing import Pool, set_start_method
from random import randrange
import random

import click
import torch
import yaml

from src.evaluation.aggregate_results import aggregate_results, save_aggregated_result
from src.evaluation.evaluate_query_tables import evaluate_query_table
from src.model.querytable import load_query_table_from_file, get_gt_tables, get_query_table_paths
from src.strategy.ranking.similarity.similarity_re_ranking_factory import select_similarity_re_ranker
from src.strategy.retrieval.retrieval_strategy_factory import select_retrieval_strategy
from src.strategy.pipeline_building import build_pipelines_from_configuration, validate_configuration

def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@click.command()
@click.option('--path_to_config')
@click.option('--worker', type=int, default=0)
def run_experiments_from_configuration(path_to_config, worker):
    logger = logging.getLogger()

    set_seed(42)
    # Load yaml configuration
    with open(path_to_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validate_configuration(config)

    config_name = path_to_config.split('/')[-1].replace('.yml','')

    context_attributes = config['query-tables']['context-attributes']
    experiment_type = config['general']['experiment-type']

    # Load query tables
    dataset = config['query-tables']['dataset']
    switched = config['query-tables']['switched'] if 'switched' in config['query-tables'] else False
    logger.info('Switched: {}'.format(switched))
    query_table_paths = []
    if type(config['query-tables']['path-to-query-table']) is str:
        # Run for single query table
        query_table_paths.append(config['query-tables']['path-to-query-table']) # query_table_paths must be an array
    elif config['query-tables']['gt-table'] is not None:
        # Run on query tables for gt table
        query_table_paths.extend(get_query_table_paths(config['general']['experiment-type'],
                                                  config['query-tables']['dataset'],
                                                  config['query-tables']['gt-table'], switched=switched))
    else:
        # Run on all query tables of the dataset
        for gt_table in get_gt_tables(config['general']['experiment-type'], dataset):
            query_table_paths.extend(get_query_table_paths(config['general']['experiment-type'], dataset,
                                                           gt_table, switched=switched))

    string_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(query_table_paths[0])

    if config['general']['k'] is not None:
        k_range = [config['general']['k']]
    elif config['general']['k_range'] is not None:
        k_range = range(config['general']['k_range'][0], config['general']['k_range'][1] + 1)

    logger.info(f'Will test the following values of k: {k_range}')

    save_results_with_evidences = config['general']['save_results_with_evidences']
    clusters = config['general']['clusters']
    #os.environ["ES_INSTANCE"] = config['general']['es_instance']

    pool = None
    async_results = None
    if worker > 0:
        pool = Pool(worker)
        async_results = []

    file_name = 'results_{}_{}.json'.format(string_timestamp, config_name)

    # Build pipelines from yaml configuration
    pipelines = build_pipelines_from_configuration(config)

    # Start run experiments by combining pipelines and query tables
    for k in k_range:
        for pipeline in pipelines:
            retrieval_strategy = pipeline['retrieval_strategy']
            similarity_re_ranking_strategy = pipeline['similarity_re_ranking_strategy']
            source_re_ranking_strategy = pipeline['source_re_ranking_strategy']
            voting_strategies = pipeline['voting_strategies']

            if worker == 0:
                results, execution_times = run_experiments(experiment_type, retrieval_strategy, similarity_re_ranking_strategy,
                                                           source_re_ranking_strategy,
                                                           voting_strategies, query_table_paths, dataset, k,
                                                           context_attributes, clusters=clusters, switched=switched)
                if results is not None:
                    for result in results:
                        result.save_result(file_name, save_results_with_evidences)

                    aggregated_result = aggregate_results(results, k, execution_times)
                    save_aggregated_result(aggregated_result, file_name)

            elif worker > 0:

                async_results.append(pool.apply_async(run_experiments, (experiment_type, retrieval_strategy,
                                                                        similarity_re_ranking_strategy,
                                                                        source_re_ranking_strategy, voting_strategies,
                                                                        query_table_paths, dataset, k,
                                                                        context_attributes, clusters, switched,)))

        if worker > 0:
            logger.info('Waiting for all experiments to finish!')

            while len(async_results) > 0:
                logger.info('Number of chunks: {}'.format(len(async_results)))
                time.sleep(5)
                async_results = collect_results_of_finished_experiments(async_results, file_name, k,
                                                                        save_results_with_evidences, True)

    if worker > 0:
        pool.close()
    logger.info('Finished running experiments!')


def run_experiments(experiment_type, retrieval_str_conf, similarity_re_ranking_str_conf, source_re_ranking_str_conf,
                    voting_strategies, query_table_paths, dataset, evidence_count, context_attributes=None, clusters=False, switched=False):
    """Run Pipeline on query tables"""

    time.sleep(randrange(30))
    logger = logging.getLogger()
    # Initialize strategy
    retrieval_strategy = select_retrieval_strategy(retrieval_str_conf, dataset, clusters, switched)
    similarity_re_ranker = select_similarity_re_ranker(similarity_re_ranking_str_conf, dataset,
                                                       context_attributes)
    #source_re_ranker = select_source_re_ranker(source_re_ranking_str_conf, dataset)
    source_re_ranker = None # Exclude source re-ranking for now
    logger.info('Run experiments on {} query tables'.format(len(query_table_paths)))
    results = []
    execution_times = []

    materialized_pairs = []
    for query_table_path in query_table_paths:

        query_table = load_query_table_from_file(query_table_path)
        # FIX context attributes
        if experiment_type == 'augmentation' and context_attributes is not None:
            if query_table.target_attribute in context_attributes:
                continue
            # Run experiments only on a subset of context attributes
            removable_attributes = [attr for attr in query_table.context_attributes
                                    if attr not in context_attributes and attr != 'name']
            for attr in removable_attributes:
                query_table.remove_context_attribute(attr)

        query_table.retrieved_evidences, execution_times_per_run = retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                                     similarity_re_ranker, source_re_ranker)
        materialized_pairs.extend(query_table.materialize_pairs())
        execution_times.append(execution_times_per_run)

        if retrieval_str_conf['name'] == 'generate_entity':
            k_intervals = [1]
        else:
            k_intervals = [evidence_count]

        for voting_str_conf in voting_strategies:
            split = None if similarity_re_ranker is None else 'test'
            new_results = evaluate_query_table(query_table, experiment_type, retrieval_strategy, similarity_re_ranker,
                                               source_re_ranker, k_intervals, voting_str_conf['name'], split=split,
                                               collect_result_context=True)
            results.extend(new_results)

    aggregated_execution_times = {key: sum([execution_time[key] for execution_time in execution_times])
                                  for key in execution_times[0]}

    logger.info('Finished running experiments on subset of query tables!')

    return results, aggregated_execution_times


def retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                     similarity_re_ranker, source_re_ranker, entity_id=None):
    execution_times = {}
    start_time = time.time()

    # Run retrieval strategy
    evidences = retrieval_strategy.retrieve_evidence(query_table, evidence_count, entity_id)
    retrieval_time = time.time()
    execution_times['retrieval_time'] = retrieval_time - start_time

    # Filter evidences by ground truth tables
    evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(evidences)

    # Run re-ranker
    if similarity_re_ranker is not None:
        # Re-rank evidences by cross encoder - to-do: Does it make sense to track both bi encoder and reranker?
        evidences = similarity_re_ranker.re_rank_evidences(query_table, evidences)
        similarity_re_ranking_time = time.time()
        execution_times['sim_reanker_time'] = similarity_re_ranking_time - retrieval_time

    if source_re_ranker is not None:
        # Re-rank evidences by source information
        evidences = source_re_ranker.re_rank_evidences(query_table, evidences)
        source_re_ranking_time = time.time()
        if similarity_re_ranker is not None:
            execution_times['source_reranker_time'] = source_re_ranking_time - similarity_re_ranking_time
        else:
            execution_times['source_reranker_time'] = source_re_ranking_time - retrieval_time

    execution_times['complete_execution_time'] = time.time() - start_time

    return evidences, execution_times


def collect_results_of_finished_experiments(async_results, file_name, evidence_count, with_evidences=True, with_extended_results=False):
    """Collect results and write them to file"""
    logger = logging.getLogger()
    collected_results = []
    for async_result in async_results:
        if async_result.ready():
            results, execution_times = async_result.get()
            collected_results.append(async_result)

            # Save query table to file
            if results is not None:
                logger.info('Will collect {} results now!'.format(len(results)))
                if with_extended_results:
                    for result in results:
                        result.save_result(file_name, with_evidences)

                #for i in range(1, 11):
                aggregated_result = aggregate_results(results, evidence_count, execution_times)
                save_aggregated_result(aggregated_result, file_name)

    # Remove collected results from list of results
    async_results = [async_result for async_result in async_results if async_result not in collected_results]

    return async_results


def run_strategy_to_retrieve_evidence(query_table_id, schema_org_class, experiment_type, retrieval_str_conf,
                                      similarity_re_ranking_str_conf, source_re_ranking_str_conf, entity_id=None):
    # TO-DO: UPDATE SO THAT THE ANNOTATION TOOL CONTINUES TO WORK!
    # Initialize Table Augmentation Strategy
    evidence_count = 30  # Deliver 20 evidence records for now

    print(retrieval_str_conf)
    #To-Do: Does it make sense to set clusters always to true?
    retrieval_strategy = select_retrieval_strategy(retrieval_str_conf, schema_org_class, clusters=True)
    similarity_re_ranker = select_similarity_re_ranker(similarity_re_ranking_str_conf, schema_org_class)
    #source_re_ranker = select_source_re_ranker(source_re_ranking_str_conf, schema_org_class)
    source_re_ranker = None # Exclude source re-ranker for now

    query_table = None
    context_attributes = ['name', 'addresslocality']

    for gt_table in get_gt_tables(experiment_type, schema_org_class):
        for query_table_path in get_query_table_paths('retrieval', schema_org_class, gt_table):
            if query_table_path.endswith('_{}.json'.format(query_table_id)):
                query_table = load_query_table_from_file(query_table_path)
                # Run experiments only on a subset of context attributes
                removable_attributes = [attr for attr in query_table.context_attributes
                                        if attr not in context_attributes and attr != 'name']
                for attr in removable_attributes:
                    query_table.remove_context_attribute(attr)

    evidences = retrieve_evidences_with_pipeline(query_table, retrieval_strategy, evidence_count,
                                                 similarity_re_ranker, source_re_ranker, entity_id=entity_id)

    # Return evidence --> (Filter for single entity)
    requested_evidence = [evidence for evidence in evidences
                          if evidence.entity_id == entity_id]

    return requested_evidence[:evidence_count]


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    set_start_method('spawn')
    run_experiments_from_configuration()
