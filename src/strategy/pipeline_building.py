import itertools
import logging
import copy


def validate_configuration(config):
    """Validate configuration and raise ValueError exception if configuration is not valid"""

    def check_configuration_attributes(parent_attribute, child_attributes):
        for child_attribute in child_attributes:
            if child_attribute not in config[parent_attribute]:
                raise ValueError('Configuration for {} - {} missing'.format(parent_attribute, child_attribute))

    # Check if all general attributes are defined!
    general_attributes = ['k', 'k_range', 'save_results_with_evidences']
    check_configuration_attributes('general', general_attributes)
    query_table_attributes = ['dataset', 'context-attributes']
    check_configuration_attributes('query-tables', query_table_attributes)

    pipeline_steps = ['retrieval-strategies', 'similarity-re-ranking-strategies', 'source-re-ranking-strategies',
                        'voting-strategies']
    check_configuration_attributes('pipelines', pipeline_steps)

    return True


def build_pipelines_from_configuration(config):
    """Build pipelines from yaml configuration file and return these pipelines"""

    if validate_configuration(config):
        retrieval_strategies = config['pipelines']['retrieval-strategies']
        similarity_re_ranking_strategies = config['pipelines']['similarity-re-ranking-strategies']
        source_re_ranking_strategies = config['pipelines']['source-re-ranking-strategies']
        voting_strategies = config['pipelines']['voting-strategies']

        pipelines = []
        generate_entity_added = False
        gold_standard_added = False

        # Build pipelines
        for retrieval_strategy, similarity_re_ranking_strategy, source_re_ranking_strategy \
                in itertools.product(retrieval_strategies, similarity_re_ranking_strategies, source_re_ranking_strategies):



            if retrieval_strategy['name'] not in ['generate_entity', 'query_by_goldstandard']:
                if retrieval_strategy['name'] == 'query_by_neural_entity' or retrieval_strategy['name'] == 'query_by_entity_and_neural_entity':

                    for model_name, pooling, similarity in itertools.product(retrieval_strategy['model_name'],
                                                               retrieval_strategy['pooling'],
                                                               retrieval_strategy['similarity']):
                        specific_retrieval_strategy = {'name': retrieval_strategy['name'],
                                                       'bi-encoder': retrieval_strategy['bi-encoder'],
                                                       'model_name': model_name,
                                                       'base_model': retrieval_strategy['base_model'],
                                                       'with_projection': retrieval_strategy['with_projection'],
                                                       'projection': retrieval_strategy['projection'],
                                                       'pooling': pooling,
                                                       'similarity': similarity,
                                                       'without_special_tokens_and_attribute_names': retrieval_strategy['without_special_tokens_and_attribute_names'] if 'without_special_tokens_and_attribute_names' in retrieval_strategy else False,}
                        pipelines.append({'retrieval_strategy': specific_retrieval_strategy,
                                          'similarity_re_ranking_strategy': similarity_re_ranking_strategy,
                                          'source_re_ranking_strategy': source_re_ranking_strategy,
                                          'voting_strategies': voting_strategies})

                else:
                    pipelines.append({'retrieval_strategy': retrieval_strategy,
                                      'similarity_re_ranking_strategy': similarity_re_ranking_strategy,
                                      'source_re_ranking_strategy': source_re_ranking_strategy,
                                      'voting_strategies': voting_strategies})

            elif not generate_entity_added and retrieval_strategy['name'] == 'generate_entity':
                # No re-ranking and voting necessary if result is generated!
                pipelines.append({'retrieval_strategy': retrieval_strategy,
                                  'similarity_re_ranking_strategy': None,
                                  'source_re_ranking_strategy': None,
                                  'voting_strategies': None})
                generate_entity_added = True

            elif not gold_standard_added and retrieval_strategy['name'] == 'query_by_goldstandard':
                # No re-ranking and voting necessary if result is generated!
                pipelines.append({'retrieval_strategy': retrieval_strategy,
                                  'similarity_re_ranking_strategy': similarity_re_ranking_strategy,
                                  'source_re_ranking_strategy': None,
                                  'voting_strategies': voting_strategies})
                gold_standard_added = True

        # Check if pipelines need to be augmented
        all_pipelines = []
        for pipeline in pipelines:
            if pipeline['retrieval_strategy']['name'] == 'query_by_neural_entity' and \
                'TRAININGSETSIZE' in pipeline['retrieval_strategy']['model_name']:
                for percentage in range(5, 105, 5):
                    new_pipeline = copy.deepcopy(pipeline)
                    new_pipeline['retrieval_strategy']['model_name'] = new_pipeline['retrieval_strategy']['model_name'].replace('TRAININGSETSIZE', str(percentage))
                    all_pipelines.append(new_pipeline)

            elif pipeline['retrieval_strategy']['name'] == 'combined_retrieval_strategy' and \
                    pipeline['retrieval_strategy']['retrieval_strategy_2']['name'] == 'query_by_neural_entity' and \
                'TRAININGSETSIZE' in pipeline['retrieval_strategy']['retrieval_strategy_2']['model_name']:
                for percentage in range(5, 105, 5):
                    new_pipeline = copy.deepcopy(pipeline)
                    new_pipeline['retrieval_strategy']['retrieval_strategy_2']['model_name'] = pipeline['retrieval_strategy']['retrieval_strategy_2']['model_name'].replace('TRAININGSETSIZE', str(percentage))
                    all_pipelines.append(new_pipeline)
            else:
                all_pipelines.append(pipeline)

        logging.info('Built {} Pipelines!'.format(len(all_pipelines)))

        return all_pipelines

    else:
        return None

