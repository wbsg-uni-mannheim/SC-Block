import logging

from src.strategy.retrieval.query_by_entity import QueryByEntity
from src.strategy.retrieval.combined_retrieval_strategy import CombinedRetrievalStrategy
from src.strategy.retrieval.query_by_goldstandard import QueryByGoldStandard
from src.strategy.retrieval.query_by_neural_entity import QueryByNeuralEntity
from src.strategy.retrieval.query_by_table import QueryByTable
from src.strategy.retrieval.retrieval_strategy import RetrievalStrategy


def select_retrieval_strategy(retrieval_strategy, dataset, clusters, switched=False):
    """
    Initialize Table Augmentation Strategy

    :param strategy: Chosen table augmentation strategy
    :param dataset: Chosen schema org class
    :return: RetrievalStrategy: Initialized Retrieval Strategy
    """
    strategy_name = retrieval_strategy['name']
    logger = logging.getLogger()
    logger.info('Select Retrieval Strategy {}!'.format(strategy_name))
    logger.info('Retrieval Strategy Switched: {}'.format(switched))

    if strategy_name == 'query_by_table':
        strategy_obj = QueryByTable(dataset, clusters)
    elif strategy_name == 'query_by_table_boolean':
        strategy_obj = QueryByTable(dataset, clusters)
    elif strategy_name == 'query_by_entity':
        strategy_obj = QueryByEntity(dataset, clusters, all_attributes=retrieval_strategy['all_attributes'],
                                     tokenizer=retrieval_strategy['tokenizer'], switched=switched)
    elif strategy_name == 'query_by_entity_rank_by_table':
        strategy_obj = QueryByEntity(dataset, clusters, rank_evidences_by_table=True)
    elif strategy_name == 'query_by_neural_entity':
        without_special_tokens_and_attribute_names = retrieval_strategy['without_special_tokens_and_attribute_names'] \
            if 'without_special_tokens_and_attribute_names' in retrieval_strategy else False
        strategy_obj = QueryByNeuralEntity(dataset, retrieval_strategy['bi-encoder'], clusters,
                                           retrieval_strategy['model_name'], retrieval_strategy['base_model'],
                                           retrieval_strategy['with_projection'], retrieval_strategy['projection'],
                                           retrieval_strategy['pooling'], retrieval_strategy['similarity'],
                                           without_special_tokens_and_attribute_names=without_special_tokens_and_attribute_names,
                                           switched=switched)
    elif strategy_name == 'combined_retrieval_strategy':
        # Initialize both combined retrieval strategies before handing them over to the combined retrieval strategy
        retrieval_strategy_1 = select_retrieval_strategy(retrieval_strategy['retrieval_strategy_1'], dataset, clusters)
        retrieval_strategy_2 = select_retrieval_strategy(retrieval_strategy['retrieval_strategy_2'], dataset,
                                                         clusters)
        strategy_obj = CombinedRetrievalStrategy(dataset, retrieval_strategy_1, retrieval_strategy_2, clusters)
    #elif strategy_name == 'generate_entity':
    #    strategy_obj = TargetAttributeValueGenerator(schema_org_class, retrieval_strategy['model_name'])
    elif strategy_name == 'query_by_goldstandard':
        strategy_obj = QueryByGoldStandard(dataset, clusters)
    else:
        # Fall back to default open book strategy
        strategy_obj = RetrievalStrategy(dataset, None, clusters=clusters)

    return strategy_obj