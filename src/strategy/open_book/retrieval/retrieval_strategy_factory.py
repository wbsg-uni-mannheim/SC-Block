import logging

from src.strategy.closed_book.generate_target_attribute_value import TargetAttributeValueGenerator
from src.strategy.open_book.retrieval.query_by_entity import QueryByEntity
from src.strategy.open_book.retrieval.combined_retrieval_strategy import CombinedRetrievalStrategy
from src.strategy.open_book.retrieval.query_by_goldstandard import QueryByGoldStandard
from src.strategy.open_book.retrieval.query_by_neural_entity import QueryByNeuralEntity
from src.strategy.open_book.retrieval.query_by_table import QueryByTable
from src.strategy.open_book.retrieval.retrieval_strategy import RetrievalStrategy


def select_retrieval_strategy(retrieval_strategy, schema_org_class, clusters):
    """
    Initialize Table Augmentation Strategy

    :param strategy: Chosen table augmentation strategy
    :param schema_org_class: Chosen schema org class
    :return: RetrievalStrategy: Initialized Retrieval Strategy
    """
    strategy_name = retrieval_strategy['name']
    logger = logging.getLogger()
    logger.info('Select Retrieval Strategy {}!'.format(strategy_name))

    if strategy_name == 'query_by_table':
        strategy_obj = QueryByTable(schema_org_class, clusters)
    elif strategy_name == 'query_by_table_boolean':
        strategy_obj = QueryByTable(schema_org_class, clusters)
    elif strategy_name == 'query_by_entity':
        strategy_obj = QueryByEntity(schema_org_class, clusters, all_attributes=retrieval_strategy['all_attributes'],
                                     tokenizer=retrieval_strategy['tokenizer'])
    elif strategy_name == 'query_by_entity_rank_by_table':
        strategy_obj = QueryByEntity(schema_org_class, clusters, rank_evidences_by_table=True)
    elif strategy_name == 'query_by_neural_entity':
        strategy_obj = QueryByNeuralEntity(schema_org_class, retrieval_strategy['bi-encoder'], clusters,
                                           retrieval_strategy['model_name'], retrieval_strategy['base_model'],
                                           retrieval_strategy['with_projection'], retrieval_strategy['projection'],
                                           retrieval_strategy['pooling'], retrieval_strategy['similarity'])
    elif strategy_name == 'combined_retrieval_strategy':
        # Initialize both combined retrieval strategies before handing them over to the combined retrieval strategy
        retrieval_strategy_1 = select_retrieval_strategy(retrieval_strategy['retrieval_strategy_1'], schema_org_class, clusters)
        retrieval_strategy_2 = select_retrieval_strategy(retrieval_strategy['retrieval_strategy_2'], schema_org_class,
                                                         clusters)
        strategy_obj = CombinedRetrievalStrategy(schema_org_class, retrieval_strategy_1, retrieval_strategy_2, clusters)
    elif strategy_name == 'generate_entity':
        strategy_obj = TargetAttributeValueGenerator(schema_org_class, retrieval_strategy['model_name'])
    elif strategy_name == 'query_by_goldstandard':
        strategy_obj = QueryByGoldStandard(schema_org_class, clusters)
    else:
        # Fall back to default open book strategy
        strategy_obj = RetrievalStrategy(schema_org_class, None, clusters=clusters)

    return strategy_obj