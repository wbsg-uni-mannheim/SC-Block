import logging

from src.strategy.open_book.ranking.similarity.ditto_re_ranker import DittoSimilarityReRanker
from src.strategy.open_book.ranking.similarity.hf_re_ranker import HuggingfaceSimilarityReRanker
from src.strategy.open_book.ranking.similarity.magellan_re_ranker import MagellanSimilarityReRanker
from src.strategy.open_book.ranking.similarity.similarity_re_ranker import SimilarityReRanker
from src.strategy.open_book.ranking.similarity.subcon_re_ranker import SupConSimilarityReRanker
from src.strategy.open_book.ranking.similarity.symbolic_re_ranker import SymbolicSimilarityReRanker
from src.strategy.open_book.ranking.similarity.threshold_re_ranker import ThresholdReRanker


def select_similarity_re_ranker(re_ranking_strategy, schema_org_class, context_attributes=None):
    """Return a re-ranker based on the defined re-ranking-strategy
        :return SimilarityReRanker
    """
    logger = logging.getLogger()

    if re_ranking_strategy is None:
        re_ranking_strategy_name = None
    else:
        re_ranking_strategy_name = re_ranking_strategy['name']

    logger.info('Select Similarity Re-ranking Strategy {}!'.format(re_ranking_strategy_name))

    if re_ranking_strategy_name == 'huggingface_re_ranker':
        re_ranking_strategy = HuggingfaceSimilarityReRanker(schema_org_class, re_ranking_strategy['model_name'],
                                                            context_attributes, re_ranking_strategy['matcher'])
    elif re_ranking_strategy_name == 'ditto_re_ranker':
        re_ranking_strategy = DittoSimilarityReRanker(schema_org_class, re_ranking_strategy['model_name'],
                                                      re_ranking_strategy['base_model'],
                                                      re_ranking_strategy['ditto_config'], context_attributes,
                                                      re_ranking_strategy['matcher'],
                                                      max_len=re_ranking_strategy['max_len'])
    elif re_ranking_strategy_name == 'supcon_re_ranker':
        re_ranking_strategy = SupConSimilarityReRanker(schema_org_class, re_ranking_strategy['model_name'],
                                                       re_ranking_strategy['base_model'], context_attributes,
                                                       re_ranking_strategy['matcher'])
    elif re_ranking_strategy_name == 'magellan_re_ranker':
            re_ranking_strategy = MagellanSimilarityReRanker(schema_org_class, re_ranking_strategy['model_name'],
                                                             context_attributes, re_ranking_strategy['matcher'])
    elif re_ranking_strategy_name == 'symbolic_re_ranker':
            re_ranking_strategy = SymbolicSimilarityReRanker(schema_org_class, re_ranking_strategy['similarity_measure'],
                                                             context_attributes, re_ranking_strategy['matcher'], re_ranking_strategy['threshold'])
    elif re_ranking_strategy_name == 'threshold_re_ranker':
            re_ranking_strategy = ThresholdReRanker(schema_org_class, context_attributes,
                                                    re_ranking_strategy['matcher'], re_ranking_strategy['threshold'])

    elif re_ranking_strategy_name is None:
        # Do not supply a re-ranking strategy
        re_ranking_strategy = None
    else:
        # Fall back to default reranker, which does no re-ranking
        re_ranking_strategy = SimilarityReRanker(schema_org_class, 'Dummy Re-Ranker')

    return re_ranking_strategy
