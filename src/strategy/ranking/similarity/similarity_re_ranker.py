import logging

from src.strategy.entity_serialization import EntitySerializer


class SimilarityReRanker:
    def __init__(self, schema_org_class, name, context_attributes=None, matcher=False):

        self.logger = logging.getLogger()
        self.schema_org_class = schema_org_class
        self.entity_serializer = EntitySerializer(schema_org_class, context_attributes)
        self.matcher = matcher
        self.name = name

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences"""
        logger = logging.getLogger()
        logger.warning('Method not implemented!')

        return evidences
