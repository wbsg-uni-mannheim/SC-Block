import logging

from src.strategy.open_book.entity_serialization import EntitySerializer


class BiEncoder:

    def __init__(self, schema_org_class, context_attributes=None):
        self.schema_org_class = schema_org_class
        self.entity_serializer = EntitySerializer(schema_org_class, context_attributes)

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        """Fills the provided query table and delivers evidence if expected
            :param entity entity to be encoded
            :param excluded_attributes   Attributes, which will be excluded
        """

        logger = logging.getLogger()
        logger.warning('Method not implemented!')

        raise NotImplementedError('Method not implemented!')
