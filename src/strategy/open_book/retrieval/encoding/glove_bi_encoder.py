import logging

from src.strategy.open_book.retrieval.encoding.bi_encoder import BiEncoder


class GloveBiEncoder(BiEncoder):
    def __init__(self, schema_org_class):
        """Initialize Entity Biencoder"""
        super().__init__(schema_org_class)

        # To-Do Haritha


    def encode_entity(self, entity, excluded_attributes=None):
        """Encode the provided entity"""

        # To-Do Haritha

        return None

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""

        # To-Do Haritha

        return None

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        # To-Do Haritha

        return None
