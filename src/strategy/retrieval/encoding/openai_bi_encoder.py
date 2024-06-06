import logging

from dotenv import load_dotenv

from src.finetuning.open_book.contrastive_pretraining.src.contrastive.models.modeling import ContrastiveModel
from src.strategy.retrieval.encoding.bi_encoder import BiEncoder

from langchain_openai import OpenAIEmbeddings
class OpenAIBiEncoder(BiEncoder):
    def __init__(self, model_name, dataset):
        """Initialize Entity Biencoder"""
        super().__init__(dataset)
        load_dotenv()
        # text-embedding-3-small - dimenions: 1536
        # text-embedding-3-large - dimenions: 3072
        self.open_ai_embeddings = OpenAIEmbeddings(model=model_name)


    def encode_entities(self, entities, excluded_attributes=None, without_special_tokens_and_attribute_names=False):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes,
                                                                            without_special_tokens_and_attribute_names=without_special_tokens_and_attribute_names)
                       for entity in entities]

        embeddings = self.open_ai_embeddings.embed_documents(entity_strs)

        return embeddings

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None, without_special_tokens_and_attribute_names=False):

        return self.encode_entities(entity, excluded_attributes, without_special_tokens_and_attribute_names=without_special_tokens_and_attribute_names)
