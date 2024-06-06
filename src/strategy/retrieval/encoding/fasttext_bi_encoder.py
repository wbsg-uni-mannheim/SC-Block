import json
from collections import Counter

import numpy as np
import random
import torch
import torch.nn.functional as F
import fasttext
import spacy

from src.strategy.retrieval.encoding.bi_encoder import BiEncoder

class FastTextBiEncoder(BiEncoder):
    def __init__(self, model_path, schema_org_class):
        """Initialize Entity Biencoder"""
        super().__init__(schema_org_class)

        # Make results reproducible
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        FASTTEXT_EMBEDDIG_PATH = model_path

        self.tokenizer = spacy.load('en_core_web_sm')

        self.word_embedding_model = fasttext.load_model(FASTTEXT_EMBEDDIG_PATH)

    def encode_entities(self, entities, excluded_attributes=None, without_special_tokens_and_attribute_names=False):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes, without_special_tokens_and_attribute_names)
                       for entity in entities]

        # Tokenize and embed the entities
        entity_strs_tokenized = [self.tokenizer(entity_str) for entity_str in entity_strs]
        embeddings = []
        for entity_str_tokenized in entity_strs_tokenized:
            # Embed the tokens and average them
            embedding = np.mean(np.array([self.word_embedding_model.get_word_vector(token.text) for token in entity_str_tokenized]), axis=0)
            embedding = torch.tensor(embedding, dtype=torch.float32)
            embeddings.append(embedding)

        embeddings = torch.stack(embeddings)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.tolist()

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None, without_special_tokens_and_attribute_names=False):
        return self.encode_entities(entities, excluded_attributes, without_special_tokens_and_attribute_names)


