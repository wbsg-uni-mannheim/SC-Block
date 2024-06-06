import os

import numpy as np
import random
import torch
from sentence_transformers import SentenceTransformer

from src.strategy.retrieval.encoding.bi_encoder import BiEncoder


class SBERTBiEncoder(BiEncoder):
    def __init__(self, model_name, pooling, normalize, schema_org_class):
        """Initialize Entity Biencoder"""
        super().__init__(schema_org_class)

        # Make results reproducible
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.pooling = pooling
        self.normalize = normalize

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # # Initialize tokenizer and model for BERT if necessary
        # if model_name is not None:
        #
        #     model_path = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)
        #     if not os.path.isdir(model_path):
        #         # Try to load model from huggingface - enhance model and save locally
        #         tokenizer = AutoTokenizer.from_pretrained(model_name)
        #         model = AutoModel.from_pretrained(model_name)
        #         # Add special tokens - inspired by Dito - Li et al. 2020
        #         special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        #         num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        #         model.resize_token_embeddings(len(tokenizer))
        #
        #         # Cache model and tokenizer locally --> reduce number of calls to huggingface
        #         model.save_pretrained(model_path)
        #         tokenizer.save_pretrained(model_path)

        self.model = SentenceTransformer(model_name).to(self.device)

    def encode_entity(self, entity, excluded_attributes=None):
        """Encode the provided entity"""
        entity_str = self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)

        inputs = None
        with torch.no_grad():
            outputs = self.model.encode(entity_str, show_progress_bar=False, normalize_embeddings=self.normalize)

        return inputs, outputs

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Introduce batches here!

        inputs = None
        with torch.no_grad():
            outputs = self.model.encode(entity_strs, show_progress_bar=False, normalize_embeddings=self.normalize)

        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        torch.cuda.empty_cache()
        inputs, outputs = self.encode_entities(entity, excluded_attributes)

        # Train SBert models always with poolings, hence no additional pooling is necessary
        return outputs.squeeze().tolist()
