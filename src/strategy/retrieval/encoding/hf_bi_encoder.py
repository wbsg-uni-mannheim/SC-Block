import logging
import os

import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModel

import torch.nn.functional as F

from src.strategy.retrieval.encoding.bi_encoder import BiEncoder


def select_pooled_output(inputs, outputs, pooling, normalize):
    hidden_state = outputs[0]  # (bs, seq_len, dim)
    pooled_output = None
    if pooling == 'mean':
        pooled_output = mean_pooling(hidden_state, inputs['attention_mask'])
    elif pooling == 'cls':
        pooled_output = hidden_state[:, 0]  # (bs, dim)
    else:
        logging.getLogger().warning('Pooling {} not defined'.format(pooling))

    if normalize:
        # Necessary for cosine similarity
        pooled_output = F.normalize(pooled_output, p=2, dim=1)

    return pooled_output.squeeze().tolist()


# Mean Pooling - Take attention mask into account for correct averaging - Inspired by s-bert
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HuggingfaceBiEncoder(BiEncoder):
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

        # Initialize tokenizer and model for BERT if necessary
        if model_name is not None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_path = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)
            if not os.path.isdir(model_path):
                # Try to load model from huggingface - enhance model and save locally
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                # Add special tokens - inspired by Dito - Li et al. 2020
                special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
                num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
                model.resize_token_embeddings(len(tokenizer))

                # Cache model and tokenizer locally --> reduce number of calls to huggingface
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)

    # def encode_entity(self, entity, excluded_attributes=None):
    #     """Encode the provided entity"""
    #     entity_str = self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
    #
    #     # Introduce batches here!
    #
    #     inputs = self.tokenizer(entity_str, return_tensors='pt', padding=True,
    #                             truncation=True, max_length=128).to(self.device)
    #
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #
    #     return inputs, outputs

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        inputs = self.tokenizer(entity_strs, return_tensors='pt', padding=True,
                                truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return inputs, outputs

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        inputs, outputs = self.encode_entities(entity, excluded_attributes)

        pooled_output = select_pooled_output(inputs, outputs, self.pooling, self.normalize)

        return pooled_output
