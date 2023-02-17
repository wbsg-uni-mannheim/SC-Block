import json
import os
import pickle
from collections import Counter

import fasttext
import numpy as np
import random
import torch
import torch.nn.functional as F

from src.finetuning.open_book.deepblocker.configurations import EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE, \
    FASTTEXT_EMBEDDIG_PATH
from src.finetuning.open_book.deepblocker.dl_models import CTTModel, AutoEncoder
from src.finetuning.open_book.deepblocker.tuple_embedding_models import AutoEncoderTupleEmbedding, CTTTupleEmbedding, \
    HybridTupleEmbedding
from src.strategy.open_book.retrieval.encoding.bi_encoder import BiEncoder

class DLBlockBiEncoder(BiEncoder):
    def __init__(self, model_path, base_model, schema_org_class):
        """Initialize Entity Biencoder"""
        super().__init__(schema_org_class)

        # Make results reproducible
        seed = 42
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        input_dimension = EMB_DIMENSION_SIZE
        hidden_dimensions = (2 * AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE)
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.base_model == 'CTT':
            self.model = CTTTupleEmbedding(hidden_dimensions)
            self.model.ctt_model = CTTModel(input_dimension, hidden_dimensions)
            self.model.ctt_model.load_state_dict(torch.load(model_path))
            self.model.ctt_model.eval()
            self.model.ctt_model.to(self.device)

        elif self.base_model == 'AUTO':
            self.model = AutoEncoderTupleEmbedding(hidden_dimensions)
            self.model.autoencoder_model = AutoEncoder(input_dimension, hidden_dimensions)
            self.model.autoencoder_model.load_state_dict(torch.load(model_path))
            self.model.autoencoder_model.to(self.device)
            self.model.autoencoder_model.eval()

        elif self.base_model == 'HYBRID':
            self.model = HybridTupleEmbedding(hidden_dimensions)
            self.model.ctt_model = CTTModel(input_dimension, hidden_dimensions)
            self.model.autoencoder_model = AutoEncoder(input_dimension, hidden_dimensions)


            self.model.ctt_model.load_state_dict(torch.load(model_path))
            self.model.ctt_model.to(self.device)
            self.model.ctt_model.eval()

            auto_model_path = model_path.replace('HYBRID', 'AUTO')
            self.model.autoencoder_model.load_state_dict(torch.load(auto_model_path))
            self.model.autoencoder_model.to(self.device)
            self.model.autoencoder_model.eval()

        sif_path = model_path.replace('model', 'sif').replace('.bin', '.json').replace('-50-256', '')
        print(sif_path)
        with open(sif_path) as handle:
            self.model.sif_embedding_model.word_to_frequencies = Counter(json.load(handle))
            #print(self.model.sif_embedding_model.word_to_frequencies)
            self.model.sif_embedding_model.calculate_token_statistics()


    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]
        # embeddings = []
        # for entity_str in entity_strs:
        #     print(entity_str)
        #     embedding = self.model.get_tuple_embedding([entity_str])
        #     embeddings.append(embedding)
        # embeddings = torch.stack(embeddings)
        embeddings = self.model.get_tuple_embedding(entity_strs)

        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings.tolist()

    def encode_entities_and_return_pooled_outputs(self, entities, excluded_attributes=None):
        return self.encode_entities(entities, excluded_attributes)


