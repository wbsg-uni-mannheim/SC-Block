import os

import torch
from transformers import AutoTokenizer, AutoModel

from src.finetuning.open_book.supervised_contrastive_pretraining.src.contrastive.models.modeling import ContrastiveModel
from src.strategy.open_book.retrieval.encoding.bi_encoder import BiEncoder


class SupConBiEncoder(BiEncoder):
    def __init__(self, model_path, base_model, with_projection, proj, pooling, normalize, schema_org_class, max_length=128):
        """Initialize Entity Biencoder"""
        super().__init__(schema_org_class)

        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model for BERT if necessary
        if model_path is not None:
            #
            #model_path = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)
            #print(model_path)
            # if not os.path.isdir(model_path):
            #     # Try to load model from huggingface - enhance model and save locally
            #     tokenizer = AutoTokenizer.from_pretrained(base_model)
            #     model = AutoModel.from_pretrained(base_model)
            #     # Add special tokens - inspired by Dito - Li et al. 2020
            #     special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
            #     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            #     model.resize_token_embeddings(len(tokenizer))
            #
            #     # Cache model and tokenizer locally --> reduce number of calls to huggingface
            #     model.save_pretrained(model_path)
            #     tokenizer.save_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, additional_special_tokens=('[COL]', '[VAL]'))
            # Note: Normalization + pooling happen in the model
            self.model = ContrastiveModel(len_tokenizer=len(self.tokenizer), model=base_model).to(self.device)
            if self.pooling != 'mean':
                self.model.pool = False
            self.model.load_state_dict(torch.load('{}/pytorch_model.bin'.format(model_path), map_location=torch.device(self.device)), strict=False)

    # def encode_entity(self, entity, excluded_attributes=None):
    #     """Encode the provided entity"""
    #     entity_str = self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
    #
    #     inputs = self.tokenizer(entity_str, return_tensors='pt', padding=True,
    #                             truncation=True, max_length=128).to(self.device)
    #
    #     with torch.no_grad():
    #         outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    #
    #     return inputs, outputs[1]

    def encode_entities(self, entities, excluded_attributes=None):
        """Encode the provided entities"""
        entity_strs = [self.entity_serializer.convert_to_str_representation(entity, excluded_attributes)
                       for entity in entities]

        # Introduce batches here!
        inputs = self.tokenizer(entity_strs, return_tensors='pt', padding=True,
                                truncation=True, max_length=self.max_length).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])


        return inputs, outputs[1]

    def encode_entities_and_return_pooled_outputs(self, entity, excluded_attributes=None):
        torch.cuda.empty_cache()
        inputs, outputs = self.encode_entities(entity, excluded_attributes)

        return outputs.squeeze().tolist()
