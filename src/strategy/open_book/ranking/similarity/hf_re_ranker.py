import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.strategy.open_book.ranking.similarity.similarity_re_ranker import SimilarityReRanker


def determine_path_to_model(model_name, schema_org_class, context_attributes):
    context_attribute_string = '_'.join(context_attributes)
    path_to_model = '{}/models/open_book/finetuned_cross_encoder-{}-{}-{}'.format(os.environ['DATA_DIR'],
                                                                                  model_name, schema_org_class,
                                                                                  context_attribute_string)
    return path_to_model


class HuggingfaceSimilarityReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, model_path, context_attributes=None, matcher=False):
        super().__init__(schema_org_class, 'Huggingface Cross Encoder {}'.format(model_path.split('/')[-2] + '/' + model_path.split('/')[-1]), context_attributes, matcher)

        # # Initialize tokenizer and model for BERT if necessary
        # if model_name is not None:
        #     model_path = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     ##self.device = 'cpu' # Use CPU for now
        #
        #     print(model_path)
        #     if not os.path.isdir(model_path):
        #         # Try to load model from huggingface - enhance model and save locally
        #         tokenizer = AutoTokenizer.from_pretrained(model_name)
        #         model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        #         # Add special tokens - inspired by Dito - Li et al. 2020
        #         special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        #         num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        #         model.resize_token_embeddings(len(tokenizer))
        #
        #         # Cache model and tokenizer locally --> reduce number of calls to huggingface
        #         model.save_pretrained(model_path)
        #         tokenizer.save_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load model from checkpoint for testing purposes --> example: '/checkpoint-90000'
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        num_labels=2).to(self.device)

    def predict_matches(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        entities1_serial = [self.entity_serializer.convert_to_str_representation(entity1, excluded_attributes1)
                            for entity1 in entities1]
        entities2_serial = [self.entity_serializer.convert_to_str_representation(entity2, excluded_attributes2)
                            for entity2 in entities2]

        con_entities_serial = [entity1 + '[SEP]' + entity2 for entity1, entity2 in
                               zip(entities1_serial, entities2_serial)]

        encoded_entities = self.tokenizer(con_entities_serial, return_tensors='pt', padding=True, truncation=True).to(
            self.device)

        with torch.no_grad():
            pred = self.model(**encoded_entities)

        return pred

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]

            if len(rel_evidences) > 0:

                # Create smaller batches of entities
                def batches(lst, chunk_size):
                    for i in range(0, len(lst), chunk_size):
                        yield lst[i:i + chunk_size]

                for evidence_chunk in batches(rel_evidences, 8):
                    left_entities = [row] * len(evidence_chunk)
                    right_entities = [rel_evidence.context for rel_evidence in evidence_chunk]
                    preds = self.predict_matches(entities1=left_entities, entities2=right_entities)

                    #preds = F.softmax(preds.logits, dim=1)
                    preds = np.argmax(preds.logits.cpu(), axis=-1)
                    for evidence, pred in zip(evidence_chunk, preds):
                        # Overwrite existing scores
                        #evidence.scores = {self.name: pred[1].item()}
                        evidence.scores[self.name] = pred.item()
                        evidence.similarity_score = pred.item()

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]

        return updated_evidences
