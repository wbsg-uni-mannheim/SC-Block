import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.finetuning.open_book.contrastive_pretraining.src.contrastive.models.modeling import \
    ContrastiveClassifierModel
from src.strategy.ranking.similarity.similarity_re_ranker import SimilarityReRanker


def determine_path_to_model(model_name, schema_org_class, context_attributes):
    context_attribute_string = '_'.join(context_attributes)
    path_to_model = '{}/models/open_book/finetuned_cross_encoder-{}-{}-{}'.format(os.environ['DATA_DIR'],
                                                                                  model_name, schema_org_class,
                                                                                  context_attribute_string)
    return path_to_model


class SupConSimilarityReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, model_path, base_model, context_attributes=None, matcher=False, max_length=128):
        super().__init__(schema_org_class, 'SupCon Matcher {}'.format(model_path.split('/')[-2] + '/' + model_path.split('/')[-1]), context_attributes, matcher)

        # Initialize tokenizer and model for BERT if necessary
        if model_path is not None:
            #model_path = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.max_length = max_length
            ##self.device = 'cpu' # Use CPU for now

            # if not os.path.isdir(model_path):
            #     # Try to load model from huggingface - enhance model and save locally
            #     tokenizer = AutoTokenizer.from_pretrained(model_name)
            #     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            #     # Add special tokens - inspired by Dito - Li et al. 2020
            #     special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
            #     num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            #     model.resize_token_embeddings(len(tokenizer))
            #
            #     # Cache model and tokenizer locally --> reduce number of calls to huggingface
            #     model.save_pretrained(model_path)
            #     tokenizer.save_pretrained(model_path)

            self.tokenizer = AutoTokenizer.from_pretrained(base_model, additional_special_tokens=('[COL]', '[VAL]'))
            # Load model from checkpoint for testing purposes --> example: '/checkpoint-90000'

            complete_model_path = '{}/pytorch_model.bin'.format(model_path)
            self.model = ContrastiveClassifierModel(len_tokenizer=len(self.tokenizer), checkpoint_path=complete_model_path, model=base_model).to(self.device)

    def predict_matches(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        records1_serial = [self.entity_serializer.convert_to_str_representation(entity1, excluded_attributes1)
                            for entity1 in entities1]
        #print(records1_serial)
        #print(records1_serial)
        records2_serial = [self.entity_serializer.convert_to_str_representation(entity2, excluded_attributes2)
                            for entity2 in entities2]

        # print('Left:')
        # print(records1_serial)
        #
        # print('Right:')
        # print(records2_serial)

        records_encoded = self.tokenizer(records1_serial, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(
            self.device)

        records2_encoded = self.tokenizer(records2_serial, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(
            self.device)

        records_encoded['input_ids_right'] = records2_encoded['input_ids']
        records_encoded['attention_mask_right'] = records2_encoded['attention_mask']
        records_encoded['labels'] = None


        with torch.no_grad():
            _, logits = self.model(**records_encoded)

        return logits

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]

            if len(rel_evidences) > 0:

                # Create smaller batches of entities
                def batches(lst, chunk_size):
                    for i in range(0, len(lst), chunk_size):
                        yield lst[i:i + chunk_size]

                for evidence_chunk in batches(rel_evidences, min(8, len(rel_evidences))):
                    left_entities = [row] * len(evidence_chunk)
                    right_entities = [rel_evidence.context for rel_evidence in evidence_chunk]
                    logits = self.predict_matches(entities1=left_entities, entities2=right_entities)
                    preds = [0 if pred.item() < 0.5 else 1 for pred in logits]

                    for evidence, pred in zip(evidence_chunk, preds):
                        # Overwrite existing scores
                        #evidence.scores = {self.name: pred[1].item()}
                        evidence.scores[self.name] = pred
                        evidence.similarity_score = pred

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]

        return updated_evidences
