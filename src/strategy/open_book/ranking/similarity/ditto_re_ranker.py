import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.finetuning.open_book.ditto.ditto_light.ditto import DittoModel
from src.finetuning.open_book.ditto.ditto_light.knowledge import ProductDKInjector, GeneralDKInjector
from src.finetuning.open_book.ditto.ditto_light.summarize import Summarizer
from src.finetuning.open_book.ditto.matcher import classify, to_str, set_seed
from src.strategy.open_book.ranking.similarity.similarity_re_ranker import SimilarityReRanker


class DittoSimilarityReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, model_name, base_model, ditto_config, context_attributes=None, matcher=False, max_len=256):
        super().__init__(schema_org_class, 'Ditto Cross Encoder - {} {}'.format(base_model, max_len), context_attributes, matcher)

        self.model_name = model_name
        # Initialize tokenizer and model for ditto model
        if model_name is not None:
            model_path = '{}/ditto/{}/model.pt'.format(os.environ['DATA_DIR'], model_name)

            #self.device = torch.device('cpu') # Keep Ditto model on cpu for now.
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model = DittoModel(device=self.device, lm=base_model)
            self.base_model = base_model
            saved_state = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(saved_state['model'])
            self.threshold = saved_state['threshold']
            self.model = self.model.to(self.device)

            self.ditto_config = ditto_config
            if self.ditto_config['dk_injector'] == 'product':
                self.dk_injector = ProductDKInjector(self.ditto_config, self.ditto_config['dk_injector'])
            else:
                self.dk_injector = GeneralDKInjector(self.ditto_config, self.ditto_config['dk_injector'])

            self.summarizer = None
            if self.ditto_config['summarizer']:
                self.summarizer = Summarizer(self.ditto_config, lm=base_model)

            # threshold_path = '{}/ditto/{}/threshold.txt'.format(os.environ['DATA_DIR'], model_name)
            # with open(threshold_path) as f:
            #     self.threshold = float(f.readline().replace('threshold:', ''))

            self.max_len = max_len

    def predict_matches(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        records1_serial = [self.entity_serializer.convert_to_str_representation(entity1, excluded_attributes1)
                            for entity1 in entities1]
        records2_serial = [self.entity_serializer.convert_to_str_representation(entity2, excluded_attributes2)
                            for entity2 in entities2]

        # Potential to-do: Remove squared brackets - Done by Ditto, too
        records1_serial = [record.replace('[COL]', 'COL').replace('[VAL]', 'VAL').replace('\t', '')
                           for record in records1_serial]
        records2_serial = [record.replace('[COL]', 'COL').replace('[VAL]', 'VAL').replace('\t', '')
                           for record in records2_serial]
        if self.schema_org_class in ['amazon-google', 'dblp-acm_1', 'dblp-googlescholar_1', 'walmart-amazon_1'] \
                or 'wdcproducts' in self.schema_org_class:
            records1_serial = [record.replace('COL name VAL', 'COL title VAL') for record in records1_serial]
            records2_serial = [record.replace('COL name VAL', 'COL title VAL') for record in records2_serial]

        # Apply Domain Knowledge Injection & summarization
        pairs = [to_str(record1, record2, self.summarizer, self.max_len, self.dk_injector) for record1, record2 in zip(records1_serial, records2_serial)]
        #print(pairs)
        # Convert pairs to format expected by ditto
        #pairs = ['\t'.join([record1, record2, '0']) for record1,  record2 in zip(records1_serial, records2_serial)]
        set_seed(123)
        pred, _ = classify(pairs, self.model, lm=self.base_model, max_len=self.max_len) # threshold=self.threshold --> exclude Threshold for testing
        set_seed(42)
        # ds_path = '{}/ditto/{}/prediction.csv'.format(os.environ['DATA_DIR'], self.model_name)
        # with open(ds_path, 'w') as file:
        #     for pair, prediction in zip(pairs, pred):
        #         file.write('{}\t{}\n'.format(pair, prediction))

        return pred

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        left_entities = []
        right_entities = []
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]
            if len(rel_evidences) > 0:
                    left_entities.extend([row] * len(rel_evidences))
                    right_entities.extend([rel_evidence.context for rel_evidence in rel_evidences])

        preds = self.predict_matches(entities1=left_entities, entities2=right_entities)

        i = 0
        # qt_path = '{}/ditto/{}/querytable.csv'.format(os.environ['DATA_DIR'], self.model_name)
        # with open(qt_path, 'w') as file:
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]
            if len(rel_evidences) > 0:
                for evidence in rel_evidences:
                    evidence.scores[self.name] = preds[i]
                    evidence.similarity_score = preds[i]
                    #file.write('{}\t{}\n'.format(self.entity_serializer.convert_to_str_representation(row), self.entity_serializer.convert_to_str_representation(evidence.context)))
                    i += 1

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            print('Number of evidences before matching: {}'.format(len(updated_evidences)))
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]
            print('Number of evidences after matching: {}'.format(len(updated_evidences)))

        return updated_evidences
