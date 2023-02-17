import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.strategy.open_book.ranking.similarity.similarity_re_ranker import SimilarityReRanker

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


class SymbolicSimilarityReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, similarity_measure, context_attributes=None, matcher=False, threshold=0.5):
        super().__init__(schema_org_class, 'Symbolic re-ranker', context_attributes, matcher)
        self.similarity_measure = similarity_measure
        self.threshold = threshold

    def similarity(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        entities1_serial = [self.entity_serializer.convert_to_str_representation(entity1, excluded_attributes1, without_special_tokens=True)
                            for entity1 in entities1]
        entities2_serial = [self.entity_serializer.convert_to_str_representation(entity2, excluded_attributes2, without_special_tokens=True)
                            for entity2 in entities2]

        similarity_scores = []
        for entity1_serial, entity2_serial in zip(entities1_serial, entities2_serial):
            if self.similarity_measure == 'jaccard':
                entity1_serial_strings = entity1_serial.lower().split(' ')
                entity2_serial_strings = entity2_serial.lower().split(' ')
                similarity_score = jaccard_similarity(entity1_serial_strings, entity2_serial_strings)
                similarity_scores.append(similarity_score)
            else:
                ValueError('Similarity Measure {} is unknown!'.format(self.similarity_measure))

        return similarity_scores

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
                    similarity_scores = self.similarity(entities1=left_entities, entities2=right_entities)

                    for evidence, similarity_score in zip(evidence_chunk, similarity_scores):
                        # Overwrite existing scores
                        #evidence.scores = {self.name: pred[1].item()}
                        evidence.scores[self.name] = similarity_score
                        # Aggregate similarity scores to combine retrieval score and re-ranking score
                        evidence.aggregate_scores_to_similarity_score()


        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if
                                 evidence.similarity_score > self.threshold]

        return updated_evidences
