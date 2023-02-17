import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.strategy.open_book.ranking.similarity.similarity_re_ranker import SimilarityReRanker

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


class ThresholdReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, context_attributes=None, matcher=False, threshold=0.5):
        super().__init__(schema_org_class, 'Threshold re-ranker', context_attributes, matcher)
        self.threshold = threshold


    def re_rank_evidences(self, query_table, evidences):
        """Filter evidences based on threshold"""
        for evidence in evidences:
            evidence.aggregate_scores_to_similarity_score()

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        print(len(updated_evidences))
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if
                                 evidence.similarity_score > self.threshold]
            print(len(updated_evidences))

        return updated_evidences
