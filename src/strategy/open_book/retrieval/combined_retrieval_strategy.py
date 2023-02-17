
from src.strategy.open_book.retrieval.retrieval_strategy import RetrievalStrategy


class CombinedRetrievalStrategy(RetrievalStrategy):

    def __init__(self, schema_org_class, retrieval_strategy_1, retrieval_strategy_2, clusters):
        super().__init__(schema_org_class, 'combined_retrieval_strategy')

        self.retrieval_strategy_1 = retrieval_strategy_1
        self.retrieval_strategy_2 = retrieval_strategy_2
        self.clusters = clusters

        self.model_name = '{}+{}'.format(self.retrieval_strategy_1.model_name, self.retrieval_strategy_2.model_name)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """Retrieve evidences from multiple retrieval strategies"""

        evidences_1 = self.retrieval_strategy_1.retrieve_evidence(query_table, evidence_count, entity_id)

        evidences_2 = self.retrieval_strategy_2.retrieve_evidence(query_table, evidence_count, entity_id)

        merged_evidences = []
        new_evidence_id = 1
        for row in query_table.table:
            evidences_1_per_entity = [evidence for evidence in evidences_1 if evidence.entity_id == row['entityId']]
            evidences_1_per_entity.sort(key=lambda evidence: evidence.similarity_score, reverse=True)

            evidences_2_per_entity = [evidence for evidence in evidences_2 if evidence.entity_id == row['entityId']]
            evidences_2_per_entity.sort(key=lambda evidence: evidence.similarity_score, reverse=True)

            merged_evidences_per_entity = []

            while True:
                if len(merged_evidences_per_entity) >= evidence_count:
                    break

                if len(evidences_1_per_entity) > 0:
                    next_evidence = evidences_1_per_entity.pop(0)
                    if next_evidence not in merged_evidences_per_entity:
                        #self.logger.info('Added entity!')
                        next_evidence.identifier = new_evidence_id
                        similarity_score = 1 - (len(merged_evidences_per_entity) / evidence_count)
                        next_evidence.scores = {self.name: similarity_score}
                        next_evidence.similarity_score = similarity_score
                        new_evidence_id += 1
                        merged_evidences_per_entity.append(next_evidence)

                if len(merged_evidences_per_entity) >= evidence_count:
                    break

                if len(evidences_2_per_entity) > 0:
                    next_evidence = evidences_2_per_entity.pop(0)
                    if next_evidence not in merged_evidences_per_entity:
                        #self.logger.info('Added neural entity!')
                        next_evidence.identifier = new_evidence_id
                        similarity_score = 1 - (len(merged_evidences_per_entity) / evidence_count)
                        next_evidence.scores = {self.name: similarity_score}
                        next_evidence.similarity_score = similarity_score
                        new_evidence_id += 1
                        merged_evidences_per_entity.append(next_evidence)

                if len(evidences_1_per_entity) == 0 and len(evidences_2_per_entity) == 0:
                    break

            merged_evidences.extend(merged_evidences_per_entity)

        return merged_evidences
