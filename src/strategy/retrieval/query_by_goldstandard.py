
from src.strategy.es_helper import determine_es_index_name
from src.strategy.retrieval.retrieval_strategy import RetrievalStrategy


class QueryByGoldStandard(RetrievalStrategy):

    def __init__(self, dataset, clusters):
        name = 'query_by_goldstandard'
        super().__init__(dataset, name, clusters=clusters)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """Just return the verified evidences of the query table"""

        # Early on filter by ground truth tables to retrieve as many evidences as possible
        evidences = []
        verified_evidences = self.filter_evidences_by_ground_truth_tables(query_table.verified_evidences)

        for row in query_table.table:
            new_evidences = [evidence for evidence in verified_evidences if evidence.entity_id == row['entityId'] and evidence.signal]
            evidences.extend(new_evidences)
            #evidences.extend(new_evidences[:evidence_count])

        #Retrieve contexts of verified evidences
        index_name = determine_es_index_name(self.schema_org_class, clusters=self.clusters)
        #verified_evidence_ids = [evidence.identifier for evidence in query_table.verified_evidences]
        #retrieval_result = self.query_tables_index_by_id(verified_evidence_ids, index_name)

        for evidence in evidences:
            evidence.context = self.query_tables_index_by_table_row_id(evidence.table, evidence.row_id, index_name)
            # Set similarity to 1, because no explicit retrieval strategy is defined
            if evidence.context is not None:
                evidence.scores[self.name] = 1.0
                evidence.similarity_score = 1.0

        return evidences

    def re_rank_evidences(self, query_table, evidences):
        """No re-ranking is done"""
        return evidences
