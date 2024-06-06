import logging

from src.model.evidence import RetrievalEvidence, AugmentationEvidence
from src.strategy.entity_serialization import EntitySerializer
from src.strategy.es_helper import determine_es_index_name
from src.strategy.retrieval.retrieval_strategy import RetrievalStrategy


class QueryByEntity(RetrievalStrategy):

    def __init__(self, dataset, clusters=False, rank_evidences_by_table=False, all_attributes=False, tokenizer=None, switched=False):
        name = 'query_by_entity' if not rank_evidences_by_table else 'query_by_entity_rank_by_table'
        super().__init__(dataset, name, clusters=clusters, switched=switched)
        self.rank_evidences_by_table = rank_evidences_by_table
        self.model_name = 'BM25'

        self.all_attributes = all_attributes
        if self.all_attributes:
            self.entity_serizalizer = EntitySerializer(dataset)
            self.model_name = '{}-all_attributes'.format(self.model_name)

        self.tokenizer = tokenizer
        if self.tokenizer:
            self.model_name = '{}-{}'.format(self.model_name, tokenizer)

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        logger = logging.getLogger()
        evidence_id = 1
        evidences = []

        # Iterate through query table
        for row in query_table.table:

            if entity_id is not None and entity_id != row['entityId'] and not self.rank_evidences_by_table:
                continue
            index_name = determine_es_index_name(self.schema_org_class, clusters=self.clusters, tokenizer=self.tokenizer, switched=self.switched)
            if self.all_attributes:
                entity_str = self.entity_serizalizer.convert_to_str_representation(row)
                entity_result = self.query_tables_index_by_all_attributes(entity_str, evidence_count, index_name)
            else:
                entity_result = self.query_tables_index(row, query_table.context_attributes, evidence_count, index_name)
            logger.info('Found {} results for entity {} of query table {}!'.format(len(entity_result['hits']['hits']),
                                                                                    row['entityId'],
                                                                                    query_table.identifier))

            first_score = None # Normalize BM25 scores with first score
            for hit in entity_result['hits']['hits']:
                if first_score is None:
                    first_score = hit['_score']

                # Deal with cases, in which there is not target attribute value
                found_value = None
                if query_table.type == 'augmentation' and query_table.target_attribute in hit['_source']:
                    found_value = hit['_source'][query_table.target_attribute]

                rowId = hit['_source']['row_id']
                table_name = hit['_source']['table']

                if query_table.type == 'retrieval':
                    evidence = RetrievalEvidence(evidence_id, query_table.identifier, row['entityId'],
                                                 table_name, rowId, hit['_source'])
                elif query_table.type == 'augmentation':
                    evidence = AugmentationEvidence(evidence_id, query_table.identifier, row['entityId'], table_name,
                                                    rowId,  hit['_source'], found_value, query_table.target_attribute)
                else:
                    raise ValueError('Query Table Type {} is not defined!'.format(query_table.type))
                score = hit['_score'] / first_score
                evidence.scores[self.name] = score
                evidence.similarity_score = score

                evidences.append(evidence)
                self.logger.debug('Added evidence {} to query table'.format(evidence_id))
                evidence_id += 1

        if self.rank_evidences_by_table:
            # Rank evidences by most often retrieved table
            tables = [evidence.table for evidence in evidences]
            table_counts = [[table, tables.count(table)] for table in set(tables)]
            table_counts.sort(key=lambda x: x[1])

            collected_evidences = evidences.copy()
            evidences = []
            for table in set(tables):
                evidences.extend([evidence for evidence in collected_evidences if evidence.table == table])

        return evidences
