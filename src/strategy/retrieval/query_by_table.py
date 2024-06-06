import json
from itertools import groupby

from src.model.evidence import RetrievalEvidence
from src.strategy.es_helper import index_table, determine_es_index_name
from src.strategy.retrieval.retrieval_strategy import RetrievalStrategy


class QueryByTable(RetrievalStrategy):
    def __init__(self, dataset, clusters, boolean_retrieval=False, row_by_row=False):
        super().__init__(dataset, clusters=clusters)
        self.boolean_retrieval = boolean_retrieval
        self.row_by_row = row_by_row

    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        """Retrieve Evidences from table corpus by table"""

        evidence_id = 1
        evidences = []

        if self.row_by_row:
            # Query for each row separately
            table_result = []
            for row in query_table.table:
                table_result.extend(self.query_elasticsearch(query_table, row['name']))

        else:
            # Concatenate all name values
            name_values = ' '.join([value['name'] for value in query_table.table])
            table_result = self.query_elasticsearch(query_table, name_values)

        # Aggregate table results - Continue here
        table_result.sort(key=lambda k: k['_source']['table_name'])
        groups = [list(g) for _, g in groupby(table_result, lambda k: k['_source']['table_name'])]
        table_scores = [(g[0]['_source']['table_name'], sum([entity['_score'] for entity in g])) for g in groups]
        table_scores.sort(key=lambda k: k[1], reverse=True)

        # Take first result to fill values
        for row in query_table.table:

            if entity_id is not None and entity_id != row['entityId']:
                continue

            hits = []
            for table_score in table_scores:
                table_index_name = table_score[0].lower()

                # Check if index exists and index table if it does not exist
                if not self._es.indices.exists(index=table_index_name):
                    self.logger.info('Index {} does not exist - Will index table'.format(table_index_name))
                    index_table(self.path_to_table_corpus, self.schema_org_class, table_score[0], table_index_name,
                                self._es)

                new_entity_results = self.query_tables_index(row, query_table.target_attribute, evidence_count,
                                                             table_index_name)

                hits.extend(new_entity_results['hits']['hits'])
            hits.sort(key=lambda k: k['_score'], reverse=True)
            hits = hits[:evidence_count]

            for hit in hits:
                #found_value = hit['_source'][query_table.target_attribute]

                rowId = hit['_source']['row_id']
                table_name = hit['_source']['table']
                evidence = RetrievalEvidence(evidence_id, query_table.identifier, row['entityId'],
                                              table_name, rowId, hit['_source'])

                evidences.append(evidence)
                self.logger.debug('Added evidence {} to query table'.format(evidence_id))
                evidence_id += 1

        return evidences

    def query_elasticsearch(self, query_table, query_content):

        if self.boolean_retrieval:
            query_body_table = {
                'size': 10,
                'query':
                    {'bool':
                        {'should': [
                            {'match': {'content': {'query': query_content}}},
                            {'match': {'schema': {'query': query_table.target_attribute}}}
                        ]
                        }
                    }
            }
        else:
            query_body_table = {
                'size': 10,
                'query':
                    {'bool':
                        {'should': [
                            {'match': {'boolean_content': {'query': query_content}}},
                            {'match': {'schema': {'query': query_table.target_attribute}}}
                        ]
                        }
                    }
            }
        table_index_name = determine_es_index_name(self.schema_org_class, table=True)
        return self._es.search(body=json.dumps(query_body_table), index=table_index_name)['hits']['hits']

