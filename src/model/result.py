import json
import os


class Result:

    def __init__(self, querytable, retrieval_strategy, similarity_reranker, source_reranker, k_interval, type, voting, split, seen):
        self.querytable = querytable

        self.retrieval_strategy = retrieval_strategy.name
        self.model_name = retrieval_strategy.model_name if hasattr(retrieval_strategy, 'model_name') else None
        self.pooling = retrieval_strategy.pooling if hasattr(retrieval_strategy, 'pooling') else None
        self.similarity = retrieval_strategy.similarity if hasattr(retrieval_strategy, 'similarity') else None
        self.similarity_reranker = similarity_reranker.name if similarity_reranker is not None else None
        self.source_reranker = source_reranker.name if source_reranker is not None else None
        self.k_interval = k_interval
        self.ranking_level = type
        self.voting_strategy = voting
        self.split = split
        self.seen = seen
        self.precision_per_entity = {}
        self.recall_per_entity = {}
        self.f1_per_entity = {}
        self.not_annotated_per_entity = {}
        self.fusion_accuracy = {}
        self.no_retrieved_verified_evidences = {}
        self.no_retrieved_evidences = {}
        self.no_verified_evidences = {}
        self.seen_training = {}
        self.corner_cases = {}
        self.retrieved_corner_cases = {}
        self.serialization = {}

        self.different_values = {}
        self.different_evidences = {}
        self.different_tables = {}
        self.found_values = {}
        self.predicted_values = {}

        self.target_values = {}

        for k in k_interval:
            self.precision_per_entity[k] = {}
            self.recall_per_entity[k] = {}
            self.f1_per_entity[k] = {}
            self.not_annotated_per_entity[k] = {}
            self.fusion_accuracy[k] = {}
            self.no_retrieved_verified_evidences[k] = {}
            self.no_retrieved_evidences[k] = {}
            self.no_verified_evidences[k] = {}
            self.seen_training[k] = {}
            self.corner_cases[k] = {}
            self.retrieved_corner_cases[k] = {}

            self.different_values[k] = {}
            self.different_evidences[k] = {}
            self.different_tables[k] = {}
            self.found_values[k] = {}
            self.predicted_values[k] = {}

    def save_result(self, file_name, with_evidences=False):
        path_to_results = '{}/result/{}'.format(os.environ['DATA_DIR'], self.querytable.schema_org_class)
        if not os.path.isdir(path_to_results):
            os.makedirs(path_to_results)

        path_to_results = '{}/{}'.format(path_to_results, file_name)

        with open(path_to_results, 'a+', encoding='utf-8') as f:
            unpacked_results = self.unpack(with_evidences)
            for unpacked_result in unpacked_results:
                json.dump(unpacked_result, f)
                f.write('\n')

    def unpack(self, with_evidences=False):
        results = []
        if self.querytable.type == 'retrieval':
            template_row = {'ranking_level': self.ranking_level, 'querytableId': self.querytable.identifier,
                            'schemaOrgClass': self.querytable.schema_org_class, 'gt_table': self.querytable.gt_table,
                            'retrieval_strategy': self.retrieval_strategy, 'model_name': self.model_name,
                            'pooling': self.pooling, 'similarity': self.similarity, 'split': self.split,
                            'seen': self.seen, 'similarity_reranker': self.similarity_reranker,
                            'source_reranker': self.source_reranker,
                            'assemblingStrategy': self.querytable.assembling_strategy,
                            'contextAttributes': ', '.join(self.querytable.context_attributes)}

        elif self.querytable.type == 'augmentation':
            template_row = {'ranking_level': self.ranking_level, 'querytableId': self.querytable.identifier,
                            'schemaOrgClass': self.querytable.schema_org_class, 'gt_table': self.querytable.gt_table,
                            'retrieval_strategy': self.retrieval_strategy, 'model_name': self.model_name,
                            'pooling': self.pooling, 'similarity': self.similarity, 'split': self.split,
                            'seen': self.seen,
                            'similarity_reranker': self.similarity_reranker, 'source_reranker': self.source_reranker,
                            'useCase': self.querytable.use_case, 'assemblingStrategy': self.querytable.assembling_strategy,
                            'targetAttribute': self.querytable.target_attribute, 'voting_strategy': self.voting_strategy,
                            'contextAttributes': ', '.join(self.querytable.context_attributes)}
        else:
            raise ValueError('Query Table Type {} is not defined!'.format(self.querytable.type))

        # One hot encoding of 11 requirements --> Change if number of requirements changes
        #for i in range(11):
        #    template_row['RQ'+ str(i+1)] = i + 1 in self.querytable.requirements

        for k in self.k_interval:
            k_row = template_row.copy()
            k_row['k'] = k
            for entity_id in self.f1_per_entity[k].keys():
                row = k_row.copy()
                row['entityId'] = entity_id
                row['serialization'] = self.serialization[entity_id]

                row['precision'] = self.precision_per_entity[k][entity_id]
                row['recall'] = self.recall_per_entity[k][entity_id]
                row['f1'] = self.f1_per_entity[k][entity_id]
                row['not_annotated'] = self.not_annotated_per_entity[k][entity_id]
                row['retrieved_verified_evidences'] = self.no_retrieved_verified_evidences[k][entity_id]
                row['retrieved_evidences'] = self.no_retrieved_evidences[k][entity_id]
                row['seen_training'] = self.seen_training[k][entity_id]
                row['corner_cases'] = self.corner_cases[k][entity_id]
                row['retrieved_corner_cases'] = self.retrieved_corner_cases[k][entity_id]
                row['verified_evidences'] = self.no_verified_evidences[k][entity_id]
                row['different_tables'] = self.different_tables[k][entity_id]

                if with_evidences:
                    row['different_evidences'] = self.different_evidences[k][entity_id]
                    #print('Found different evidences')

                if self.querytable.type == 'augmentation':
                    row['fusion_accuracy'] = self.fusion_accuracy[k][entity_id]
                    row['different_values'] = self.different_values[k][entity_id]
                    row['found_values'] = self.found_values[k][entity_id]
                    row['target_value'] = self.target_values[entity_id]
                    row['predicted_value'] = self.predicted_values[k][entity_id]

                evidence_statistics = self.querytable.calculate_evidence_statistics_of_row(entity_id)
                row['evidences'] = evidence_statistics[0]
                row['correct_entity'] = evidence_statistics[3]
                row['not_correct_entity'] = evidence_statistics[4]

                if self.querytable.type == 'augmentation':
                    row['correct_value_entity'] = evidence_statistics[1]
                    row['rel_value_entity'] = evidence_statistics[2]

                results.append(row)

        return results
