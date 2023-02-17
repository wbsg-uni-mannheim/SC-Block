import itertools
import logging
import time

from tqdm import tqdm

from src.model.result import Result
from src.preprocessing.value_normalizer import get_datatype, normalize_value
from src.similarity.coordinate import haversine
from src.strategy.open_book.entity_serialization import EntitySerializer


def evaluate_query_table(query_table, experiment_type, retrieval_strategy, similarity_reranker, source_reranker,
                         k_interval, voting='weighted', collect_result_context=False, split=None):
    """
    Calculate mean precision and recall for list of provided evidences
    :param  Querytable query_table: Query Table
    :param  experiment_type String: type of experiments (retrieval/ augmentation)
    :param  RetrievalStrategy strategy_obj: Retrieval strategy
    :param  list[integer]  k_interval: Interval at which the retrieved list of evidences is evaluated
    :param  String   voting: voting strategy- simple (value majority) or weighted (similarity scores)
    """
    logger = logging.getLogger()
    logger.info('Evaluate query table {}: {}'.format(query_table.identifier, query_table.assembling_strategy))

    # ranking_lvls = ['3 - Correct Value and Entity', '3,2 - Relevant Value and Correct Entity', '3,2,1 - Correct Entity']
    ranking_lvls = ['3,2,1 - Correct Entity']   # Fix ranking lvl for now! (Reduce number of combinations)
    results = []
    splits = ['train', 'valid', 'test', None]   # None results in all evidences being analysed
    seen_values = ['seen', 'left_seen', 'right_seen', 'unseen', 'all']

    # Aggregate different scores of evidences to final similarity score
    for evidence in query_table.retrieved_evidences:
        evidence.aggregate_scores_to_similarity_score()

    entity_serializer = EntitySerializer(query_table.schema_org_class, None)

    for ranking_lvl in ranking_lvls:

        for split, seen in itertools.product(splits, seen_values):
            if split is None and seen in ['both_seen', 'left_seen', 'right_seen', 'none_seen']:
                # Skip these two scenarios for now, because it is difficult to interpret the respective results.
                continue

            result = Result(query_table, retrieval_strategy, similarity_reranker, source_reranker, k_interval,
                            ranking_lvl, voting, split, seen)
            if not query_table.has_verified_evidences():
                logger.warning('No verified evidences found for query table {}!'.format(query_table.identifier))
                return results

            if experiment_type == 'augmentation':
                if ranking_lvl == '3 - Correct Value and Entity':
                    relevance_classification = [3]
                elif ranking_lvl == '3,2 - Relevant Value and Correct Entity':
                    relevance_classification = [3, 2]
                else:
                    relevance_classification = [3, 2, 1]

                positive_evidences = [evidence for evidence in query_table.verified_evidences
                                     if evidence.scale in relevance_classification]
                negative_evidences = [evidence for evidence in query_table.verified_evidences
                                     if evidence.scale not in relevance_classification]

                # Filter evidences - Remove ground truth tables
                positive_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(positive_evidences)
                negative_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(negative_evidences)
            else:
                positive_evidences = [evidence for evidence in query_table.verified_evidences
                                      if evidence.signal]
                negative_evidences = [evidence for evidence in query_table.verified_evidences
                                      if not evidence.signal]

                # Filter by split
                if split in ['train', 'valid', 'test']:
                    positive_evidences = [evidence for evidence in positive_evidences
                                          if evidence.split == split]
                    negative_evidences = [evidence for evidence in negative_evidences if evidence.split == split]

                # Filter by seen/unseen/all
                if seen in ['both_seen', 'left_seen', 'right_seen', 'none_seen']:
                    positive_evidences = [evidence for evidence in positive_evidences
                                          if evidence.seen_training == seen]
                    negative_evidences = [evidence for evidence in negative_evidences
                                          if evidence.seen_training == seen]

                # Filter evidences - Remove ground truth tables
                positive_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(positive_evidences)
                negative_evidences = retrieval_strategy.filter_evidences_by_ground_truth_tables(negative_evidences)

            for row in tqdm(query_table.table):

                # positive_evidences_per_row = [evidence for evidence in query_table.verified_evidences
                #                                 if evidence.signal and evidence.split == split]
                # positive_evidences_per_row = [evidence for evidence in query_table.verified_evidences
                #                                 if not evidence.signal and evidence.split == split]

                all_retrieved_evidences = [evidence for evidence in query_table.retrieved_evidences if
                                               evidence.entity_id == row['entityId']]

                if split in ['train', 'valid', 'test']:
                    all_retrieved_evidences = [evidence for evidence in all_retrieved_evidences if
                                               (evidence in positive_evidences) or (evidence in negative_evidences)]

                all_retrieved_evidences.sort(key=lambda evidence: evidence.similarity_score, reverse=True)

                # if logger.level == logging.DEBUG:
                #     logger.debug(' ')
                #     logger.debug(query_table.identifier)
                #     logger.debug(row['entityId'])
                #     for evidence in all_retrieved_evidences:
                #         logger.debug(evidence.table)
                #         logger.debug(evidence.row_id)
                #         logger.debug(evidence.similarity_score)

                if query_table.type == 'augmentation':
                    result.target_values[row['entityId']] = row[query_table.target_attribute]

                no_verified_evidences = sum(
                    [1 for evidence in positive_evidences if evidence.entity_id == row['entityId']])

                scores_per_k = [calculate_aggregated_evidence_scores(all_retrieved_evidences, positive_evidences, k)
                                for k in k_interval]

                for k, rel_retrieved_evidences, no_retrieved_evidences, no_retrieved_verified_evidences in scores_per_k:
                    if k == 1 and experiment_type == 'augmentation' and voting == 'weighted':
                        continue

                    # Calculate precision at k
                    precision = 0
                    # if no_retrieved_evidences > 0:
                    #     precision = no_retrieved_verified_evidences / no_retrieved_evidences
                    result.precision_per_entity[k][row['entityId']] = precision

                    # Calculate recall at k
                    recall = 0
                    # if no_verified_evidences > 0:
                    #     recall = no_retrieved_verified_evidences / min(no_verified_evidences, k)

                    result.recall_per_entity[k][row['entityId']] = recall

                    f1 = 0
                    # if (precision + recall) > 0:
                    #     f1 = (2 * precision * recall) / (precision + recall)
                    result.f1_per_entity[k][row['entityId']] = f1

                    # Calculate not annotated
                    no_not_annotated = 0
                    # if no_retrieved_evidences > 0:
                    #     no_not_annotated = sum([1 for evidence in rel_retrieved_evidences
                    #                             if evidence not in positive_evidences
                    #                             and evidence not in negative_evidences]) / no_retrieved_evidences

                    result.serialization[row['entityId']] = entity_serializer.convert_to_str_representation(row)
                    result.no_retrieved_verified_evidences[k][row['entityId']] = no_retrieved_verified_evidences
                    result.no_retrieved_evidences[k][row['entityId']] = no_retrieved_evidences
                    result.no_verified_evidences[k][row['entityId']] = no_verified_evidences
                    result.not_annotated_per_entity[k][row['entityId']] = no_not_annotated

                    no_retrieved_corner_case_evidences = 0
                    result_evidences = []
                    # Exclude Corner cases for now
                    for retrieved_evidence in rel_retrieved_evidences[:k]:
                        # for verified_evidence in [evidence for evidence in positive_evidences if evidence.entity_id == row['entityId']]:
                        #     if retrieved_evidence == verified_evidence:
                        #         if verified_evidence.corner_case:
                        #             no_retrieved_corner_case_evidences += 1
                        #         break

                        if retrieved_evidence.context is not None and collect_result_context:
                            # Add evidence information to context
                            result_evidence = retrieved_evidence.context.copy()
                            result_evidence['similarity_score'] = retrieved_evidence.similarity_score
                            result_evidence['relevant_evidence'] = True if retrieved_evidence in positive_evidences else False
                            result_evidence['table'] = retrieved_evidence.table
                            result_evidence['row_id'] = retrieved_evidence.row_id
                            result_evidences.append(result_evidence)
                    #
                    # found_positive_evidences = [evidence for evidence in positive_evidences
                    #                             if evidence.entity_id == row['entityId']]
                    found_positive_evidences = []

                    if len(found_positive_evidences) > 0:
                        result.seen_training[k][row['entityId']] = found_positive_evidences[0].seen_training
                    else:
                        result.seen_training[k][row['entityId']] = None

                    # Count number of corner cases
                    # result.corner_cases[k][row['entityId']] = sum([1 for evidence in positive_evidences
                    #                                                if evidence.entity_id == row['entityId']
                    #                                                and evidence.corner_case])
                    result.corner_cases[k][row['entityId']] = 0
                    result.retrieved_corner_cases[k][row['entityId']] = no_retrieved_corner_case_evidences

                    result.different_evidences[k][row['entityId']] = result_evidences
                    result.different_tables[k][row['entityId']] = None

                    # Commented to speed up evaluation
                    # result.different_tables[k][row['entityId']] = \
                    #     list(set([evidence.table for evidence in rel_retrieved_evidences][:k]))

                    if experiment_type == 'augmentation':
                        values = []
                        similarities = []

                        for evidence in rel_retrieved_evidences[:k]:
                            # Exclude evidence value from augmentation if it is None
                            if evidence.value is not None:
                                if type(evidence.value) is str:
                                    value = evidence.value
                                elif type(evidence.value) is list:
                                    value = ', '.join([value for value in evidence.value if value is not None
                                                       and type(value) not in [dict, list]])
                                else:
                                    value = str(evidence.value)

                                values.append(value)
                                similarities.append(evidence.similarity_score)

                        # simple voting vs. weighted voting - To-Do: Separate fusion from evaluation!
                        if voting == 'simple':
                            value_counts = [(value, values.count(value)) for value in set(values)]
                        elif voting == 'weighted':
                            dict_value_similarity = {}
                            total_similarity = 0
                            initial_value_counts = { value: values.count(value) for value in set(values)}
                            for value, similarity_score in zip(values, similarities):
                                if value not in dict_value_similarity:
                                    dict_value_similarity[value] = 0
                                dict_value_similarity[value] += similarity_score
                                total_similarity += similarity_score

                            # Normalize similarity scores by number of appearances
                            dict_value_norm_similarity = {value: sim/initial_value_counts[value]
                                                           for value, sim in dict_value_similarity.items()}
                            if total_similarity > 0:
                                value_counts = [(value, sim/total_similarity) for value, sim in dict_value_norm_similarity.items()]
                            else:
                                value_counts = [(value, 0) for value, sim in
                                                dict_value_norm_similarity.items()]
                        else:
                            raise ValueError('Unknown voting strategy {}.'.format(voting))

                        dict_value_counts = [{'value': value_count[0], 'count': value_count[1]} for value_count in value_counts]
                        value_counts.sort(key=lambda x: x[1], reverse=True)

                        # Calculate Accuracy
                        accuracy = 0
                        predicted_value = None
                        if len(value_counts) > 0:
                            datatype = get_datatype(query_table.target_attribute)
                            if datatype == 'coordinate':
                                # TO-DO: Replace hack with proper approach to retrieve full coordinates!
                                target_value, predicted_value = determine_full_coordinates(value_counts[0][0],
                                                                                           query_table.target_attribute, row,
                                                                                           rel_retrieved_evidences[:k])
                                accuracy = calculate_accuracy(target_value, predicted_value, datatype)

                            else:
                                target_value = row[query_table.target_attribute]
                                predicted_value = value_counts[0][0]

                                accuracy = calculate_accuracy(target_value, predicted_value, datatype)

                        result.fusion_accuracy[k][row['entityId']] = accuracy
                        result.different_values[k][row['entityId']] = dict_value_counts
                        result.found_values[k][row['entityId']] = len(values)
                        result.predicted_values[k][row['entityId']] = predicted_value

            results.append(result)

    return results


def determine_full_coordinates(predicted_coordinate_part, target_attribute, row, rel_evidences):
    """Determine full coordinates"""
    complementary_attribute = {'latitude': 'longitude', 'longitude': 'latitude'}
    predicted_dict = {target_attribute : predicted_coordinate_part,
                      complementary_attribute[target_attribute] : 0}
    target_dict = {target_attribute: row[target_attribute],
                   complementary_attribute[target_attribute]: row[complementary_attribute[target_attribute]]}

    for evidence in rel_evidences:
        if normalize_value(evidence.context[target_attribute], 'coordinate', None) == predicted_coordinate_part:
            if complementary_attribute[target_attribute] in evidence.context:
                predicted_dict[complementary_attribute[target_attribute]] = \
                    evidence.context[complementary_attribute[target_attribute]]
                break

    # Normalize values
    target_value = (normalize_value(target_dict['longitude'], 'coordinate', None),
                    normalize_value(target_dict['latitude'], 'coordinate', None))
    predicted_value = (normalize_value(predicted_dict['longitude'], 'coordinate', None),
                       normalize_value(predicted_dict['latitude'], 'coordinate', None))

    return target_value, predicted_value


def calculate_accuracy(predicted_value, target_value, data_type):
    """Calculate the accuracy for the two provided values based on the data type"""
    accuracy = 0
    if data_type == 'coordinate':
        try:
            dist = haversine(target_value[0], target_value[1], predicted_value[0], predicted_value[1])
            # Accurarcy == 1 if the distance of the points is 100 m at max
            accuracy = 1 if dist <= 0.1 else 0
        except TypeError as e:
            logger = logging.getLogger()
            logger.warning(e)

    else:
        accuracy = 1 if target_value == predicted_value else 0
    return accuracy


def calculate_aggregated_evidence_scores(all_retrieved_evidences, positive_evidences, k):
    """Calculate aggregated evidence scores"""
    rel_retrieved_evidences = all_retrieved_evidences[:k]
    no_retrieved_evidences = len(rel_retrieved_evidences)

    no_retrieved_verified_evidences = sum([1 for evidence in rel_retrieved_evidences if evidence in positive_evidences])

    return k, rel_retrieved_evidences, no_retrieved_evidences, no_retrieved_verified_evidences
