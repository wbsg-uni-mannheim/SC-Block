import itertools
import json
import os

from tqdm import tqdm


def aggregate_results(results, k, execution_times):
    # Collect numbers for precision, recall & F1
    splits = ['train', 'valid', 'test', None]   # None results in all evidences being analysed
    seen_values = ['seen', 'left_seen','right_seen', 'unseen', 'all']

    no_retrieved_verified_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_retrieved_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_verified_evidences = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_corner_cases = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}
    no_retrived_corner_cases = {f'{split}_{seen_value}': 0 for split, seen_value in itertools.product(splits, seen_values)}

    # Make sure that all results come from the same retrieval experiment
    aggregated_result = {'model_name': None, 'pooling': None, 'retrieval_strategy': None, 'similarity_reranker': None,
                         'source_reranker': None, 'voting_strategy': None}

    for split, seen_value in itertools.product(splits, seen_values):

        if split is None and seen_value in ['seen', 'unseen']:
            # Skip these two scenarios for now, because it is difficult to interpret the respective results.
            continue

        subset_results = [result for result in results if result.seen == seen_value and result.split == split]

        for result in tqdm(subset_results):
            # print(result.no_retrieved_verified_evidences)
            no_retrieved_verified_evidences[f'{split}_{seen_value}'] += sum(
                [value for value in result.no_retrieved_verified_evidences[k].values()])
            # no_retrieved_verified_evidences['seen'] += sum(
            #     [value for value, seen_training in
            #      zip(result.no_retrieved_verified_evidences[k].values(), result.seen_training[k].values()) if
            #      seen_training])
            # no_retrieved_verified_evidences['unseen'] += sum(
            #     [value for value, seen_training in
            #      zip(result.no_retrieved_verified_evidences[k].values(), result.seen_training[k].values()) if
            #      not seen_training])

            no_retrieved_evidences[f'{split}_{seen_value}'] += sum([value for value in result.no_retrieved_evidences[k].values()])
            # no_retrieved_evidences['seen'] += sum([value for value, seen_training in
            #                                        zip(result.no_retrieved_evidences[k].values(),
            #                                            result.seen_training[k].values()) if seen_training])
            # no_retrieved_evidences['unseen'] += sum([value for value, seen_training in
            #                                          zip(result.no_retrieved_evidences[k].values(),
            #                                              result.seen_training[k].values()) if not seen_training])

            no_verified_evidences[f'{split}_{seen_value}'] += sum([value for value in result.no_verified_evidences[k].values()])
            # no_verified_evidences['seen'] += sum([value for value, seen_training in
            #                                       zip(result.no_verified_evidences[k].values(),
            #                                           result.seen_training[k].values()) if seen_training])
            # no_verified_evidences['unseen'] += sum([value for value, seen_training in
            #                                         zip(result.no_verified_evidences[k].values(),
            #                                             result.seen_training[k].values()) if not seen_training])

            no_corner_cases[f'{split}_{seen_value}'] += sum([value for value in result.corner_cases[k].values()])
            # no_corner_cases['seen'] += sum([value for value, seen_training in zip(result.corner_cases[k].values(),
            #                                                                       result.seen_training[k].values()) if
            #                                 seen_training])
            # no_corner_cases['unseen'] += sum([value for value, seen_training in zip(result.corner_cases[k].values(),
            #                                                                         result.seen_training[k].values())
            #                                   if not seen_training])

            no_retrived_corner_cases[f'{split}_{seen_value}'] += sum([value for value in result.retrieved_corner_cases[k].values()])
            # no_retrived_corner_cases['seen'] += sum([value for value, seen_training in
            #                                          zip(result.retrieved_corner_cases[k].values(),
            #                                              result.seen_training[k].values()) if seen_training])
            # no_retrived_corner_cases['unseen'] += sum([value for value, seen_training in
            #                                            zip(result.retrieved_corner_cases[k].values(),
            #                                                result.seen_training[k].values()) if not seen_training])

            dict_result = result.__dict__
            for key in aggregated_result:
                if aggregated_result[key] is None:
                    aggregated_result[key] = dict_result[key]
                else:
                    if aggregated_result[key] != dict_result[key]:
                        raise ValueError('Results aggregation is only possible for a single setup. Found values {} and {} '
                                         'for configuration {}'.format(aggregated_result[key], dict_result[key], key))

    # print('Retrieved verified evidences: {}'.format(str(no_retrieved_verified_evidences)))
    # print('Retrieved evidences: {}'.format(str(no_retrieved_evidences)))

    options = [f'{split}_{seen_value}' for split, seen_value in itertools.product(splits, seen_values)]
    # print('No of retrieved verified evidences: {}'.format(no_retrieved_verified_evidences['all_all']))
    # print('No of verified evidences: {}'.format(no_verified_evidences['all_all']))
    for option in options:

        if option in ['None_seen', 'None_unseen']:
            # Skip these two scenarios for now, because it is difficult to interpret the respective results.
            continue

        aggregated_result['no_retrieved_verified_{}'.format(option)] = no_retrieved_verified_evidences[option]
        aggregated_result['no_retrieved_{}'.format(option)] = no_retrieved_evidences[option]
        aggregated_result['no_verified_{}'.format(option)] = no_verified_evidences[option]

        aggregated_result['precision_{}'.format(option)] = no_retrieved_verified_evidences[option] / \
                                                           no_retrieved_evidences[option] \
            if no_retrieved_evidences[option] > 0 else 0

        # print('Verified evidences: {}'.format(str(no_verified_evidences)))
        aggregated_result['recall_{}'.format(option)] = no_retrieved_verified_evidences[option] / no_verified_evidences[
            option] \
            if no_verified_evidences[option] > 0 else 0
        aggregated_result['f1_{}'.format(option)] = 2 * aggregated_result['precision_{}'.format(option)] * \
                                                    aggregated_result['recall_{}'.format(option)] \
                                                    / (aggregated_result['precision_{}'.format(option)] +
                                                       aggregated_result['recall_{}'.format(option)]) \
            if (aggregated_result['precision_{}'.format(option)] + aggregated_result[
            'recall_{}'.format(option)]) > 0 else 0

        aggregated_result['corner_case_recall_{}'.format(option)] = no_retrived_corner_cases[option] / no_corner_cases[
            option] \
            if no_corner_cases[option] > 0 else 0

    # Determine schema_org_class using first result
    aggregated_result['schema_org_class'] = results[0].querytable.schema_org_class
    aggregated_result['k'] = k
    aggregated_result.update(execution_times)

    return aggregated_result



def save_aggregated_result(aggregated_result, file_name):
    path_to_results = '{}/result/{}'.format(os.environ['DATA_DIR'], aggregated_result['schema_org_class'])
    if not os.path.isdir(path_to_results):
        os.makedirs(path_to_results)

    path_to_results = '{}/aggregated_{}'.format(path_to_results, file_name)

    with open(path_to_results, 'a+', encoding='utf-8') as f:
        json.dump(aggregated_result, f)
        f.write('\n')
