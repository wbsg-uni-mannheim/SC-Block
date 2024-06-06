import json
import os
# Iterate through all files in the dataset directories
repo_path = '/ceph/alebrink/tableAugmentation/data/result'
#datasets = ['abt-buy', 'amazon-google',  'walmart-amazon_1', 'wdcproducts80cc20rnd050un_block_s_train_l',
#            'wdcproducts80cc20rnd050un_block_m_train_l', 'wdcproducts80cc20rnd050un_block_l_train_l']
datasets = [ 'walmart-amazon_1' ]

for dataset in datasets:
    print(f'Processing {dataset}...')
    print('model_name, k, retrieval_time, recall_test_all, filename')
    dataset_path = os.path.join(repo_path, dataset)
    for root, dirs, files in os.walk(dataset_path):
        # Sort files descending
        files.sort(reverse=True)
        for file in files:
            if file.endswith('.json') and 'aggregated' in file and 'results' in file:
                # Read the file line by line
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        try:
                            json_line = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if 'recall_test_all' in json_line:
                            if 'recall_test_all' in json_line:
                                # Print model_name, k, retrieval_time, recall_test_all & filename
                                print(json_line['model_name'].split('/')[-1], json_line['k'], json_line['retrieval_time'], json_line['recall_test_all'], file, dataset, sep=', ')
                            #else:
                            #    print(json_line['model_name'], json_line['k'], json_line['retrieval_time'],
                            #          json_line['recall_test'], file)
    print('---------------------------------')

