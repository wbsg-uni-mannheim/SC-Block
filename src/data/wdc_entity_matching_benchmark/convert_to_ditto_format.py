import logging
import os
import random

import click
import pandas as pd

from src.strategy.open_book.entity_serialization import EntitySerializer


@click.command()
@click.option('--dataset')
@click.option('--testset')
@click.option('--size', default='large')
def convert_table_to_ditto_format(dataset, testset, size):

    random.seed(42)

    splits = ['train', 'valid', 'test']
    ds_sets = {}
    converted_data_sets = {}
    for split in splits:
        if split == 'test':
            path = '{}/wdc_product_entity_matching_benchmark/80pair/{}{}_{}.json.gz'.format(os.environ['DATA_DIR'], dataset, testset, split)
        else:
            path = '{}/wdc_product_entity_matching_benchmark/80pair/{}000un_{}_{}.json.gz'.format(os.environ['DATA_DIR'], dataset,
                                                                                  split, size)

        ds_sets[split] = pd.read_json(path, lines=True, encoding='utf-8')
        #ds_sets[split] = ds_sets[split].sort_values(by=['label'], ascending=False)

        # Convert records
        sides = ['left', 'right']
        records = []
        for index, row in ds_sets[split].iterrows():
            record = []
            for side in sides:
                record.append(serialize_sample_wdcproduct(row, side).replace('\t', ''))

            record.append(str(row['label']))
            #print(' \t '.join(record))
            records.append(' \t '.join(record))

        # Write records to file
        file_path = 'C:/Users/alebrink/Documents/02_Research/TableAugmentation/table-augmentation-framework/data/ditto/WDCBenchmark/wdcproducts80cc20rnd{}/{}.txt'.format(testset, split)
        with open(file_path, 'w', encoding='utf-8') as file:
            for record in records:
                file.write('{}\n'.format(record))


def serialize_sample_wdcproduct(sample, side):

    entity_serializer = EntitySerializer('wdcproducts')
    dict_sample = sample.to_dict()
    dict_sample['name'] = dict_sample['title_{}'.format(side)]
    dict_sample['brand'] = dict_sample['brand_{}'.format(side)]
    dict_sample['description'] = dict_sample['description_{}'.format(side)]
    dict_sample['price'] = dict_sample['price_{}'.format(side)]
    dict_sample['pricecurrency'] = dict_sample['priceCurrency_{}'.format(side)]
    string = entity_serializer.convert_to_str_representation(dict_sample)

    # Postprocessing for Ditto
    string = string.replace('[COL]', 'COL').replace('[VAL]', 'VAL')

    # string = ''
    # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
    # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
    # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
    # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

    return string


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    convert_table_to_ditto_format()