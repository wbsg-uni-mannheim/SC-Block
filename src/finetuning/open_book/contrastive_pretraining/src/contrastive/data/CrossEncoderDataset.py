import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.strategy.open_book.entity_serialization import EntitySerializer


class CrossEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256,
                 dataset='lspc', aug=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug

        # if self.aug:
        #     self.augmenter = Augmenter(self.aug)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
        else:
            data = pd.read_json(path, lines=True)

        if dataset == 'abt-buy':
            data['brand_left'] = ''
            data['brand_right'] = ''

        if dataset == 'amazon-google':
            data['description_left'] = ''
            data['description_right'] = ''

        data = data.fillna('')

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/computers_valid_{size}.csv')
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'dblp-acm_1':
                validation_ids = pd.read_csv('../../data/interim/dblp-acm_1/dblp-acm_1-valid.csv')
            elif dataset == 'dblp-googlescholar':
                validation_ids = pd.read_csv('../../data/interim/dblp-googlescholar/dblp-googlescholar-valid.csv')
            elif dataset == 'walmart-amazon_1':
                validation_ids = pd.read_csv('../../data/interim/walmart-amazon_1/walmart-amazon_1-valid.csv')
            elif dataset == 'wdcproducts80cc20rnd050un':
                validation_ids = pd.read_csv('../../data/interim/wdcproducts80cc20rnd050un/wdcproducts80cc20rnd050un-valid.csv')
            elif dataset == 'wdcproducts80cc20rnd000un':
                validation_ids = pd.read_csv('../../data/interim/wdcproducts80cc20rnd000un/wdcproducts80cc20rnd000un-valid.csv')

            if self.dataset_type == 'train':
                data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
            else:
                data = data[data['pair_id'].isin(validation_ids['pair_id'])]

        data = data.reset_index(drop=True)

        data = self._prepare_data(data)

        self.data = data

    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        # if self.aug:
        #     example['features_left'] = self.augmenter.apply_aug(example['features_left'])
        #     example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)

    def _prepare_data(self, data):

        # if self.dataset == 'lspc':
        #     data['features_left'] = data.apply(self.serialize_sample_lspc, args=('left',), axis=1)
        #     data['features_right'] = data.apply(self.serialize_sample_lspc, args=('right',), axis=1)
        if self.dataset == 'abt-buy':
            data['features'] = data.apply(self.serialize_sample_abtbuy, axis=1)
            #data['features_right'] = data.apply(self.serialize_sample_abtbuy, args=('right',), axis=1)
        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(self.serialize_sample_amazongoogle, axis=1)
        elif self.dataset == 'dblp-acm_1':
            data['features'] = data.apply(self.serialize_sample_dblpacm, axis=1)
        elif self.dataset == 'dblp-googlescholar':
            data['features'] = data.apply(self.serialize_sample_dblpgooglescholar, axis=1)
        elif self.dataset == 'walmart-amazon_1':
            data['features'] = data.apply(self.serialize_sample_walmartamazon, axis=1)
        elif self.dataset == 'wdcproducts80cc20rnd050un':
            data['features'] = data.apply(self.serialize_sample_wdcproduct, axis=1)
        elif self.dataset == 'wdcproducts80cc20rnd000un':
            data['features'] = data.apply(self.serialize_sample_wdcproduct, axis=1)
            #data['features_right'] = data.apply(self.serialize_sample_amazongoogle, args=('right',), axis=1)

        data = data[['features', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data

    def serialize_sample_lspc(self, sample, side):

        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split(" ")[:5])}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split(" ")[:100])}'.strip()
        string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent_{side}"].split(" ")[:200])}'.strip()

        return string

    def serialize_sample_abtbuy(self, sample):

        entity_serializer = EntitySerializer('abt-buy')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['brand'] = dict_sample['brand_{}'.format(side)]
            dict_sample['name'] = dict_sample['name_{}'.format(side)]
            dict_sample['price'] = dict_sample['price_{}'.format(side)]
            dict_sample['description'] = dict_sample['description_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))

        string = '[SEP]'.join(strings)
        print(string)
        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_amazongoogle(self, sample):

        entity_serializer = EntitySerializer('amazon-google')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['manufacturer'] = dict_sample['manufacturer_{}'.format(side)]
            dict_sample['name'] = dict_sample['title_{}'.format(side)]
            dict_sample['price'] = dict_sample['price_{}'.format(side)]
            dict_sample['description'] = dict_sample['description_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))

        string = '[SEP]'.join(strings)
        # dict_sample = sample.to_dict()
        # dict_sample['manufacturer'] = dict_sample['manufacturer_{}'.format(side)]
        # dict_sample['name'] = dict_sample['title_{}'.format(side)]
        # dict_sample['price'] = dict_sample['price_{}'.format(side)]
        # dict_sample['description'] = dict_sample['description_{}'.format(side)]
        # string = entity_serializer.convert_to_str_representation(dict_sample)

        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_dblpacm(self, sample):

        entity_serializer = EntitySerializer('dblp-acm_1')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['name'] = dict_sample['title_{}'.format(side)]
            dict_sample['authors'] = dict_sample['authors_{}'.format(side)]
            dict_sample['venue'] = dict_sample['venue_{}'.format(side)]
            dict_sample['year'] = dict_sample['year_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))

        string = '[SEP]'.join(strings)
        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_dblpgooglescholar(self, sample):

        entity_serializer = EntitySerializer('dblp-googlescholar_1')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['name'] = dict_sample['title_{}'.format(side)]
            dict_sample['authors'] = dict_sample['authors_{}'.format(side)]
            dict_sample['venue'] = dict_sample['venue_{}'.format(side)]
            dict_sample['year'] = dict_sample['year_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))

        string = '[SEP]'.join(strings)

        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string


    def serialize_sample_walmartamazon(self, sample):

        entity_serializer = EntitySerializer('walmart-amazon_1')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['name'] = dict_sample['title_{}'.format(side)]
            dict_sample['category'] = dict_sample['category_{}'.format(side)]
            dict_sample['brand'] = dict_sample['brand_{}'.format(side)]
            dict_sample['modelno'] = dict_sample['modelno_{}'.format(side)]
            dict_sample['price'] = dict_sample['price_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))

        string = '[SEP]'.join(strings)

        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string


    def serialize_sample_wdcproduct(self, sample):

        entity_serializer = EntitySerializer('wdcproducts')
        strings = []
        for side in ['left', 'right']:
            dict_sample = sample.to_dict()
            dict_sample['name'] = dict_sample['title_{}'.format(side)]
            dict_sample['brand'] = dict_sample['brand_{}'.format(side)]
            dict_sample['description'] = dict_sample['description_{}'.format(side)]
            dict_sample['price'] = dict_sample['price_{}'.format(side)]
            dict_sample['pricecurrency'] = dict_sample['pricecurrency_{}'.format(side)]
            strings.append(entity_serializer.convert_to_str_representation(dict_sample))
        string = '[SEP]'.join(strings)

        # string = ''
        # string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        # string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        # string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        # string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string
