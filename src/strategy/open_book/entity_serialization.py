import logging
import math

import numpy as np


def preprocess_attribute_value(entity, attr):
    """Preprocess attribute values"""
    attribute_value = None

    if entity is None:
        raise ValueError('Entity must not be None!')

    if attr in entity and len(str(entity[attr])) > 0 \
            and entity[attr] is not None and entity[attr] is not np.nan:
        if type(entity[attr]) is list and all(type(element) is str for element in entity[attr]):
            attribute_value = ', '.join(entity[attr])
        elif type(entity[attr]) is str:
            attribute_value = entity[attr]
        elif isinstance(entity[attr], np.floating) or type(entity[attr]) is float:
            if not math.isnan(entity[attr]):
                attribute_value = str(entity[attr])
        else:
            attribute_value = str(entity[attr])

    return attribute_value


class EntitySerializer:
    def __init__(self, schema_org_class, context_attributes=None):
        self.schema_org_class = schema_org_class
        logger = logging.getLogger()

        # Select attributes that will be encoded
        if context_attributes is not None:
            self.context_attributes = context_attributes
        else:
            if self.schema_org_class in ['movie', 'movie_daniel']:
                self.context_attributes = ['name', 'director', 'duration', 'datepublished']
            elif self.schema_org_class == 'hotel':
                self.context_attributes = ['address', 'addresslocality', 'addressregion', 'latitude', 'longitude', 'postalcode',
                                            'name', 'streetaddress', 'telephone']
            elif self.schema_org_class == 'localbusiness':
                # self.context_attributes = ['name', 'addresslocality', 'addressregion', 'addresscountry', 'postalcode',
                #                    'streetaddress']
                self.context_attributes = ['name', 'addresslocality']
            elif self.schema_org_class == 'product':
                self.context_attributes = ['name']
            elif self.schema_org_class == 'abt-buy':
                self.context_attributes = ['name', 'price', 'description']
            elif self.schema_org_class == 'amazon-google':
                self.context_attributes = ['manufacturer', 'name', 'price']
            elif schema_org_class in ['dblp-acm_1', 'dblp-acm_2', 'dblp-googlescholar_1', 'dblp-googlescholar_2']:
                self.context_attributes = ['name', 'authors', 'venue', 'year']
            elif schema_org_class in ['itunes-amazon_1', 'itunes-amazon_2']:
                self.context_attributes = ['name', 'artist_name', 'album_name', 'genre']
            elif schema_org_class in ['walmart-amazon_1', 'walmart-amazon_2']:
                self.context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
            elif 'wdcproducts' in schema_org_class:
                self.context_attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
            else:
                raise ValueError('Entity Serialization not defined for schema org class {}'.format(self.schema_org_class))

    def convert_to_str_representation(self, entity, excluded_attributes=None, without_special_tokens=False):
        """Convert to string representation of entity"""
        entity_str = ''
        selected_attributes = self.context_attributes

        if entity is None:
            raise ValueError('Entity must not be None!')

        if excluded_attributes is not None:
            selected_attributes = [attr for attr in self.context_attributes if attr not in excluded_attributes]

        for attr in selected_attributes:
            attribute_value = preprocess_attribute_value(entity, attr)
            if attr == 'description' and attribute_value is not None:
                attribute_value = attribute_value[:100]
            if attribute_value is not None:
                if without_special_tokens:
                    entity_str = '{} {}'.format(entity_str, attribute_value)
                else:
                    entity_str = '{}[COL] {} [VAL] {} '.format(entity_str, attr, attribute_value)
            if attribute_value is None:
                if without_special_tokens:
                    entity_str = '{}'.format(entity_str)
                else:
                    entity_str = '{}[COL] {} [VAL] '.format(entity_str, attr)

        return entity_str

    def convert_to_cross_encoder_representation(self, entity1, entity2):

        entity_str_1 = self.convert_to_str_representation(entity1)
        entity_str_2 = self.convert_to_str_representation(entity2)

        cross_encoder_representation = '{}[SEP]{}'.format(entity_str_1, entity_str_2)
        return cross_encoder_representation

    def project_entity(self, entity, excluded_attributes=None, selected_attributes=None):
        """Project entity to a subset of attributes if the attribute value is not None"""

        if entity is None:
            raise ValueError('Entity must not be None!')

        entity_projection = {}
        focus_attributes = self.context_attributes.copy()

        if excluded_attributes is not None and selected_attributes is not None:
            raise ValueError('Either select or exclude attributes - Do not do both!')
        elif excluded_attributes is not None:
            # Exclude Attributes
            focus_attributes = [attr for attr in focus_attributes if attr not in excluded_attributes]

        elif selected_attributes is not None:
            # Select attributes
            focus_attributes = [attr for attr in focus_attributes if attr in selected_attributes]

        for attr in focus_attributes:
            attribute_value = preprocess_attribute_value(entity, attr)
            if attribute_value is not None:
                entity_projection[attr] = attribute_value

        return entity_projection


    def convert_str_to_entity(self, string_representation):
        """Convert String representation to entity"""
        entity = {}
        string_values = string_representation.split('[COL]')
        for value in string_values:
            if '[VAL]' in value:
                values = value.split('[VAL]')
                entity[values[0].strip()] = values[1].strip()

        return entity
