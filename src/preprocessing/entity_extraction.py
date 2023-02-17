import re

from src.preprocessing.value_normalizer import get_datatype, normalize_value, detect_not_none_value


def extract_entity(raw_entity, schema_org_class):
    """Extract entity from raw json"""
    entity = {}

    # Normalize/ unpack raw_entity if necessary
    for raw_key in raw_entity.keys():
        key = normalize_key(raw_key, schema_org_class)

        if check_key_is_relevant(key, schema_org_class):
            if type(raw_entity[raw_key]) is str and len(raw_entity[raw_key]) > 0 and detect_not_none_value(raw_entity[raw_key]):
                normalized_value = normalize_value(raw_entity[raw_key], get_datatype(key), raw_entity, entity)
                if len(normalized_value) > 0 and detect_not_none_value(normalized_value):
                    entity[key] = normalized_value
            elif type(raw_entity[raw_key]) is dict:
                # First case: property has name sub-property --> lift name
                if 'name' in raw_entity[raw_key] and len(raw_entity[raw_key]['name']) > 0 \
                        and detect_not_none_value(raw_entity[raw_key]['name']):
                    normalized_value = normalize_value(raw_entity[raw_key]['name'], get_datatype(key), raw_entity, entity)
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value):
                        entity[key] = normalized_value
                # Second case: lift all values by sub-property name
                else:
                    for raw_property_key in raw_entity[raw_key].keys():

                        property_key = normalize_key(raw_property_key, schema_org_class)
                        if check_key_is_relevant(property_key, schema_org_class):
                            if len(raw_entity[raw_key][raw_property_key]) > 0 \
                                    and detect_not_none_value(raw_entity[raw_key][raw_property_key]):
                                normalized_value = normalize_value(raw_entity[raw_key][raw_property_key],
                                                                       get_datatype(property_key), raw_entity, entity)
                                if len(normalized_value) > 0 and detect_not_none_value(normalized_value)\
                                        and property_key not in entity:
                                    entity[property_key] = normalized_value

            elif type(raw_entity[raw_key]) is list:
                # Check if element type is str
                if all(type(element) is str for element in raw_entity[raw_key]):
                    normalized_value = ', '.join([normalize_value(element, get_datatype(key), raw_entity, entity)
                                                  for element in raw_entity[raw_key] if len(element) > 0])
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value)\
                            and key not in entity:
                        entity[key] = normalized_value
                # Check if nested object has name attribute
                elif all(type(element) is dict for element in raw_entity[raw_key]) \
                        and all('name' in element for element in raw_entity[raw_key]):
                    normalized_value = ', '.join([normalize_value(element['name'], get_datatype(key), raw_entity, entity)
                                                  for element in raw_entity[raw_key] if len(element['name']) > 0
                                                  and type(element['name']) is str])
                    if len(normalized_value) > 0 and detect_not_none_value(normalized_value) \
                            and key not in entity:
                        entity[key] = normalized_value
            else:
                entity[key] = str(raw_entity[raw_key])

    return entity


def normalize_key(key_value, schema_org):
    """Normalise key value"""
    replace_values = [replace_value for replace_value in ['\\u201d', '%20', '\\u201c']
                      if replace_value in key_value]
    for replace_value in replace_values:
        key_value = key_value.replace(replace_value, '')

    key_value = re.sub("[^0-9a-zA-Z]+", '', key_value)

    if schema_org == 'localbusiness':
        # Superficial schema mappings
        schema_mappings = {'telephon': 'telephone', 'telephonenumber': 'telephone', 'telepone': 'telephone',
                         'tu00e9lu00e9phone': 'telephone', 'streetadress': 'streetaddress', 'street': 'streetaddress',
                         'postcode': 'postcalcode', 'lat': 'latitude', 'long': 'longitude'}

        if key_value in schema_mappings:
            key_value = schema_mappings[key_value]

    return key_value


def check_key_is_relevant(key, schema_org_class):
    """Check if entity key is reasonable for the schema org class"""
    #Check key length > 0
    if schema_org_class == 'movie':
        attributes = ['name', 'director', 'description', 'duration', 'datepublished']
    elif schema_org_class == 'hotel':
        attributes = ['address', 'addresslocality', 'addressregion', 'addresscountry', 'latitude', 'longitude', 'postalcode',
                      'name', 'description', 'streetaddress', 'telephone', 'geo', 'geomidpoint', 'email', 'location', 'brand']
    elif schema_org_class == 'localbusiness':
        attributes = ['address', 'addresslocality', 'addressregion', 'addresscountry', 'latitude', 'longitude', 'postalcode',
                      'name', 'description', 'streetaddress', 'telephone', 'geo', 'geomidpoint', 'email', 'location', 'brand',
                      'openinghours', 'vatid', 'openinghoursspecification', 'opens', 'closes', 'dayofweek', 'taxid',
                      'vatid']
    elif schema_org_class == 'product':
        attributes = ['name', 'brand', 'manufacturer', 'weight', 'model', 'releasedate', 'width', 'height', 'depth',
                      'description', 'color', 'mpn', 'sku', 'gtin14', 'gtin13', 'gtin12', 'gtin8', 'gtin']
    elif schema_org_class == 'abt-buy':
        attributes = ['name', 'description', 'price']
    elif schema_org_class == 'amazon-google':
        attributes = ['name', 'manufacturer', 'price']
    elif schema_org_class in ['dblp-acm_1', 'dblp-acm_2', 'dblp-googlescholar_1', 'dblp-googlescholar_2']:
        attributes = ['name', 'authors', 'venue', 'year']
    elif schema_org_class in ['itunes-amazon_1', 'itunes-amazon_2']:
        attributes = ['name', 'artist_name', 'album_name', 'genre', 'price', 'copyright', 'time', 'released']
    elif schema_org_class in ['walmart-amazon_1', 'walmart-amazon_2']:
        attributes = ['name', 'category', 'brand', 'modelno', 'price']
    elif 'wdcproducts' in schema_org_class:
        attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']
    else:
        raise ValueError('Schema org class {} is not known!'.format(schema_org_class))

    return key in attributes