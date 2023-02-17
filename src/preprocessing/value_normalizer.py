import logging
import re

import phonenumbers
from dateutil.parser import parse
from datetime import datetime, timedelta
import country_converter as coco

from phonenumbers import NumberParseException


def get_datatype(attribute):
    """Get Data Type of provided attribute if possible"""
    attr_2_datatype = {'datepublished': 'date',
                       'duration': 'duration',
                       'latitude': 'coordinate',
                       'longitude': 'coordinate',
                       'telephone': 'telephone',
                       'addresscountry': 'country'}

    if attribute in attr_2_datatype:
        return attr_2_datatype[attribute]
    else:
        return 'string'


def detect_not_none_value(value):
    """Check if the provided value does not correspond to a none value"""
    return str(value).lower() not in ['none', '-', '--', ' ', 'tbd', 'tba', 'n/a', 'na', '?', 'null', '#', '.', ',']


def normalize_value(value, datatype, raw_entity=None, entity=None):
    """Entrance method for value normalisation -
    To-do: Error message when parsing was not successful (?)"""
    logger = logging.getLogger()
    final_value = value
    if datatype == 'string' or type(value) is not str:
        logger.debug('No manipulation of data type string.')
    elif datatype == 'date':
        try:
            d = parse(value, yearfirst=True, default=datetime(1900, 1, 1))
            final_value = d.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug(e)

    elif datatype == 'telephone':
        country = None
        if entity is not None \
                and 'addresscountry' in entity \
                and type(entity['addresscountry']) is str:
            country = entity['addresscountry']

        elif raw_entity is not None and 'address' in raw_entity \
                and raw_entity['address'] is not None and 'addresscountry' in raw_entity['address'] \
                and type(raw_entity['address']['addresscountry']) is str:
            country = normalize_value(raw_entity['address']['addresscountry'], 'country', None)
        try:
            phone = phonenumbers.parse(value, country)
            final_value = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)
        except NumberParseException as e:
            # Remove all non numeric characters
            value = re.sub('[^0-9]', '', value)
            try:
                phone = phonenumbers.parse(value, None)
                final_value = phonenumbers.format_number(phone, phonenumbers.PhoneNumberFormat.E164)
            except NumberParseException as e:
                logger.debug(e)
                # Check if all chars are 0
                if len(value) > 0 and value[0] == '0' and value == len(value) * value[0]:
                    final_value = ''
                else:
                    final_value = value

    elif datatype == 'duration':
        try:
            d = parse_timedelta(value)
            time_dict = {'H': int(d.seconds / 3600), 'M': int((d.seconds % 3600) / 60),
                         'S': int((d.seconds % 3600) % 60)}
            strftduration = 'PT'
            for key, value in time_dict.items():
                if value > 0:
                    strftduration = '{}{}{}'.format(strftduration, value, key)

            final_value = strftduration
        except Exception as e:
            logger.debug(e)

    elif datatype == 'coordinate':
        value = value.strip().replace('\"', '').replace('\\', '')
        try:
            final_value = parse_coordinate(value)
        except ValueError as e:
            logger.debug(e)

    elif datatype == 'country':
        try:
            coco_logger = coco.logging.getLogger()
            if coco_logger.level != logging.CRITICAL:
                coco_logger.setLevel(logging.CRITICAL)
            final_value = coco.convert(names=[value], to='ISO2', not_found=None)
        except Exception as e:
            logger.debug(e)
            final_value = value

    else:
        raise ValueError('Normalization of datatype {} is not implemented!'.format(datatype))

    return str(final_value)


def parse_timedelta(value):
    """Parse Duration into time delta object"""
    value = str(value).replace(' ', '')
    regex_patterns = [
        r'P?T?((?P<hours>\d+?)(hr|h|H))?((?P<minutes>\d+?)(m|M|min|Min|phút|мин|分钟|perc|dakika))?((?P<seconds>\d+?)(s|S))?',
        r'P?T?((H)(?P<hours>\d+))?((M)(?P<minutes>\d+))?',
        r'(?P<hours>\d+):(?P<minutes>\d+)',
        r'^(?P<minutes>\d+)$']

    parts = None
    for pattern in regex_patterns:
        regex = re.compile(pattern)
        parts = regex.match(value)

        if parts is not None and parts.lastindex is not None:
            break

    if not parts or not parts.lastindex:
        raise ValueError('Unknown string format: {}'.format(value))

    parts = parts.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)

    return timedelta(**time_params)


def parse_coordinate(original_value):
    value = original_value.replace(',', '.')

    regex = re.compile(r'((?P<coordinate>-?\d+(\.)\d+)(E)?)?((?P<exp>-?\d+?))?')
    parts = regex.match(value)
    if not parts or sum([1 for part in parts.groupdict().values() if part]) == 0:
        raise ValueError('Unknown string format: {}'.format(value))

    parts = parts.groupdict()
    if 'coordinate' in parts and parts['coordinate']:
        coordinate = float(parts['coordinate'])
        if 'exp' in parts and parts['exp']:
            coordinate = coordinate * 10 ** float(parts['exp'])
    else:
        raise ValueError('Unknown string format: {}'.format(original_value))

    return round(coordinate, 6)
