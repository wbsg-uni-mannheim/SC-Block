import os

import fasttext


class LanguageDetector:
    def __init__(self):
        """Use fasttext for language detection"""
        # Silence warning: `load_model` does not return WordVectorModel or SupervisedModel any more,
        #                   but a `FastText` object which is very similar.
        fasttext.FastText.eprint = lambda x: None
        path_to_pretrained_model = '{}/models/fasttext/lid.176.bin'.format(os.environ['DATA_DIR'])
        self.fmodel = fasttext.load_model(path_to_pretrained_model)

    def check_language_is_not_english(self, raw_entity):
        """Check using fasttext if the value is with high confidence not english"""
        checked_attributes = ['name', 'description']
        value = ' '.join([raw_entity[checked_attribute] for checked_attribute in checked_attributes
                          if checked_attribute in raw_entity and raw_entity[checked_attribute] is not None
                          and type(raw_entity[checked_attribute]) not in [list, dict]])
        output = self.fmodel.predict([str(value)])
        return output[0][0][0] != '__label__en' and output[1][0][0] > 0.5

