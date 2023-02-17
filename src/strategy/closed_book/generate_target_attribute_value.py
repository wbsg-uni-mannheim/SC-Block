import logging
import os
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.model.deprecated.deprecated_evidence import DeprecatedEvidence
from src.strategy.open_book.retrieval.retrieval_strategy import RetrievalStrategy


def create_source_sequence(entity, target_attribute, schema_org_class):
    if schema_org_class == 'movie':
        identifying_attributes = ['name', 'director', 'duration', 'datepublished']
    elif schema_org_class == 'localbusiness':
        # Focus on Locality and name for now
        #identifying_attributes = ['addresslocality', 'addressregion', 'addresscountry', 'postalcode', 'name', 'streetaddress']
        identifying_attributes = ['name', 'addresslocality']
    else:
        logging.warning(
            'Identifying attributes are not defined for schema org class {}'.format(schema_org_class))

    # No prefix for now, because of the focus on one task --> table augmentation
    #prefix = "table augmentation: "
    encoded_entity = ''
    for identifying_attribute in identifying_attributes:
        if identifying_attribute != target_attribute \
                and identifying_attribute in entity:
            encoded_entity = "{}[COL]{}[VAL]{}".format(encoded_entity, identifying_attribute,
                                                       entity[identifying_attribute])
    source = "{}. target:[COL]{}".format(encoded_entity, target_attribute)
    return source



class TargetAttributeValueGenerator(RetrievalStrategy):

    def __init__(self, schema_org_class, model_name):
        super().__init__(schema_org_class, 'generate_entity')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        self.model_name = model_name
        model_path = '{}/models/closed_book/{}'.format(os.environ['DATA_DIR'], model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)


    # Sequence generator
    def retrieve_evidence(self, query_table, evidence_count, entity_id):
        logger = logging.getLogger()
        evidence_id = 1
        evidences = []

        # Iterate through query table and create entity embeddings (neural representations)
        for row in query_table.table:
            source = create_source_sequence(row, query_table.target_attribute, self.schema_org_class)

            #print(self.tokenizer.tokenize(source))

            input_ids = self.tokenizer(source, return_tensors='pt').input_ids.to(self.device)
            outputs = self.model.generate(input_ids)
            found_value = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace('[VAL]', '')

            evidence = DeprecatedEvidence(evidence_id, query_table.identifier, row['entityId'], found_value,
                                          None, None, query_table.target_attribute, None)
            evidences.append(evidence)
            evidence_id += 1

        return evidences

    def re_rank_evidences(self, query_table, evidences):
        """Just return evidences for now"""

        return evidences
