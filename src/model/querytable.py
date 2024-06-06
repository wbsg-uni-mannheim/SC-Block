import copy
import logging
import json
import os
import itertools

from src.model.evidence import RetrievalEvidence, AugmentationEvidence
from src.strategy.entity_serialization import EntitySerializer


def load_query_table(raw_json):
    # Load and initialize verified evidences
    logger = logging.getLogger()

    verified_evidences = []
    for raw_evidence in raw_json['verifiedEvidences']:
        context = None
        if 'context' in raw_evidence:
            context = raw_evidence['context']

        if raw_json['type'] == 'retrieval':
            evidence = RetrievalEvidence(raw_evidence['id'], raw_evidence['queryTableId'], raw_evidence['entityId'],
                            raw_evidence['table'], raw_evidence['rowId'], context, raw_evidence['split'])
        elif raw_json['type'] == 'augmentation':
            evidence = AugmentationEvidence(raw_evidence['id'], raw_evidence['queryTableId'], raw_evidence['entityId'],
                                            raw_evidence['table'], raw_evidence['rowId'], context,
                                            raw_evidence['value'], raw_evidence['attribute'], raw_evidence['split'])
        else:
            raise ValueError('Retrieval Type {} is not defined!'.format(raw_json['type']))

        if 'seenTraining' in raw_evidence:
            evidence.seen_training = raw_evidence['seenTraining']

        if 'scale' in raw_evidence:
            evidence.scale = raw_evidence['scale']

        if 'signal' in raw_evidence:
            evidence.verify(raw_evidence['signal'])
            if 'scale' not in raw_evidence:
                evidence.determine_scale(raw_json['table'])

        if 'cornerCase' in raw_evidence:
            evidence.corner_case = raw_evidence['cornerCase']

        if evidence.query_table_id == raw_json['id'] and evidence not in verified_evidences:
            verified_evidences.append(evidence)
        elif evidence in verified_evidences:
            logger.warning('Evidence: {} already contained in query table {}'.format(evidence, raw_json['id']))
        else:
            logger.warning('Evidence: {} does not belong to query table {}'.format(evidence, raw_json['id']))

    if raw_json['type'] == 'retrieval':
        return RetrievalQueryTable(raw_json['id'], 'retrieval', raw_json['assemblingStrategy'],
                      raw_json['gtTable'],
                      raw_json['schemaOrgClass'],
                      raw_json['contextAttributes'],
                      raw_json['table'], verified_evidences)
    elif raw_json['type'] == 'augmentation':
        return AugmentationQueryTable(raw_json['id'], 'augmentation', raw_json['assemblingStrategy'],
                      raw_json['gtTable'],
                      raw_json['schemaOrgClass'],
                      raw_json['contextAttributes'],
                      raw_json['table'], verified_evidences, raw_json['targetAttribute'], raw_json['useCase'])
    else:
        raise ValueError('Retrieval Type {} is not defined!'.format(raw_json['type']))


def load_query_table_from_file(path):
    """Load query table from provided path and return new Querytable object"""
    logger = logging.getLogger()

    with open(path, encoding='utf-8') as gsFile:
        logger.warning('Load query table from ' + path)
        try:
            querytable = load_query_table(json.load(gsFile))
        except UnicodeDecodeError:
            logger.warning('Not able to load query table from {}'.format(path))
            querytable = None
        if type(querytable) is not BaseQueryTable and type(querytable) is not RetrievalQueryTable \
                and type(querytable) is not AugmentationQueryTable:
            print(type(querytable))
            logger.warning('Not able to load query table from {}'.format(path))
        return querytable


def load_query_tables(type):
    """Load all query tables"""
    query_tables = []
    for query_table_path in get_all_query_table_paths(type):
        query_tables.append(load_query_table_from_file(query_table_path))

    return query_tables


def load_query_tables_by_class(type, schema_org_class):
    """Load all query tables of a specific query table"""
    query_tables = []
    for gt_table in get_gt_tables(type, schema_org_class):
        for query_table_path in get_query_table_paths(type, schema_org_class, gt_table):
            query_tables.append(load_query_table_from_file(query_table_path))

    return query_tables


def get_schema_org_classes(type):
    """Get a list of all schema org classes
        :param type string Type of query table that has to be loaded - Retrieval/ Augmentation"""
    schema_org_classes = []
    path_to_classes = '{}/querytables/'.format(os.environ['DATA_DIR'])
    print(path_to_classes)
    if os.path.isdir(path_to_classes):
        schema_org_classes = [schema_org_class for schema_org_class in os.listdir(path_to_classes)
                              if schema_org_class != 'deprecated' and 'test' not in schema_org_class]

    return schema_org_classes


def get_gt_tables(type, schema_org_class):
    """Get list of categories by schema org"""
    gt_tables = []
    path_to_gt_tables = '{}/querytables/{}/{}/'.format(os.environ['DATA_DIR'], schema_org_class, type)
    if os.path.isdir(path_to_gt_tables):
        gt_tables = [gt_table for gt_table in os.listdir(path_to_gt_tables) if gt_table != 'deprecated']

    return gt_tables


def get_query_table_paths(type, schema_org_class, gt_table, switched=False):
    """Get query table paths"""
    query_table_files = []
    if switched:
        path_to_query_tables = '{}/querytables/{}/switched/{}/{}/'.format(os.environ['DATA_DIR'], schema_org_class, type, gt_table)
    else:
        path_to_query_tables = '{}/querytables/{}/{}/{}/'.format(os.environ['DATA_DIR'], schema_org_class, type, gt_table)

    if os.path.isdir(path_to_query_tables):
        # Filter for json files
        query_table_files = ['{}{}'.format(path_to_query_tables, filename) for filename in os.listdir(path_to_query_tables)
                             if '.json' in filename]

    return query_table_files


def get_all_query_table_paths(type):
    query_table_paths = []
    for schema_org_class in get_schema_org_classes(type):
        for gt_table in get_gt_tables(type,schema_org_class):
            query_table_paths.extend(get_query_table_paths(type, schema_org_class, gt_table))

    return query_table_paths


def create_context_attribute_permutations(querytable):
    """Create all possible query table permutations based on the context attributes of the provided query table"""
    permutations = []
    for i in range(len(querytable.context_attributes)):
        permutations.extend(itertools.permutations(querytable.context_attributes, i))

    # Remove permutations that do not contain name attribute
    permutations = [permutation for permutation in permutations if 'name' in permutation
                    and permutation != querytable.context_attributes]

    querytables = []
    for permutation in permutations:
        new_querytable = copy.deepcopy(querytable)
        removable_attributes = [attr for attr in new_querytable.context_attributes if attr not in permutation]
        for attr in removable_attributes:
            new_querytable.remove_context_attribute(attr)
        querytables.append(new_querytable)

    return querytables


class BaseQueryTable:

    def __init__(self, identifier, type, assembling_strategy, gt_table, schema_org_class,
                 context_attributes, table, verified_evidences):
        self.identifier = identifier
        self.type = type
        self.assembling_strategy = assembling_strategy
        self.gt_table = gt_table
        self.schema_org_class = schema_org_class
        self.context_attributes = context_attributes
        self.table = table
        self.verified_evidences = verified_evidences
        self.retrieved_evidences = None
        self.switched = False  # Flag to indicate if the query table and index table have been switched

    def __str__(self):
        return self.to_json(with_evidence_context=False)

    def to_json(self, with_evidence_context, with_retrieved_evidences=False):
        encoded_evidence = {}
        # Camelcase encoding for keys and fill encoded evidence
        for key in self.__dict__.keys():
            if key == 'identifier':
                encoded_evidence['id'] = self.__dict__['identifier']
            elif key == 'verified_evidences':
                encoded_evidence['verifiedEvidences'] = [evidence.to_json(with_evidence_context) for evidence in
                                                         self.verified_evidences]
            elif key == 'retrieved_evidences':
                if with_retrieved_evidences and self.retrieved_evidences is not None:
                    encoded_evidence['retrievedEvidences'] = [evidence.to_json(with_evidence_context, without_score=False) for evidence in
                                                             self.retrieved_evidences]
            else:
                camel_cased_key = ''.join([key_part.capitalize() for key_part in key.split('_')])
                camel_cased_key = camel_cased_key[0].lower() + camel_cased_key[1:]
                encoded_evidence[camel_cased_key] = self.__dict__[key]

        return encoded_evidence

    def no_known_positive_evidences(self, entity_id):
        """Calculate number of know positive evidences"""
        return sum([1 for evidence in self.verified_evidences
                    if evidence.signal and evidence.entity_id == entity_id])

    def has_verified_evidences(self):
        return len(self.verified_evidences) > 0

    def determine_path_to_querytable(self):
        gt_table = self.gt_table.lower().replace(" ", "_")
        file_name = 'gs_querytable_{}_{}.json'.format(gt_table, self.identifier)
        logging.info('Switched: {}'.format(self.switched))
        if self.switched:
            return '{}/querytables/{}/switched/{}/{}/{}'.format(os.environ['DATA_DIR'], self.schema_org_class,
                                                               self.type, gt_table, file_name)
        else:
            return '{}/querytables/{}/{}/{}/{}'.format(os.environ['DATA_DIR'], self.schema_org_class,
                                                               self.type, gt_table, file_name)

    def save(self, with_evidence_context, with_retrieved_evidences=False):
        """Save query table on disk"""
        logger = logging.getLogger()

        path_to_query_table = self.determine_path_to_querytable()

        # Check if file path exists --> create if necessary
        file_name = path_to_query_table.split('/')[-1]
        if not os.path.isdir(path_to_query_table.replace('/{}'.format(file_name), '')):
            os.makedirs(path_to_query_table.replace('/{}'.format(file_name), ''))

        # Save query table to file
        with open(path_to_query_table, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(with_evidence_context, with_retrieved_evidences), f, indent=2, ensure_ascii=False)
            logger.info('Save query table {}'.format(path_to_query_table))


    def calculate_evidence_statistics_of_row(self, entity_id):
        """Export Query Table Statistics per entity"""
        row = [row for row in self.table if row['entityId'] == entity_id].pop()

        evidences = sum([1 for evidence in self.verified_evidences
                         if evidence.entity_id == row['entityId']])
        correct_value_entity = sum([1 for evidence in self.verified_evidences
                                    if evidence.entity_id == row['entityId'] and evidence.scale == 3])
        rel_value_entity = sum([1 for evidence in self.verified_evidences
                                if evidence.entity_id == row['entityId'] and evidence.scale == 2])
        correct_entity = sum([1 for evidence in self.verified_evidences
                              if evidence.entity_id == row['entityId'] and evidence.scale == 1])
        not_correct_entity = sum([1 for evidence in self.verified_evidences
                                  if evidence.entity_id == row['entityId'] and evidence.scale == 0])

        return evidences, correct_value_entity, rel_value_entity, correct_entity, not_correct_entity

    def remove_context_attribute(self, attribute):
        """Remove specified context attribute"""
        logger = logging.getLogger()
        if attribute == 'name':
            raise ValueError('It is not allowed to remove the name attribute from the query table!')

        if attribute not in self.context_attributes:
            raise ValueError('Context attribute {} not found in query table {}!'.format(attribute, self.identifier))

        if attribute in self.context_attributes:
            for row in self.table:
                try:
                    del row[attribute]
                except KeyError as e:
                    logger.warning('Keyerrror {}: Identifier {} - Ground Truth table {}'.format(attribute, self.identifier,
                                                                                      self.gt_table))

            self.context_attributes.remove(attribute)
            logger.debug('Removed context attribute {} from querytable {}!'.format(attribute, self.identifier))

    def add_verified_evidence(self, evidence):
        logger = logging.getLogger()
        if evidence.query_table_id != self.identifier:
            logger.warning('Evidence {} does not belong to query table {}!'.format(evidence.identifier, self.identifier))
        else:
            self.verified_evidences.append(evidence)

    def normalize_query_table_numbering(self):
        # Change numbering of entities in query table
        # i = 1000  # Move entity ids to value range that is not used
        # for row in self.table:
        #     if row['entityId'] != i:
        #         relevant_evidences = [evidence for evidence in self.verified_evidences
        #                               if evidence.entity_id == row['entityId']]
        #         for evidence in relevant_evidences:
        #             evidence.entity_id = i
        #         row['entityId'] = i
        #     i += 1

        i = 0
        for row in self.table:
            if row['entityId'] != i:
                relevant_evidences = [evidence for evidence in self.verified_evidences
                                      if evidence.entity_id == row['entityId']]
                for evidence in relevant_evidences:
                    evidence.entity_id = i
                row['entityId'] = i
            i += 1

        # Change numbering of evidences in query table
        i = 0
        #self.verified_evidences = list(set(self.verified_evidences))
        for evidence in self.verified_evidences:
            if evidence.identifier != i:
                evidence.identifier = i
            i += 1

    def append(self, query_table):
        """
            Append rows and evidences to query table -  Not implemented for Base Query Table
            :param BaseQueryTable query_table : Query table to be appended
        """
        pass

    def materialize_pairs(self):

        entity_serializer = EntitySerializer(self.schema_org_class, self.context_attributes)

        pairs = []
        pair = {'dataset': self.schema_org_class}
        positive_evidences = [evidence for evidence in self.verified_evidences
                              if evidence.signal and evidence.split == 'test']
        negative_evidences = [evidence for evidence in self.verified_evidences
                              if not evidence.signal and evidence.split == 'test']
        for row in self.table:
            pair['ltableid'] = row['entityId']
            pair['lencoded'] = entity_serializer.convert_to_str_representation(row).replace('[COL]', 'COL').replace('[VAL]', 'VAL')
            all_retrieved_evidences = [evidence for evidence in self.retrieved_evidences if
                                       evidence.entity_id == row['entityId'] and
                                       (evidence in positive_evidences or evidence in negative_evidences)]
            for evidence in all_retrieved_evidences:
                new_pair = pair.copy()
                new_pair['rtableid'] = evidence.row_id
                new_pair['matched_table'] = evidence.table
                new_pair['rencoded'] = entity_serializer.convert_to_str_representation(evidence.context).replace('[COL]', 'COL').replace('[VAL]', 'VAL')
                new_pair['match'] = 1 if evidence.signal else 0
                pairs.append(new_pair)

        #df_pairs = pd.DataFrame(pairs)
        # path = self.determine_path_to_querytable()
        # path = path.replace('.json', '')
        # path = '{}_{}.csv'.format(path, retrieval_method)
        # df_pairs.to_csv(path)
        return pairs


class RetrievalQueryTable(BaseQueryTable):

    def __init__(self, identifier, type, assembling_strategy, gt_table, schema_org_class, context_attributes,
                 table, verified_evidences):
        super().__init__(identifier, type, assembling_strategy, gt_table, schema_org_class, context_attributes,
                         table, verified_evidences)

    def append(self, query_table):
        """
            Append rows and evidences to query table
            :param RetrievalQueryTable query_table : Query table to be appended
        """
        map_old_entity_id_new_entity_id = {}
        next_entity_id = max([row['entityId'] for row in self.table]) + 1
        exclude_rows = []

        for row in query_table.table:
            # Sanity check for localbusiness tables
            for existing_row in self.table:
                if existing_row['name'] == row['name'] and existing_row['addresslocality'] == row['addresslocality']:
                    exclude_rows.append(row['entityId'])
                    break
            if row['entityId'] not in exclude_rows:
                map_old_entity_id_new_entity_id[row['entityId']] = next_entity_id
                new_row = row.copy()
                new_row['entityId'] = next_entity_id
                next_entity_id += 1
                self.table.append(new_row)

        for evidence in query_table.verified_evidences:
            if evidence.entity_id not in exclude_rows:
                new_evidence = RetrievalEvidence(evidence.identifier, self.identifier,
                                                 map_old_entity_id_new_entity_id[evidence.entity_id],
                                                 evidence.table, evidence.row_id, evidence.context)
                new_evidence.scale = evidence.scale
                new_evidence.signal = evidence.signal
                new_evidence.corner_case = evidence.corner_case
                self.verified_evidences.append(new_evidence)


class AugmentationQueryTable(BaseQueryTable):

    def __init__(self, identifier, type, assembling_strategy, gt_table, schema_org_class,
                 context_attributes, table, verified_evidences, target_attribute, use_case):
        super().__init__(identifier, type, assembling_strategy, gt_table, schema_org_class,
                         context_attributes, table, verified_evidences)
        self.target_attribute = target_attribute
        self.use_case = use_case

    def determine_path_to_querytable(self):
        gt_table = self.gt_table.lower().replace(" ", "_")
        file_name = 'gs_querytable_{}_{}_{}.json'.format(gt_table, self.target_attribute, self.identifier)
        return '{}/querytables/{}/{}/{}/{}'.format(os.environ['DATA_DIR'], self.schema_org_class,
                                                   self.type, gt_table, file_name)

    def append(self, query_table):
        """
            Append rows and evidences to query table
            :param AugmentationQueryTable query_table : Query table to be appended
        """
        map_old_entity_id_new_entity_id = {}
        next_entity_id = max([row['entityId'] for row in self.table]) + 1
        exclude_rows = []

        for row in query_table.table:
            # Sanity check for localbusiness tables
            for existing_row in self.table:
                if existing_row['name'] == row['name'] and existing_row['addresslocality'] == row['addresslocality']:
                    exclude_rows.append(row['entityId'])
                    break

            if row['entityId'] not in exclude_rows:
                map_old_entity_id_new_entity_id[row['entityId']] = next_entity_id
                new_row = row.copy()
                new_row['entityId'] = next_entity_id
                next_entity_id += 1
                self.table.append(new_row)

        for evidence in query_table.verified_evidences:
            if evidence.entity_id not in exclude_rows:
                new_evidence = AugmentationEvidence(evidence.identifier, self.identifier,
                                                    map_old_entity_id_new_entity_id[evidence.entity_id],
                                                    evidence.table, evidence.row_id, evidence.context,
                                                    evidence.value, evidence.attribute)
                new_evidence.scale = evidence.scale
                new_evidence.signal = evidence.signal
                new_evidence.corner_case = evidence.corner_case
                self.verified_evidences.append(new_evidence)
