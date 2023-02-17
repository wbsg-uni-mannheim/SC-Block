class BaseEvidence:

    def __init__(self, identifier, query_table_id, entity_id, table, row_id, context, split=None):
        self.identifier = identifier
        self.query_table_id = query_table_id
        self.entity_id = entity_id
        self.table = table
        self.row_id = row_id
        self.context = context
        self.split = split
        self.signal = None
        self.scale = None
        self.corner_case = None
        self.similarity_score = None
        self.seen_training = None
        self.scores = {}

    def verify(self, signal):
        if signal is not None:
            self.signal = signal
        else:
            raise ValueError('The value of signal must be defined (True/False)!')

    def set_context(self, context):
        self.context = context

    def to_json(self, with_evidence_context, without_score=True):
        encoded_evidence = {}

        # Camelcase encoding for keys and fill encoded evidence
        for key in self.__dict__.keys():
            camel_cased_key = ''.join([key_part.capitalize() for key_part in key.split('_')])
            camel_cased_key = camel_cased_key[0].lower() + camel_cased_key[1:]
            if camel_cased_key == 'identifier':
                encoded_evidence['id'] = self.__dict__['identifier']
            elif camel_cased_key == 'context':
                if with_evidence_context:
                    # Save evidence only if it is requested!
                    encoded_evidence[camel_cased_key] = self.__dict__[key]
            elif without_score and camel_cased_key in ['scores', 'similarityScore']:
                # Do not save similarity scores! Scores are only used at runtime.
                continue
            else:
                encoded_evidence[camel_cased_key] = self.__dict__[key]

        return encoded_evidence

    def __hash__(self):
        return hash('-'.join([str(self.query_table_id), str(self.entity_id), self.table, str(self.row_id)]))
        #return '-'.join([str(self.query_table_id), str(self.entity_id), self.table, str(self.row_id)])

    def __str__(self):
        return 'Query Table: {}, Entity Id: {}, Table: {}, Row: {}, Signal: {}' \
            .format(self.query_table_id, self.entity_id, self.table, self.row_id, self.signal)

    def __eq__(self, other):
        try:
            return self.__hash__() == other.__hash__()
        except AttributeError:
            return NotImplemented

    def __copy__(self):
        evidence_copy = BaseEvidence(self.identifier, self.query_table_id, self.entity_id, self.table, self.row_id,
                                     self.context)
        evidence_copy.scale = self.scale
        evidence_copy.signal = self.signal
        evidence_copy.corner_case = self.corner_case
        evidence_copy.similarity_score = self.similarity_score
        evidence_copy.scores = self.scores.copy()

        return evidence_copy

    def aggregate_scores_to_similarity_score(self):

        # Average scores for now
        score_values = [value for value in self.scores.values()]
        if len(score_values) > 0:
            self.similarity_score = sum(score_values) / len(score_values)
        else:
            self.similarity_score = 0


class RetrievalEvidence(BaseEvidence):

    def __init__(self, identifier, query_table_id, entity_id, table, row_id, context, split=None):
        super().__init__(identifier, query_table_id, entity_id, table, row_id, context, split)

    def __str__(self):
        return 'Query Table: {}, Entity Id: {}, Table: {}, Row: {}, Signal: {}, Similarity: {}' \
            .format(self.query_table_id, self.entity_id, self.table, self.row_id, self.signal, self.similarity_score)

    def __repr__(self):
        return self.__str__()


class AugmentationEvidence(BaseEvidence):

    def __init__(self, identifier, query_table_id, entity_id, table, row_id, context, value, attribute, split=None):
        super().__init__(identifier, query_table_id, entity_id, table, row_id, context, split)
        self.value = value
        self.attribute = attribute

    def __str__(self):
        return 'Query Table: {}, Entity Id: {}, Attribute: {}, Table: {}, Row: {}, Value: {}, Signal: {}' \
            .format(self.query_table_id, self.entity_id, self.attribute, self.table, self.row_id, self.value,
                    self.signal)

    def __copy__(self):
        evidence_copy = AugmentationEvidence(self.identifier, self.query_table_id, self.entity_id,
                                             self.value, self.table, self.row_id, self.attribute,
                                             self.context)
        evidence_copy.scale = self.scale
        evidence_copy.signal = self.signal
        evidence_copy.corner_case = self.corner_case
        evidence_copy.similarity_score = self.similarity_score
        evidence_copy.scores = self.scores.copy()

        return evidence_copy

    def determine_scale(self, table):
        """
        Determine the scale of an evidence
            3: Evidence is positive and value exactly matches the target attributes value
            2: Evidence is positive and value matches target attribute value according to a specified evaluation rule
            1: Evidence is positive, but the values does not match the target attributes value
            0: Evidence is negative
        """
        if not self.signal:
            self.scale = 0
        else:
            for row in table:
                if row['entityId'] == self.entity_id:
                    self.scale = 3 if row[self.attribute] == self.value else 1
                    break
