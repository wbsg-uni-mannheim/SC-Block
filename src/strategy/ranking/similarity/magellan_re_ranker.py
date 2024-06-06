import os
import pickle

import pandas as pd
import py_entitymatching as em

from src.strategy.ranking.similarity.similarity_re_ranker import SimilarityReRanker


def determine_path_to_feature_table(schema_org_class, context_attributes):
    attribute_string = '_'.join([attr for attr in context_attributes])
    feature_table_path = '{}/magellan/{}/feature_table_{}_{}.p'.format(os.environ['DATA_DIR'], schema_org_class,
                                                                       schema_org_class, attribute_string)

    return feature_table_path


def determine_path_to_model(model_name, schema_org_class, context_attributes):
    attribute_string = '_'.join([attr for attr in context_attributes])
    model_path = '{}/magellan/{}/{}_{}_{}.p'.format(os.environ['DATA_DIR'], schema_org_class, model_name,
                                                    schema_org_class, attribute_string)

    return model_path


def prepare_df_for_magellan(entities, excluded_attributes):
    df_entities = pd.DataFrame(entities)

    # Drop to be excluded attributes
    if excluded_attributes is not None:
        df_entities = df_entities.drop(columns=excluded_attributes)

    # Set ID
    df_entities['ID'] = range(0, len(df_entities))
    em.set_key(df_entities, 'ID')

    return df_entities


class MagellanSimilarityReRanker(SimilarityReRanker):

    def __init__(self, schema_org_class, model_name, context_attributes=None, matcher=False):
        super().__init__(schema_org_class, 'Magellan Matcher', context_attributes, matcher)
        magellan_path = '{}/magellan'.format(os.environ['DATA_DIR'])

        path_to_feature_table = determine_path_to_feature_table(schema_org_class,
                                                                self.entity_serializer.context_attributes)
        self.feature_table = em.load_object(path_to_feature_table)

        path_to_model = determine_path_to_model(model_name, schema_org_class,
                                                self.entity_serializer.context_attributes)
        self.model_name = '{}_{}_{}'.format(model_name, schema_org_class, '_'.join(self.entity_serializer.context_attributes))
        self.model = pickle.load(open(path_to_model, 'rb'))

    def predict_matches(self, entities1, entities2, excluded_attributes1=None, excluded_attributes2=None):

        # Project entities to only contain relevant attributes
        entities1 = [self.entity_serializer.project_entity(entity, excluded_attributes=excluded_attributes1)
                     for entity in entities1]
        entities2 = [self.entity_serializer.project_entity(entity, excluded_attributes=excluded_attributes2)
                     for entity in entities2]

        df_entities1 = pd.DataFrame(entities1)
        df_entities1['id'] = range(0, len(df_entities1))
        em.set_key(df_entities1, 'id')

        df_entities2 = pd.DataFrame(entities2)
        df_entities2['id'] = range(0, len(df_entities2))
        em.set_key(df_entities2, 'id')

        # Make sure that all context attributes are present in the data frame
        for attr in self.entity_serializer.context_attributes:
            if attr not in df_entities1.columns:
                df_entities1[attr] = ""
            if attr not in df_entities2.columns:
                df_entities2[attr] = ""

        data_links = [[value, value, value]for value in range(0, len(df_entities1))]
        df_links = pd.DataFrame(data_links, columns=['_id', 'entities1_id', 'entities2_id'])
        em.set_key(df_links, '_id')

        # Set foreign key relationships
        em.set_ltable(df_links, df_entities1)
        em.set_fk_ltable(df_links, 'entities1_id')
        em.set_rtable(df_links, df_entities2)
        em.set_fk_rtable(df_links, 'entities2_id')

        df_feature_vector = em.extract_feature_vecs(df_links, feature_table=self.feature_table,
                                                    show_progress=False)

        df_feature_vector.fillna(value=0, inplace=True)

        predictions = self.model.predict(table=df_feature_vector, exclude_attrs=['_id', 'entities1_id', 'entities2_id'],
                                         append=True, target_attr='predicted', inplace=False, return_probs=True,
                                         probs_attr='proba')

        return predictions[['predicted', 'proba']]

    def re_rank_evidences(self, query_table, evidences):
        """Re-rank evidences based on confidence of a cross encoder"""
        for row in query_table.table:
            rel_evidences = [evidence for evidence in evidences if evidence.entity_id == row['entityId']]

            if len(rel_evidences) > 0:

                left_entities = [row] * len(rel_evidences)
                right_entities = [rel_evidence.context for rel_evidence in rel_evidences]
                exclude_attributes = []
                if hasattr(query_table, 'target_attribute'):
                    exclude_attributes.append(query_table.target_attribute)
                preds = self.predict_matches(entities1=left_entities, entities2=right_entities,
                                             excluded_attributes1=exclude_attributes)

                for evidence, pred in zip(rel_evidences, preds['predicted'].values):
                    evidence.scores['{}-{}'.format(self.name, self.model_name)] = pred
                    evidence.similarity_score = pred

        updated_evidences = sorted(evidences, key=lambda evidence: evidence.similarity_score, reverse=True)
        if self.matcher:
            updated_evidences = [evidence for evidence in updated_evidences if evidence.similarity_score > 0.5]

        return updated_evidences
