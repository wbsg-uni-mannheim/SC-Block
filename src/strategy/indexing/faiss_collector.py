import logging
import os

import faiss
import numpy as np


def determine_path_to_faiss_index(schema_org_class, model_name, pool, sim, clusters, switched=False):
    path_to_faiss_dir = '{}/faiss/'.format(os.environ['DATA_DIR'])
    if not os.path.isdir(path_to_faiss_dir):
        os.mkdir(path_to_faiss_dir)
    if switched:
        if clusters:
            return '{}{}_faiss_{}_{}_{}_with_clusters_switched.index'.format(path_to_faiss_dir, schema_org_class,
                                                                    model_name.split('/')[-1], pool, sim)
        else:
            return '{}{}_faiss_{}_{}_{}_switched.index'.format(path_to_faiss_dir, schema_org_class,
                                                      model_name.split('/')[-1], pool, sim)
    else:
        if clusters:
            return '{}{}_faiss_{}_{}_{}_with_clusters.index'.format(path_to_faiss_dir, schema_org_class,
                                                                    model_name.split('/')[-1], pool, sim)
        else:
            return '{}{}_faiss_{}_{}_{}.index'.format(path_to_faiss_dir, schema_org_class,
                                                      model_name.split('/')[-1], pool, sim)


class FaissIndexCollector:
    def __init__(self, schema_org_class, model_name, pooling, similarity_measure, final_representation,
                 dimensions, clusters, switched=False):
        self.schema_org_class = schema_org_class
        self.model_name = model_name
        self.pooling = pooling
        self.similarity_measure = similarity_measure
        self.dimensions = dimensions
        self.clusters = clusters
        self.switched = switched

        # Initialize entity representations
        self.indices = {}
        if self.similarity_measure == 'f2':
            self.indices['index_{}_{}'.format(self.pooling, self.similarity_measure)] = faiss.IndexFlatL2(self.dimensions)
        else:
            self.indices['index_{}_{}'.format(self.pooling, self.similarity_measure)] = faiss.IndexFlatIP(self.dimensions)

        self.entity_representations = {}
        self.initialize_entity_representations()

        self.unsaved_representations = []
        self.next_representation = 0
        self.final_representation = final_representation

        self.collected_entities = 0

    def initialize_entity_representations(self):
        # Initialize entity representations
        self.entity_representations = {'{}_{}'.format(self.pooling, self.similarity_measure): []}

    def collect_entity_representation(self, entity):
        identifier = 'entity_vector_{}'.format(self.pooling)
        if self.similarity_measure == 'cos':
            identifier = identifier + '_norm'
        # entity_rep = action['_source'][identifier].copy()
        if identifier not in entity:
            logging.getLogger().warning('Identifier: {} is not defined!'.format(identifier))
        entity_rep = entity[identifier]
        self.entity_representations['{}_{}'.format(self.pooling, self.similarity_measure)].append(entity_rep)


    def add_entity_representations_to_indices(self):
        """Add entity representations to indices AND persist index"""
        # save neural entity representations to different indices
        index_identifier = 'index_{}_{}'.format(self.pooling, self.similarity_measure)
        representations = np.array(self.entity_representations['{}_{}'.format(self.pooling, self.similarity_measure)]).astype('float32')
        self.indices[index_identifier].add(representations)

    def save_indices(self):
        index_identifier = 'index_{}_{}'.format(self.pooling, self.similarity_measure)
        path_to_faiss_index = determine_path_to_faiss_index(self.schema_org_class, self.model_name, self.pooling, self.similarity_measure, self.clusters, switched=self.switched)
        # Write index to file - Persist
        faiss.write_index(self.indices[index_identifier], path_to_faiss_index)
        logging.info('Saved Index - {} for model {} and schema org class {}'.format(index_identifier,
                                                                                    self.model_name,
                                                                                    self.schema_org_class))

    def next_savable_entities(self):
        entities = None
        removable_result = None
        for result in self.unsaved_representations:
            if result[0] == self.next_representation:
                entities = result[1]
                self.next_representation += 1
                removable_result = result
                break

        if removable_result is not None:
            self.unsaved_representations.remove(removable_result)

        return entities
