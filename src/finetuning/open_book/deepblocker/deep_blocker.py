#GiG
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
import blocking_utils
from src.strategy.open_book.entity_serialization import EntitySerializer


class DeepBlocker:
    def __init__(self, tuple_embedding_model, vector_pairing_model):
        self.tuple_embedding_model = tuple_embedding_model
        self.vector_pairing_model = vector_pairing_model

    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self, dataset_name):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)

        entity_serializer = EntitySerializer(dataset_name)

        self.left_df["_merged_text"] = self.left_df.apply(entity_serializer.convert_to_str_representation, axis=1)
        self.right_df["_merged_text"] = self.right_df.apply(entity_serializer.convert_to_str_representation, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)

    def prepocess_tuple_embeddings(self, left_df, right_df, cols_to_block, train_pairs_df, dataset_name):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        self.validate_columns()
        self.preprocess_datasets(dataset_name)

        print("Performing pre-processing for tuple embeddings ")

        self.tuple_embedding_model.preprocess(self.left_df, self.right_df, train_pairs_df, dataset_name=dataset_name)

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])


        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(self.right_tuple_embeddings)

    def block_datasets(self, k):


        print("Querying the embeddings from left dataset")
        topK_neighbors = self.vector_pairing_model.query(self.left_tuple_embeddings, k)
        self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)

        return self.candidate_set_df
