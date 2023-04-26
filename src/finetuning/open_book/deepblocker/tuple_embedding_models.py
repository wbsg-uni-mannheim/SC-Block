#GiG
import json
import os
import pickle
from collections import Counter
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import torch 

import fasttext  
from torchtext.data import get_tokenizer


#This is the Abstract Base Class for all Tuple Embedding models

from src.finetuning.open_book.deepblocker import dl_models
from src.finetuning.open_book.deepblocker.configurations import FASTTEXT_EMBEDDIG_PATH, EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE, BATCH_SIZE, NUM_EPOCHS, RANDOM_SEED
from src.finetuning.open_book.deepblocker.convert_synthetic_training_data import \
    convert_synthetic_data_to_clustered_data


class ABCTupleEmbedding:
    def __init__(self):
        pass 

    #This function is used as a preprocessing step 
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, left_df, right_df, train_pairs=None, dataset_name=None):
        pass 

    #This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        pass 

    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        pass 

#This is a simple method for aggregation 
# This computes the embedding of a tuple as the average of the constituent word embeddings. 
#By default, it uses fastText model
class AverageEmbedding(ABCTupleEmbedding):
    def __init__(self):
        super().__init__()
        print("Loading FastText model")

        self.word_embedding_model = fasttext.load_model(FASTTEXT_EMBEDDIG_PATH)
        self.dimension_size = EMB_DIMENSION_SIZE

        self.tokenizer = get_tokenizer("basic_english")


    #There is no pre processing needed for Average Embedding
    def preprocess(self, left_df, right_df, train_pairs=None):
        pass

    #list_of_strings is an Iterable of tuples as strings
    def get_tuple_embedding(self, list_of_tuples):
        #This is an one liner for efficiency
        # returns a list of word embeddings for each token in a tuple 
        #   self.word_embedding_model.get_word_vector(token) for token in self.tokenizer(tuple)
        # next we convert the list of word embeddings to a numpy array using np.array(list)
        # next we compute the element wise mean via np.mean(np.array([embeddings]), axis=0)
        # we repeat this process for all tuples in list_of_tuples
        #       for tuple in list_of_tuples
        # then convert everything to a numpy array at the end through np.array([ ]) 
        # So if you send N tuples, then this will return a numpy matrix of size N x D where D is embedding dimension
        average_embeddings = np.array([np.mean(np.array([self.word_embedding_model.get_word_vector(token) for token in self.tokenizer(_tuple)]), axis=0) for _tuple in list_of_tuples]) 
        return average_embeddings

    #Return word embeddings for a list of words
    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words]



        

class SIFEmbedding(ABCTupleEmbedding):
    #sif_weighting_param is a parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # the SIF paper set the default value to 1e-3
    # remove_pc is a Boolean parameter that controls whether to remove the first principal component or not
    # min_freq: if a word is too infrequent (ie frequency < min_freq), set a SIF weight of 1.0 else apply the formula
    #   The SIF paper applies this formula if the word is not the top-N most frequent
    def __init__(self, sif_weighting_param=1e-3, remove_pc=True, min_freq=0):
        super().__init__()
        print("Loading FastText model")

        self.word_embedding_model = fasttext.load_model(FASTTEXT_EMBEDDIG_PATH)
        self.dimension_size = EMB_DIMENSION_SIZE

        self.tokenizer = get_tokenizer("basic_english")

        #Word to frequency counter
        self.word_to_frequencies = Counter()

        #Total number of distinct tokens
        self.total_tokens = 0

        self.sif_weighting_param = sif_weighting_param
        self.remove_pc = remove_pc
        self.min_freq = min_freq
        
        self.token_weight_dict = {}



    #There is no pre processing needed for Average Embedding
    def preprocess(self, left_df, right_df, train_pairs=None):
        list_of_tuples = pd.concat([left_df["_merged_text"], right_df["_merged_text"]], ignore_index=True)
        for tuple_as_str in list_of_tuples:
            self.word_to_frequencies.update(self.tokenizer(tuple_as_str))

        self.calculate_token_statistics()


    def calculate_token_statistics(self):
        # Count all the tokens in each tuples
        self.total_tokens = sum(self.word_to_frequencies.values())

        # Compute the weight for each token using the SIF scheme
        a = self.sif_weighting_param
        for word, frequency in self.word_to_frequencies.items():
            if frequency >= self.min_freq:
                self.token_weight_dict[word] = a / (a + frequency / self.total_tokens)
            else:
                self.token_weight_dict[word] = 1.0

    #list_of_strings is an Iterable of tuples as strings
    # See the comments of AverageEmbedding's get_tuple_embedding for details about how this works
    def get_tuple_embedding(self, list_of_tuples):
        num_tuples = len(list_of_tuples)
        tuple_embeddings = np.zeros((num_tuples, self.dimension_size))

        for index, _tuple in enumerate(list_of_tuples):
            #Compute a weighted average using token_weight_dict
            tuple_embeddings[index, :] = np.mean(np.array([self.word_embedding_model.get_word_vector(token) *  self.token_weight_dict[token] for token in self.tokenizer(_tuple)]), axis=0)

        #From the code of the SIF paper at 
        # https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
        if self.remove_pc:
            svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
            svd.fit(tuple_embeddings)
            pc = svd.components_

            sif_embeddings = tuple_embeddings - tuple_embeddings.dot(pc.transpose()) * pc
        else:
            sif_embeddings = tuple_embeddings

        return  sif_embeddings

    def get_word_embedding(self, list_of_words):
        return [self.word_embedding_model.get_word_vector(word) for word in list_of_words] 


class AutoEncoderTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dimensions=(2*AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE)):
        super().__init__()
        self.input_dimension = EMB_DIMENSION_SIZE
        self.hidden_dimensions = hidden_dimensions
        self.sif_embedding_model = SIFEmbedding()


    #This function is used as a preprocessing step 
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, left_df, right_df, train_pairs=None, dataset_name=None):
        print("Training AutoEncoder model")
        list_of_tuples = pd.concat([left_df["_merged_text"], right_df["_merged_text"]], ignore_index=True)
        self.sif_embedding_model.preprocess(left_df, right_df)
        embedding_matrix = self.sif_embedding_model.get_tuple_embedding(list_of_tuples)
        trainer = dl_models.AutoEncoderTrainer (self.input_dimension, self.hidden_dimensions)
        self.autoencoder_model = trainer.train(embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        # trainer.save_model('models/AUTO_model_{}.bin'.format(dataset_name))
        # with open('models/AUTO_sif_{}.json'.format(dataset_name), 'w') as handle:
        #     json.dump(dict(self.sif_embedding_model.word_to_frequencies), handle)
        trainer.save_model('{}reports/{}/AUTO-model-{}-{}-{}.bin'.format(os.environ['DATA_DIR'],
                                                                                         dataset_name, dataset_name,
                                                                                         NUM_EPOCHS, BATCH_SIZE))
        with open('{}reports/{}/AUTO-sif-{}.json'.format(os.environ['DATA_DIR'], dataset_name, dataset_name), 'w') as handle:
            json.dump(dict(self.sif_embedding_model.word_to_frequencies), handle)
 

    #This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_tuples)).float()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)
        


    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_words)).float()
        return self.autoencoder_model.get_tuple_embedding(embedding_matrix)

#This function is used by both CTT and Hybrid  - so it is put outside of any class
#It takes a list of tuple strings and outputs three lists (T, T', L)
# t_i \in T and t'_i \in T' are (potentially perturbed) tuples 
# and l_i is a label denoting whether they are duplicates or not
# for each tuple t in list_of_tuples, 
# we generate synth_tuples_per_tuple positive tuple pairs
# and synth_tuples_per_tuple * pos_to_neg_ratio negative tuple pairs
def generate_synthetic_training_data(list_of_tuples, list_of_sources, dataset_name, synth_tuples_per_tuple=5,
        pos_to_neg_ratio=1, max_perturbation=0.4):
    num_positives_per_tuple = synth_tuples_per_tuple
    num_negatives_per_tuple = synth_tuples_per_tuple * pos_to_neg_ratio
    num_tuples = len(list_of_tuples)
    total_number_of_elems = num_tuples * (num_positives_per_tuple + num_negatives_per_tuple)

    #We create three lists containing T, T' and L respectively
    #We use the following format: first num_tuples * num_positives_per_tuple correspond to T
    # and the remaining correspond to T'
    left_tuple_list = [None for _ in range(total_number_of_elems)]
    right_tuple_list = [None for _ in range(total_number_of_elems)]
    source_list = [None for _ in range(total_number_of_elems)]
    label_list = [0 for _ in range(total_number_of_elems) ]
    tuple_id_list = [None for _ in range(total_number_of_elems)]

    random.seed(RANDOM_SEED)

    tokenizer = get_tokenizer("basic_english")
    for index in range(len(list_of_tuples)):
        tokenized_tuple = tokenizer(list_of_tuples[index])
        source = list_of_sources[index]
        max_tokens_to_remove = int(len(tokenized_tuple) * max_perturbation)
     
        training_data_index = index * (num_positives_per_tuple + num_negatives_per_tuple)

        #Create num_positives_per_tuple tuple pairs with positive label
        for temp_index in range(num_positives_per_tuple):
            tokenized_tuple_copy = tokenized_tuple[:]

            #If the tuple has 10 words and max_tokens_to_remove is 0.5, then we can remove at most 5 words
            # we choose a random number between 0 and 5.
            # suppose it is 3. Then we randomly remove 3 words
            num_tokens_to_remove = random.randint(0, max_tokens_to_remove)
            for _ in range(num_tokens_to_remove):
                #randint is inclusive. so randint(0, 5) can return 5 also
                tokenized_tuple_copy.pop( random.randint(0, len(tokenized_tuple_copy) - 1) )

            left_tuple_list[training_data_index] = list_of_tuples[index]
            right_tuple_list[training_data_index] = ' '.join(tokenized_tuple_copy)
            label_list[training_data_index] = 1
            source_list[training_data_index] = source
            tuple_id_list[training_data_index] = index
            training_data_index += 1

        for temp_index in range(num_negatives_per_tuple):
            left_tuple_list[training_data_index] = list_of_tuples[index]
            right_tuple_list[training_data_index] = random.choice(list_of_tuples)
            label_list[training_data_index] = 0
            source_list[training_data_index] = source
            tuple_id_list[training_data_index] = index
            training_data_index += 1

    s_left_tuple_list = pd.Series(left_tuple_list)
    s_right_tuple_list = pd.Series(right_tuple_list)
    s_label_list = pd.Series(label_list)
    s_source_list = pd.Series(source_list)
    s_tuple_id_list = pd.Series(tuple_id_list)

    df_synthetic_data = s_left_tuple_list.to_frame(name='left_tuples').merge(s_right_tuple_list.to_frame(name='right_tuples'), left_index=True, right_index=True) \
        .merge(s_source_list.to_frame(name='source'), left_index=True, right_index=True) \
        .merge(s_tuple_id_list.to_frame(name='cluster_id'), left_index=True, right_index=True) \
        .merge(s_label_list.to_frame(name='labels'), left_index=True, right_index=True)

    convert_synthetic_data_to_clustered_data(df_synthetic_data, dataset_name)
    #string_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # filename_synthetic_training_data = 'synthetic_training_data_{}_{}.csv'.format(string_timestamp, dataset_name)
    #df_synthetic_data.to_csv(filename_synthetic_training_data, sep=';', encoding='utf-8', index=False)


    #print('Saved Synthetic training data - {}!'.format(string_timestamp))


    return left_tuple_list, right_tuple_list, label_list
    

def generate_training_data(left_df, right_df, train_pairs):

    left_tuple_list = []
    right_tuple_list = []
    label_list = []

    for index, row in train_pairs.iterrows():
        left_tuple_list.append(left_df.loc[left_df['id'] == str(row['ltable_id'])].iloc[0]['_merged_text'])
        right_tuple_list.append(right_df.loc[right_df['id'] == str(row['rtable_id'])].iloc[0]['_merged_text'])
        label_list.append(row['label'])

    return left_tuple_list, right_tuple_list, label_list


class CTTTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dimensions=(2*AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE),
            synth_tuples_per_tuple=5, pos_to_neg_ratio=1, max_perturbation=0.4):
        super().__init__()
        self.input_dimension = EMB_DIMENSION_SIZE
        self.hidden_dimensions = hidden_dimensions
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation

        #By default, CTT uses SIF as the aggregator
        self.sif_embedding_model = SIFEmbedding()
        

    #This function is used as a preprocessing step 
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, left_df, right_df, train_pairs=None, dataset_name=None):
        print("Training CTT model")

        self.sif_embedding_model.preprocess(left_df, right_df)

        if train_pairs is None:
            # Generate syntatic training data
            print("Generate syntatic training data")
            left_df['source'] = 'table_a'
            right_df['source'] = 'table_b'
            list_of_tuples = pd.concat([left_df["_merged_text"], right_df["_merged_text"]],
                                       ignore_index=True)
            list_of_sources = pd.concat([left_df["source"], right_df["source"]],
                                       ignore_index=True)
            left_tuple_list, right_tuple_list, label_list = generate_synthetic_training_data(list_of_tuples,
                                                                                             list_of_sources,
                                                                                             dataset_name,
                                                                                             self.synth_tuples_per_tuple,
                                                                                             self.pos_to_neg_ratio,
                                                                                             self.max_perturbation)
        else:
            # Use existing training data
            print("Use existing training data")
            left_tuple_list, right_tuple_list, label_list = generate_training_data(left_df, right_df, train_pairs)

        self.left_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(left_tuple_list)
        self.right_embedding_matrix = self.sif_embedding_model.get_tuple_embedding(right_tuple_list)
        self.label_list = label_list

        trainer = dl_models.CTTModelTrainer(self.input_dimension, self.hidden_dimensions)
        self.ctt_model = trainer.train(self.left_embedding_matrix, self.right_embedding_matrix, self.label_list,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        train_data = 'self-supervised' if train_pairs is None else 'supervised'
        trainer.save_model('{}reports/{}/CTT-model-{}-{}-{}-{}.bin'.format(os.environ['DATA_DIR'],
                                                                                         dataset_name, dataset_name,
                                                                                         NUM_EPOCHS, BATCH_SIZE, train_data))
        with open('{}reports/{}/CTT-sif-{}-{}.json'.format(os.environ['DATA_DIR'], dataset_name, dataset_name, train_data), 'w') as handle:
            json.dump(dict(self.sif_embedding_model.word_to_frequencies), handle)
 

    #This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_tuples)).float()
        return embedding_matrix
        


    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(self.sif_embedding_model.get_tuple_embedding(list_of_words)).float()
        return embedding_matrix


#Hybrid is same as CTT except using Autoencoder for aggregator
class HybridTupleEmbedding(ABCTupleEmbedding):
    def __init__(self, hidden_dimensions=(2*AE_EMB_DIMENSION_SIZE, AE_EMB_DIMENSION_SIZE),
            synth_tuples_per_tuple=5, pos_to_neg_ratio=1, max_perturbation=0.4):
        super().__init__()
        self.input_dimension = EMB_DIMENSION_SIZE
        self.hidden_dimensions = hidden_dimensions
        self.synth_tuples_per_tuple = synth_tuples_per_tuple
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_perturbation = max_perturbation

        #Hybrid uses autoencoder instead of SIF aggregator 
        self.autoencoder_embedding_model = AutoEncoderTupleEmbedding()
        

    #This function is used as a preprocessing step 
    # this could be used to compute frequencies, train a DL model etc
    def preprocess(self, left_df, right_df, train_pairs=None, dataset_name=None):
        print("Training CTT model")
        #list_of_tuples = pd.concat([left_df["_merged_text"], right_df["_merged_text"]], ignore_index=True)
        self.autoencoder_embedding_model.preprocess(left_df, right_df)

        if train_pairs is None:
            # Generate syntatic training data
            print("Generate syntatic training data")
            left_df['source'] = 'table_a'
            right_df['source'] = 'table_b'
            list_of_tuples = pd.concat([left_df["_merged_text"], right_df["_merged_text"]],
                                       ignore_index=True)
            list_of_sources = pd.concat([left_df["source"], right_df["source"]],
                                        ignore_index=True)
            left_tuple_list, right_tuple_list, label_list = generate_synthetic_training_data(list_of_tuples,
                                                                                             list_of_sources,
                                                                                             dataset_name,
                                                                                             self.synth_tuples_per_tuple,
                                                                                             self.pos_to_neg_ratio,
                                                                                             self.max_perturbation)
        else:
            # Use existing training data
            print("Use existing training data")
            left_tuple_list, right_tuple_list, label_list = generate_training_data(left_df, right_df, train_pairs)

        self.left_embedding_matrix = self.autoencoder_embedding_model.get_tuple_embedding(left_tuple_list)
        self.right_embedding_matrix = self.autoencoder_embedding_model.get_tuple_embedding(right_tuple_list)
        self.label_list = label_list

        trainer = dl_models.CTTModelTrainer (self.input_dimension, self.hidden_dimensions)
        self.ctt_model = trainer.train(self.left_embedding_matrix, self.right_embedding_matrix, self.label_list,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        train_data = 'self-supervised' if train_pairs is None else 'supervised'
        trainer.save_model('{}reports/{}/HYBRID-model-{}-{}-{}-{}.bin'.format(os.environ['DATA_DIR'],
                                                                                         dataset_name, dataset_name,
                                                                                         NUM_EPOCHS, BATCH_SIZE, train_data))
        with open('{}reports/{}/HYBRID-sif-{}-{}.json'.format(os.environ['DATA_DIR'], dataset_name, dataset_name, train_data), 'w') as handle:
            json.dump(dict(self.autoencoder_embedding_model.sif_embedding_model.word_to_frequencies), handle)


    #This function computes the tuple embedding.
    # Given a list of strings, it returns a list of tuple embeddings
    # each tuple embedding is 1D numpy ndarray
    def get_tuple_embedding(self, list_of_tuples):
        embedding_matrix = torch.tensor(self.autoencoder_embedding_model.get_tuple_embedding(list_of_tuples)).float()
        return embedding_matrix
        


    #This function sends a list of words and outputs a list of word embeddings
    def get_word_embedding(self, list_of_words):
        embedding_matrix = torch.tensor(self.autoencoder_embedding_model.get_tuple_embedding(list_of_words)).float()
        return embedding_matrix