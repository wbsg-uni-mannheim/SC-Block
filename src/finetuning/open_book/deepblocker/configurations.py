import os

#FASTTEXT_EMBEDDIG_PATH = "{}embedding/wiki.en.bin".format(os.environ['DATA_DIR'])
FASTTEXT_EMBEDDIG_PATH = "/ceph/alebrink/deepblocker/embedding/wiki.en.bin"
#Dimension of the word embeddings.
EMB_DIMENSION_SIZE = 300
#Embedding size of AutoEncoder embedding
AE_EMB_DIMENSION_SIZE = 300
NUM_EPOCHS = 50 
BATCH_SIZE = 256
RANDOM_SEED = 1234
LEARNING_RATE = 1e-3
