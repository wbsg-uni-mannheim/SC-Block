"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import csv

import click
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
import os


#### Just some code to print debug information to stdout

#### /print debug information to stdout
from src.strategy.entity_serialization import EntitySerializer


@click.command()
@click.option('--dataset')
@click.option('--model_pretrained_checkpoint')
@click.option('--pooling', default='mean')
@click.option('--loss', default='cosine')
@click.option('--epochs', default=50)
@click.option('--output_dir')
def sbert_finetuning(dataset, model_pretrained_checkpoint, pooling, loss, epochs, output_dir):

    logger = logging.getLogger()
    logger.info('Selected data set {}'.format(dataset))
    # Check if dataset exsits. If not, download and extract it

    # Read the dataset
    train_batch_size = 64
    num_epochs = epochs
    #model_save_path = '{}/models/open_book/finetuned_sbert_{}_{}_{}_dense_{}_extended_subset_pairs'.format(os.environ['DATA_DIR'], model_name.replace('/', ''),
    #                                                                                pooling, loss, dataset)
    model_save_path = output_dir
    logger.warning(model_save_path)

    # Complete model name if necessary
    if dataset in model_pretrained_checkpoint:
        model_name = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_pretrained_checkpoint)
        logging.info('Path to model: ' + model_name)

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    logging.info('Path to model: ' + model_pretrained_checkpoint)
    word_embedding_model = models.Transformer(model_pretrained_checkpoint)
    word_embedding_model.max_seq_length = 128

    if dataset not in model_pretrained_checkpoint:
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        num_added_toks = word_embedding_model.tokenizer.add_special_tokens(special_tokens_dict)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        logging.info('Added special tokens [COL], [VAL]')

    # Apply mean pooling to get one fixed sized sentence vector
    if pooling == 'mean':
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)
    elif pooling == 'cls':
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_cls_token=True,
                                       pooling_mode_max_tokens=False)
    elif pooling == 'max':
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=False,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=True)
    else:
        logger.warning('Pooling method {} not implemented'.format(pooling))

    # Dense Layer on top
    #dense_model = models.Dense(word_embedding_model.get_word_embedding_dimension(), out_features=128)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read {} dataset".format(dataset))

    record_dict = {}
    source_field = '__source'
    left_source = 'table_a'
    right_source = 'table_b'

    path_to_data = f'{os.environ["DATA_DIR"]}/deepmatcher/{dataset}/'
    entity_serializer = EntitySerializer(dataset)
    # Load table A and table B
    with open(f'{path_to_data}/tableA.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            record['id'] = f'{left_source}-{record["id"]}'
            if dataset in ['amazon-google', 'dblp-acm_1', 'dblp-acm_2', 'dblp-googlescholar_1',
                           'dblp-googlescholar_2', 'walmart-amazon_1', 'walmart-amazon_2']:
                record['name'] = record.pop('title')
            #record['title'] = record.pop('name')
            record[source_field] = left_source
            string_representation = entity_serializer.convert_to_str_representation(record)
            # del record['description']  # drop description, for benchmarking
            record_dict[record['id']] = string_representation

    print(f'Loaded {len(record_dict)} records')

    with open(f'{path_to_data}/tableB.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
                record['id'] = f'{right_source}-{record["id"]}'
                #record['title'] = record.pop('name')
                record[source_field] = right_source
                string_representation = entity_serializer.convert_to_str_representation(record)
                #del record['description']  # drop description, for benchmarking
                record_dict[record['id']] = string_representation
    print(f'Loaded {len(record_dict)} records')


    train_samples = []
    dev_samples = []
    # test_samples = []

    # Use both train+valid for training
    with open(f'{path_to_data}/train.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict['{}-{}'.format(left_source, record['ltable_id'])]
            right_record = record_dict['{}-{}'.format(right_source, record['rtable_id'])]

            score = float(record['label'])
            inp_example = InputExample(texts=[left_record, right_record], label=score)
            train_samples.append(inp_example)

    #print(f'Loaded {len(train_samples)} train samples')

    with open(f'{path_to_data}/valid.csv', newline='', encoding="utf-8") as f:
        for record in csv.DictReader(f):
            left_record = record_dict['{}-{}'.format(left_source, record['ltable_id'])]
            right_record = record_dict['{}-{}'.format(right_source, record['rtable_id'])]

            score = float(record['label'])
            inp_example = InputExample(texts=[left_record, right_record], label=score)
            train_samples.append(inp_example)

    print(f'Loaded {len(train_samples)} train samples')



    # df_valid = pd.read_csv(valid_dataset_path, sep=';', encoding='utf-8')
    # for index, row in df_valid.iterrows():
    #     score = float(row['score'])
    #     inp_example = InputExample(texts=[row['entity1'], row['entity2']], label=score)
    #     dev_samples.append(inp_example)
    #
    #     # if row['split'] == 'dev':
    #     #     dev_samples.append(inp_example)
    #     # # elif row['split'] == 'test':
    #     # #    test_samples.append(inp_example)
    #     # else:


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    if loss == 'cosine':
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif loss == 'contrastive':
        train_loss = losses.ContrastiveLoss(model=model)
    elif loss == 'online_contrastive':
        train_loss = losses.OnlineContrastiveLoss(model=model)
    else:
        logger.warning('Loss is not defined!')
    #
    logging.info("Read {} Training dev dataset".format(dataset))
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='{}-dev'.format(dataset))

    # Configure the training. We skip evaluation in this example

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              #evaluator=evaluator,
              epochs=num_epochs,
              #evaluation_steps=1000,
              warmup_steps=warmup_steps,
              #save_best_model=True,
              output_path=model_save_path)

    #model.save(model_save_path)

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    # model = SentenceTransformer(model_save_path)
    # test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
    # test_evaluator(model, output_path=model_save_path)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_fmt,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    sbert_finetuning()
