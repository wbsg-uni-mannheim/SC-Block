import logging
import math
import os

import click
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

from src.finetuning.open_book.cross_encoder.MatchingDataset import MatchingDataset
from src.strategy.entity_serialization import EntitySerializer
from src.strategy.open_book.ranking.similarity.hf_re_ranker import determine_path_to_model


@click.command()
@click.option('--schema_org_class')
@click.option('--model_name')
@click.option('--context_attribute_selection', type=int)
@click.option('--epochs', default=50)
@click.option('--only_predict/--not_only_predict', default=False)
@click.option('--local_rank', type=int, default=-1)
def finetune_cross_encoder(schema_org_class, model_name, context_attribute_selection, epochs, only_predict, local_rank):
    """Run cross encoder finetuning"""
    logger = logging.getLogger()
    logger.info('Selected schema org class {}'.format(schema_org_class))

    # Load pre-trained model (on data from schema_org class)
    if schema_org_class in model_name:
        model_name = '{}/models/open_book/{}'.format(os.environ['DATA_DIR'], model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if schema_org_class not in model_name:
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': ['[COL]', '[VAL]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        logger.info('Added special tokens [COL], [VAL]')

    # Read data sets line by line (2 subsequent lines describe the same entity
    logging.info("Read {} dataset".format(schema_org_class))

    dataset_path = '{}/finetuning/open_book/{}_fine-tuning_cross_extended_subset_pairs_more_negatives.csv'.format(os.environ['DATA_DIR'], schema_org_class)

    df_cross_encoder = pd.read_csv(dataset_path, sep=';', encoding='utf-8')

    if schema_org_class == 'localbusiness':
        context_attributes = ['name', 'addresslocality']
    elif schema_org_class == 'product':
        context_attributes = ['name']
    else:
        logger.warning('Option {} is not defined!'.format(schema_org_class))
        context_attributes = []


    # # Preprocess entities:
    df_cross_encoder['entities'] = df_cross_encoder['entity1'] + '[SEP]' + df_cross_encoder['entity2']
    # df_cross_encoder['entities'] = df_cross_encoder.apply(create_cross_encoder_entity_input,
    #                                                       args=(schema_org_class, context_attributes,), axis=1)


    df_train_cross_encoder = df_cross_encoder[df_cross_encoder['split'] == 'train']
    df_dev_cross_encoder = df_cross_encoder[df_cross_encoder['split'] == 'dev']
    # df_test_cross_encoder = df_cross_encoder[df_cross_encoder['split'] == 'test']
    # print(len(df_test_cross_encoder))


    train_encodings = tokenizer(list(df_train_cross_encoder['entities'].values), return_tensors='pt', padding=True,
                                truncation=True)
    dev_encodings = tokenizer(list(df_dev_cross_encoder['entities'].values), return_tensors='pt', padding=True,
                              truncation=True)
    # test_encodings = tokenizer(list(df_test_cross_encoder['entities'].values), return_tensors='pt', padding=True,
    #                            truncation=True)

    train_dataset = MatchingDataset(train_encodings,
                                    [int(value) for value in list(df_train_cross_encoder['score'])])
    val_dataset = MatchingDataset(dev_encodings, [int(value) for value in list(df_dev_cross_encoder['score'])])
    #test_dataset = MatchingDataset(test_encodings, [int(value) for value in list(df_test_cross_encoder['score'])])

    num_epochs = epochs
    warmup_steps = math.ceil(len(train_dataset) * num_epochs * 0.1)
    model_save_path = determine_path_to_model(model_name, schema_org_class, context_attributes)

    def compute_metrics(eval_preds):
        metric = load_metric("f1")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=model_save_path,  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
        #ddp_find_unused_parameters=False,
        #gradient_accumulation_steps=2
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        # data_collator=similarity_data_collator,
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    if not only_predict:
        trainer.train()

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    # eval_results = trainer.evaluate(test_dataset)
    # logger.info('Evaluation result on Test set')
    # logger.info(eval_results)
    #
    # predictions = trainer.predict(test_dataset)
    # preds = np.argmax(predictions.predictions, axis=-1)
    #
    # # Save predictions
    # df_test_cross_encoder['predictions'] = preds

    # if schema_org_class in model_name:
    #     original_model_name = model_name.split('/')[-1]
    #     dataset_path = '{}/finetuning/open_book/{}_prediction_{}.csv'.format(
    #         os.environ['DATA_DIR'],
    #         schema_org_class,
    #         original_model_name
    #         )
    # else:
    #     context_attribute_string = '_'.join(context_attributes)
    #     dataset_path = '{}/finetuning/open_book/{}_fine-tuning_cross_encoder_prediction_{}_{}.csv'.format(
    #         os.environ['DATA_DIR'],
    #         schema_org_class,
    #         model_name,
    #         context_attribute_string)
    #df_test_cross_encoder.to_csv(dataset_path, sep=';', encoding='utf-8', index=None)


def create_cross_encoder_entity_input(row, schema_org_class, context_attributes):
    """"""
    # Extract entity values
    entity1 = {index.replace('entity1_', ''): value for index, value in row.iteritems() if 'entity1_' in index}
    entity2 = {index.replace('entity2_', ''): value for index, value in row.iteritems() if 'entity2_' in index}
    entity_serializer = EntitySerializer(schema_org_class, context_attributes)

    representation = entity_serializer.convert_to_cross_encoder_representation(entity1, entity2)
    print(representation)

    return representation


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    finetune_cross_encoder()
