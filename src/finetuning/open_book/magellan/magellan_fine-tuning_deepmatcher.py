import logging
import os
import pickle
import time

import click
import pandas as pd
import py_entitymatching as em

from src.strategy.open_book.ranking.similarity.magellan_re_ranker import determine_path_to_feature_table, \
    determine_path_to_model


@click.command()
@click.option('--dataset')
def finetune_magellan(dataset):
    """Run finetuning for magellan"""
    logger = logging.getLogger()
    logger.info('Selected Data Set {}'.format(dataset))

    start_time = time.time()

    ds_path = '{}/deepmatcher/{}/'.format(os.environ['DATA_DIR'], dataset)
    ds_sets = {}
    sets = ['train', 'valid', 'test']
    for set in sets:
        complete_ds_path = '{}{}.csv'.format(ds_path, set)
        df_set = pd.read_csv(complete_ds_path, sep=',', encoding='utf-8')
        df_set['split'] = set
        ds_sets[set] = df_set

    df_all_sets = pd.concat(ds_sets.values())
    df_all_sets['_id'] = range(0, len(df_all_sets))
    sets = ['train', 'valid', 'test']
    for set in sets:
        ds_sets[set] = df_all_sets[df_all_sets['split'] == set]

    # Determine Feature Table
    df_tableA = pd.read_csv('{}tableA.csv'.format(ds_path), sep=',', encoding='utf-8')
    df_tableB = pd.read_csv('{}tableB.csv'.format(ds_path), sep=',', encoding='utf-8')

    # Set context attributes for matching
    context_attributes = None
    if dataset.lower() == 'abt-buy':
        context_attributes = ['name', 'price', 'description']
    elif dataset.lower() == 'amazon-google':
        context_attributes = [ 'name', 'manufacturer', 'price' ]
    elif dataset.lower() == 'dblp-acm_1':
        context_attributes = ['name', 'authors', 'venue', 'year']
    elif dataset.lower() == 'dblp-googlescholar_1':
        context_attributes = ['name', 'authors', 'venue', 'year']
    elif dataset.lower() == 'walmart-amazon_1':
        context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
    elif dataset.lower() == 'walmart-amazon_1':
        context_attributes = ['name', 'category', 'brand', 'modelno', 'price']
    elif 'wdcproducts' in dataset.lower():
        context_attributes = ['brand', 'name', 'price', 'pricecurrency', 'description']

    if dataset.lower() in ['amazon-google', 'dblp-acm_1', 'dblp-googlescholar_1', 'walmart-amazon_1'] \
            or 'wdcproducts' in dataset.lower():
        df_tableA = df_tableA.rename(columns={"title": "name"})
        df_tableB = df_tableB.rename(columns={"title": "name"})

    if 'wdcproducts' in dataset.lower():
        df_tableA.columns = [column.lower() for column in df_tableA.columns]
        df_tableB.columns = [column.lower() for column in df_tableB.columns]
        print(df_tableA.columns)
        df_tableA = df_tableA.drop(columns=['spectablecontent', 'cluster_id'])
        df_tableB = df_tableB.drop(columns=['spectablecontent', 'cluster_id'])

    # Set ID
    em.set_key(df_tableA, 'id')
    em.set_key(df_tableB, 'id')
    em.set_key(df_all_sets, '_id')

    # Set foreign key relationships
    em.set_ltable(df_all_sets, df_tableA)
    print(df_all_sets.columns)
    em.set_fk_ltable(df_all_sets, 'ltable_id')
    em.set_rtable(df_all_sets, df_tableB)
    em.set_fk_rtable(df_all_sets, 'rtable_id')

    atypes1 = em.get_attr_types(df_tableA)
    atypes2 = em.get_attr_types(df_tableB)

    block_c = em.get_attr_corres(df_tableA, df_tableB)
    block_c['corres'].remove(('id', 'id'))

    tok = em.get_tokenizers_for_matching()
    sim = em.get_sim_funs_for_matching()
    feature_table = em.get_features(df_tableA, df_tableB, atypes1, atypes2, block_c, tok, sim)
    #feature_table.to_csv('{}/magellan/tmp_feature_table.csv'.format(os.environ['DATA_DIR']), sep=';')

    # Save feature table for usage during prediction
    if not os.path.isdir('{}/magellan'.format(os.environ['DATA_DIR'])):
        os.mkdir('{}/magellan'.format(os.environ['DATA_DIR']))

    # if 'id' in context_attributes:
    #     context_attributes.remove('id')


    feature_table_path = determine_path_to_feature_table(dataset.lower(), context_attributes)
    em.save_object(feature_table, feature_table_path)
    logger.info('Saved Feature Table')

    # Use train and dev for training and evaluate using cross validation
    df_train_magellan = pd.concat([ds_sets['train'], ds_sets['valid']])
    em.set_key(df_train_magellan, '_id')
    em.set_key(ds_sets['test'], '_id')

    # Set foreign key relationships
    em.set_ltable(df_train_magellan, df_tableA)
    em.set_fk_ltable(df_train_magellan, 'ltable_id')
    em.set_rtable(df_train_magellan, df_tableB)
    em.set_fk_rtable(df_train_magellan, 'rtable_id')

    df_train_feature_vector = em.extract_feature_vecs(df_train_magellan, feature_table=feature_table,
                                                      attrs_after='label',
                                                      show_progress=True)
    # Fill missing values with 0
    df_train_feature_vector.fillna(value=0, inplace=True)

    if len(ds_sets['test'])> 0:
        em.set_ltable(ds_sets['test'], df_tableA)
        em.set_fk_ltable(ds_sets['test'], 'ltable_id')
        em.set_rtable(ds_sets['test'], df_tableB)
        em.set_fk_rtable(ds_sets['test'], 'rtable_id')

        df_test_feature_vector = em.extract_feature_vecs(ds_sets['test'], feature_table=feature_table,
                                                          attrs_after='label',
                                                          show_progress=True)
        df_test_feature_vector.fillna(value=0, inplace=True)




    # dt = em.DTMatcher(name='DecisionTree', random_state=42)
    # svm = em.SVMMatcher(name='SVM', random_state=42)
    rf = em.RFMatcher(name='RF', random_state=42)
    # lg = em.LogRegMatcher(name='LogReg', random_state=42)
    # ln = em.LinRegMatcher(name='LinReg')

    #models = [dt, rf, lg, ln]
    models = [rf]
    result = em.select_matcher(models, table=df_train_feature_vector,
                                exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                k=5,
                                target_attr='label', metric_to_select_matcher='f1', random_state=0)

    print(result['cv_stats'])

    for model in models:
        model.fit(table=df_train_feature_vector,
                  exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                  target_attr='label')
        save_model(model, dataset.lower(), context_attributes)

        if len(ds_sets['test'])> 0:
            predictions = evaluate_matcher(model, df_test_feature_vector)
            if predictions is not None:
                save_predictions(ds_sets['test'], predictions, dataset.lower(), model.name, context_attributes)

        # #evaluate_matcher(svm, schema_org_class, df_train_feature_vector, df_dev_feature_vector)
        # evaluate_matcher(rf, df_train_feature_vector, df_test_feature_vector)
        # save_predictions(df_test_magellan, predictions, schema_org_class, rf.name, context_attributes)
        #
        # evaluate_matcher(lg, df_train_feature_vector, df_test_feature_vector)
        # save_predictions(df_test_magellan, predictions, schema_org_class, lg.name, context_attributes)
        #
        # evaluate_matcher(ln, df_train_feature_vector, df_test_feature_vector)
        # save_predictions(df_test_magellan, predictions, schema_org_class, ln.name, context_attributes)
    execution_time = time.time() - start_time
    print('Execution time: {}'.format(execution_time))

def evaluate_matcher(model, df_test_feature_vector):
    # Predict on test set
    if df_test_feature_vector is not None:
        predictions = model.predict(table=df_test_feature_vector, exclude_attrs=['_id', 'ltable_id', 'rtable_id', 'label'],
                                    append=True, target_attr='predicted', inplace=False, return_probs=True,
                                    probs_attr='proba')

        print('Performance of model {} :'.format(model.name))
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        em.print_eval_summary(eval_result)
        print('')

        return predictions

    return None


def save_model(model, schema_org_class, context_attributes):
    # Save Model
    filepath_pickle = determine_path_to_model(model.name, schema_org_class, context_attributes)
    pickle.dump(model, open(filepath_pickle, 'wb'))
    logging.info('Model {} here: {}'.format(model.name, filepath_pickle))


def save_predictions(df_test_magellan, predictions, schema_org_class, model_name, context_attributes):
    # Save prediction
    df_test_magellan['pred'] = predictions['predicted']
    df_test_magellan['proba'] = predictions['proba']
    context_attribute_string = '_'.join([attr for attr in context_attributes])
    dataset_path = '{}/finetuning/open_book/magellan_results/{}_fine-tuning_magellan_{}_{}.csv'.format(os.environ['DATA_DIR'],
                                                                                      schema_org_class,
                                                                                      model_name,
                                                                                      context_attribute_string)
    df_test_magellan.to_csv(dataset_path, sep=';', encoding='utf-8')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    finetune_magellan()
