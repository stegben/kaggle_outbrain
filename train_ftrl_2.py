import sys
import pickle as pkl
# import tempfile
import pandas as pd
from sklearn.datasets import load_svmlight_file

from kaggler.online_model import FTRL

from utils import df2mapk

EPOCH_NUM = 2

FTRL_PARAM = \
    {'a': 0.1, 'b': 1, 'l1': 0.1, 'l2': 0.001, 'n': 2**24, 'epoch': EPOCH_NUM, 'interaction': True}

def main():
    data_fname_base = sys.argv[1]
    subtrain_fname = '../subtrain_' + data_fname_base
    subtrain_id_fname = subtrain_fname + '.id'
    validation_fname = '../validation_' + data_fname_base
    validation_id_fname = validation_fname + '.id'
    test_fname = '../test_' + data_fname_base
    test_id_fname = test_fname + '.id'

    best_param = FTRL_PARAM
    clf = FTRL(**best_param)    # use feature interaction or not

    print('Read validation data...')
    x_validation, y_validation = load_svmlight_file(validation_fname)
    df_validation = pd.read_csv(validation_id_fname, dtype=str)
    # df_validation['clicked'] = 0
    # df_validation['pred'] = 0.
    df_validation['clicked'] = y_validation
    df_validation['clicked'] = df_validation['clicked'].astype(int).astype(str)

    print('Start Training...')
    for epoch in range(EPOCH_NUM):
        for idx, (x_subtrain_1, y_subtrain_1) in enumerate(clf.read_sparse(subtrain_fname)):
            x_subtrain_1 = [int(ind) for ind in x_subtrain_1]
            pred_subtrain_1 = clf.predict_one(x_subtrain_1)
            clf.update_one(x_subtrain_1, pred_subtrain_1 - y_subtrain_1)
            if idx % 1000000 == 0:
                print('epoch {} read {} data'.format(epoch, idx))

        print('validation')
        pred_validation = clf.predict(x_validation)
        df_validation['pred'] = pred_validation
        # for idx, (x_validation_1, y_validation_1) in enumerate(clf.read_sparse(validation_fname)):
        #     x_validation_1 = [int(ind) for ind in x_validation_1]
        #     pred_validation_1 = clf.predict_one(x_validation_1)
        #     df_validation['pred'][idx] = pred_validation_1
        #     df_validation['clicked'][idx] = y_validation_1
        #     if idx % 3000000 == 0:
        #         print('validation epoch {} read {} data'.format(epoch, idx))
        validation_score = df2mapk(df_validation)
        print('Validation score: %.5f' % validation_score)
    best_score = validation_score

    param_name = '__'.join([k + '-' + str(v) for k, v in best_param.items()])
    temp_sub_fname = data_fname_base \
                     +  '_ftrl_' \
                     + param_name + '__' \
                     + '%.4f' % best_score + '.csv'
    print('generate validation result')
    validation_sub_fname = '../validation_result/' + temp_sub_fname
    df_validation[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(validation_sub_fname, index=False)

    print('generate test result')
    x_test, y_test = load_svmlight_file(test_fname)
    df_test = pd.read_csv(test_id_fname, dtype=str)
    # for idx, (x_test_1, y_test_1) in enumerate(clf.read_sparse(test_fname)):
    #     x_test_1 = [int(ind) for ind in x_test_1]
    #     pred_test_1 = clf.predict_one(x_test_1)
    #     df_test['pred'][idx] = pred_test_1
    #     df_test['clicked'][idx] = y_test_1
    pred_test = clf.predict(x_test)
    df_test['pred'] = pred_test
    test_sub_fname = '../test_result/' + temp_sub_fname
    df_test[['display_id', 'ad_id', 'pred']].to_csv(test_sub_fname, index=False)

"""
import tensorflow as tf

CATEGORICAL_COLUMNS = [
    'ad_id',
    'uuid',
    'document_id',
    'platform',
    'geo_location',
    'campaign_id',
    'advertiser_id',
]


def input_fn(df):
    label = tf.constant(df['clicked'].values)
    real_cols = {}
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
    feature_cols = {**real_cols, **categorical_cols}
    return feature_cols, label


def model_20161216_ver1(model_dir):
    ad_id = tf.contrib.layers.sparse_column_with_hash_bucket('ad_id', hash_bucket_size=1e6)
    uuid = tf.contrib.layers.sparse_column_with_hash_bucket('uuid', hash_bucket_size=1e8)
    document_id = tf.contrib.layers.sparse_column_with_hash_bucket('document_id', hash_bucket_size=1e6)
    platform = tf.contrib.layers.sparse_column_with_keys(column_name='platform', keys=['1', '2', '3'])
    geo_location = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='geo_location', hash_bucket_size=1e4)
    campaign_id = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='campaign_id', hash_bucket_size=1e5)
    advertiser_id = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='advertiser_id', hash_bucket_size=1e4)

    feature_columns = [
        ad_id,
        uuid,
        document_id,
        platform,
        geo_location,
        campaign_id,
        advertiser_id,
        tf.contrib.layers.crossed_column([ad_id, uuid], hash_bucket_size=int(1e12)),
        tf.contrib.layers.crossed_column([ad_id, document_id], hash_bucket_size=int(1e11)),
        tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
        tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
    ]

    model = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns,
        optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0),
    model_dir=model_dir)
    return 'model_20161216_ver1', model


def model_20161216_ver2(model_dir):
    ad_id = tf.contrib.layers.sparse_column_with_hash_bucket('ad_id', hash_bucket_size=1e3)
    uuid = tf.contrib.layers.sparse_column_with_hash_bucket('uuid', hash_bucket_size=1e3)
    document_id = tf.contrib.layers.sparse_column_with_hash_bucket('document_id', hash_bucket_size=1e3)
    platform = tf.contrib.layers.sparse_column_with_keys(column_name='platform', keys=['1', '2', '3'])
    geo_location = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='geo_location', hash_bucket_size=1e3)
    campaign_id = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='campaign_id', hash_bucket_size=1e3)
    advertiser_id = tf.contrib.layers.sparse_column_with_hash_bucket(column_name='advertiser_id', hash_bucket_size=1e3)

    feature_columns = [
        ad_id,
        uuid,
        document_id,
        platform,
        geo_location,
        campaign_id,
        advertiser_id,
        tf.contrib.layers.crossed_column([ad_id, uuid], hash_bucket_size=int(1e3)),
        tf.contrib.layers.crossed_column([ad_id, document_id], hash_bucket_size=int(1e3)),
    ]
    model = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns,
        optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0),
    model_dir=model_dir)
    return 'model_20161216_ver2', model


def main():
    print('Read data...')
    data_fname = sys.argv[1]
    with open(data_fname, 'rb') as f:
        data = pkl.load(f)
    df_subtrain = data['subtrain']
    df_validationdata['validation']

    print('Train model...')
    model_dir = tempfile.mkdtemp()
    model_name, model = model_20161216_ver2(model_dir)
    model.fit(
        input_fn=lambda: input_fn(df_subtrain.sample(frac=0.001)),
        steps=20,
        monitors=[tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: input_fn(df_validation.sample(frac=0.001)),
            every_n_steps=1)],
    )

    print('Validate')
    result = model.evaluate(
        input_fn=lambda: input_fn(df_validation),
        steps=1,
    )
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    import ipdb; ipdb.set_trace()
"""
if __name__ == '__main__':
    main()
