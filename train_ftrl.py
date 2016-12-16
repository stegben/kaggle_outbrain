import sys
import pickle as pkl
import tempfile

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
    uuid = tf.contrib.layers.sparse_column_with_hash_bucket('uuid', hash_bucket_size=1e4)
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
        tf.contrib.layers.crossed_column([ad_id, uuid], hash_bucket_size=int(1e5)),
        tf.contrib.layers.crossed_column([ad_id, document_id], hash_bucket_size=int(1e5)),
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
    df_validation = data['validation']

    print('Train model...')
    model_dir = tempfile.mkdtemp()
    model_name, model = model_20161216_ver2(model_dir)
    model.fit(
        input_fn=lambda: input_fn(df_subtrain),
        steps=20,
        monitors=[tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: input_fn(df_validation),
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

if __name__ == '__main__':
    main()
