import os
import sys
sys.setrecursionlimit(10000)
import pickle as pkl
import tempfile
import itertools

from keras.models import Model
from keras.regularizers import l2 as l2_reg
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import merge
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

from utils import df2mapk


def model_20161217_ffm_v1(feature_size):
    # select features
    # fields = feature_size.keys()
    FFM_L2 = 0.00002
    FFM_DIM = 3
    fields = [
        'leak',
        # 'ad_id_fact',
        'weekday',
        'day',
        'hour',
        'geo_1',
        # 'geo_2',
        # 'geo_3',
        # 'geo_location',
        'platform',
        'advertiser_id',
        'campaign_id',
        'document_id',
    ]

    # get model
    print('Create model input')
    model_inputs = {}
    for field in fields:
        model_inputs[field] = Input(shape=(1,), dtype='int32', name='input_' + field)

    print('Create ffm layers')
    ffm_layers = []
    for field1, field2 in itertools.combinations(fields, 2):
        embed1 = Flatten()(Embedding(
            feature_size[field1] + 1,
            FFM_DIM,
            input_length=1,
            name='embed_{}_{}'.format(field1, field2),
            W_regularizer=l2_reg(FFM_L2),
        )(model_inputs[field1]))

        embed2 = Flatten()(Embedding(
            feature_size[field2] + 1,
            FFM_DIM,
            input_length=1,
            name='embed_{}_{}'.format(field2, field1),
            W_regularizer=l2_reg(FFM_L2),
        )(model_inputs[field2]))

        ffm_layers.append(merge(
            [embed1, embed2],
            mode='dot',
            dot_axes=1,
        ))
    output = Activation('sigmoid', name='output')(merge(ffm_layers, mode='sum'))
    # import ipdb; ipdb.set_trace()
    print('compile model')
    input_field = model_inputs.keys()
    model = Model(input=[model_inputs[field] for field in input_field], output=output)
    optimizer = Adagrad(lr=0.0002, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())
    return input_field, model


def model_20161217_fnn_v1(feature_size):
    # select features
    # fields = feature_size.keys()
    FFM_L2 = 0.00002
    FFM_DIM = 5
    fields = [
        'leak',
        'ad_id_fact',
        'weekday',
        'day',
        'hour',
        'geo_1',
        'geo_2',
        'geo_3',
        'geo_location',
        'platform',
        'advertiser_id',
        'campaign_id',
        'document_id',
    ]

    # get model
    print('Create model input')
    model_inputs = {}
    fnn_layers = []
    for field in fields:
        model_inputs[field] = Input(shape=(1,), dtype='int32', name='input_' + field)
        embed = Flatten()(Embedding(
            feature_size[field] + 1,
            FFM_DIM,
            input_length=1,
            name='embed_{}'.format(field),
            W_regularizer=l2_reg(FFM_L2),
        )(model_inputs[field]))
        fnn_layers.append(embed)

    concat_embed = merge(fnn_layers, mode='concat')
    dense = Dense(64, activation='relu')(concat_embed)
    dense = Dense(64, activation='relu')(dense)
    output = Dense(1, activation='sigmoid')(dense)
    # import ipdb; ipdb.set_trace()
    print('compile model')
    input_field = model_inputs.keys()
    model = Model(input=[model_inputs[field] for field in input_field], output=output)
    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())
    return input_field, model


def model_20161218_fnn_v1(feature_size):
    # select features
    # fields = feature_size.keys()
    FFM_L2 = 0.00002
    FFM_DIM = 5
    fields = [
        'leak',
        'ad_id_fact',
        'weekday',
        'day',
        'hour',
        'geo_1',
        'geo_2',
        'geo_3',
        'geo_location',
        'platform',
        'advertiser_id',
        'campaign_id',
        'document_id',
    ]

    # get model
    print('Create model input')
    model_inputs = {}
    fnn_layers = []
    for field in fields:
        model_inputs[field] = Input(shape=(1,), dtype='int32', name='input_' + field)
        embed = Flatten()(Embedding(
            feature_size[field] + 1,
            FFM_DIM,
            input_length=1,
            name='embed_{}'.format(field),
            W_regularizer=l2_reg(FFM_L2),
        )(model_inputs[field]))
        fnn_layers.append(embed)

    concat_embed = merge(fnn_layers, mode='concat')
    dense = Dense(256, activation='relu')(concat_embed)
    dense = Dense(128, activation='relu')(dense)
    dense = Dense(64, activation='relu')(dense)
    output = Dense(1, activation='sigmoid')(dense)
    # import ipdb; ipdb.set_trace()
    print('compile model')
    input_field = model_inputs.keys()
    model = Model(input=[model_inputs[field] for field in input_field], output=output)
    optimizer = Adadelta()
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())
    return input_field, model


def model_20161218_fnn_v2(feature_size):
    # select features
    # fields = feature_size.keys()
    FFM_L2 = 0.00002
    FFM_DIM = 5
    fields = [
        'leak',
        'ad_id_fact',
        'weekday',
        'day',
        'hour',
        'geo_1',
        'geo_2',
        'geo_3',
        'geo_location',
        'platform',
        'advertiser_id',
        'campaign_id',
        'document_id',
    ]

    # get model
    print('Create model input')
    model_inputs = {}
    fnn_layers = []
    for field in fields:
        model_inputs[field] = Input(shape=(1,), dtype='int32', name='input_' + field)
        embed = Flatten()(Embedding(
            feature_size[field] + 1,
            FFM_DIM,
            input_length=1,
            name='embed_{}'.format(field),
            W_regularizer=l2_reg(FFM_L2),
        )(model_inputs[field]))
        fnn_layers.append(embed)

    concat_embed = merge(fnn_layers, mode='concat')
    dense = Dropout(0.3)(Dense(256, activation='relu')(concat_embed))
    dense = Dropout(0.3)(Dense(128, activation='relu')(dense))
    dense = Dropout(0.3)(Dense(64, activation='relu')(dense))
    output = Dense(1, activation='sigmoid')(dense)
    # import ipdb; ipdb.set_trace()
    print('compile model')
    input_field = model_inputs.keys()
    model = Model(input=[model_inputs[field] for field in input_field], output=output)
    optimizer = Adadelta(lr=0.01, rho=0.9)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())
    return input_field, model


def main():
    print('Read data...')
    data_fname = sys.argv[1]
    data_fname_base = os.path.basename(os.path.abspath(data_fname))
    with open(data_fname, 'rb') as f:
        data = pkl.load(f)
    df_subtrain = data['subtrain']
    df_validation = data['validation']
    # df_subtrain = data['subtrain'].sample(frac=0.001)
    # df_validation = data['validation']
    # from random import shuffle
    # display_ids = list(df_validation['display_id'].unique())
    # shuffle(display_ids)
    # sample_display_ids = set(display_ids[:int(len(display_ids)*0.001)])
    # df_validation = df_validation[df_validation['display_id'].isin(sample_display_ids)]

    df_test = data['test']
    feature_size = data['feature_size']

    # input_field, model = model_20161217_ffm_v1(feature_size)
    # input_field, model = model_20161217_fnn_v1(feature_size)
    # input_field, model = model_20161218_fnn_v1(feature_size)
    input_field, model = model_20161218_fnn_v2(feature_size)
    model_name = model_20161218_fnn_v2.__name__

    x_subtrain = [df_subtrain[field].values for field in input_field]
    x_validation = [df_validation[field].values for field in input_field]
    x_test = [df_test[field].values for field in input_field]
    y_subtrain = df_subtrain['clicked'].values
    y_validation = df_validation['clicked'].values
    print('train model')
    model.fit(
        x_subtrain,
        y_subtrain,
        batch_size=512,
        nb_epoch=3,
        shuffle=True,
        verbose=1,
        validation_data=(x_validation, y_validation)
    )
    # train model

    # generate predictions for both train, validation and test
    df_validation['pred'] = model.predict(
        x_validation,
        batch_size=512,
        verbose=1
    ).flatten()
    score = df2mapk(df_validation[['display_id', 'ad_id', 'clicked', 'pred']])
    validation_sub_fname = '../validation_' + \
                           data_fname_base + \
                           '_' + \
                           model_name + '_' +\
                           '%.4f' % score + '.csv'
    df_validation[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(validation_sub_fname, index=False)

    df_subtrain['pred'] = model.predict(
        x_subtrain,
        batch_size=512,
        verbose=1
    ).flatten()
    subtrain_sub_fname = 'subtrain_' + \
                           data_fname_base + \
                           '_' + \
                           model_name + '_' +\
                           '%.4f' % score + '.csv'
    df_subtrain[['display_id', 'ad_id', 'clicked', 'pred']].to_csv(subtrain_sub_fname, index=False)

    df_test['pred'] = model.predict(
        x_test,
        batch_size=512,
        verbose=1
    ).flatten()
    test_sub_fname = 'test_' + \
                           data_fname_base + \
                           '_' + \
                           model_name + '_' +\
                           '%.4f' % score + '.csv'
    df_test[['display_id', 'ad_id', 'pred']].to_csv(test_sub_fname, index=False)

    # display_id, ad_id, score
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
