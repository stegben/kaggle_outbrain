import sys
import csv
import pickle as pkl
import tempfile
from datetime import datetime
from random import shuffle

import numpy as np
import pandas as pd
import joblib

memory = joblib.Memory(cachedir='../cache', verbose=1)

NAN_VALUE = '0000'


@memory.cache
def get_leak_document_id_uuid_set(leak_fname):
    leak_uuid_dict= {}
    with open(leak_fname) as infile:
        infile.readline()
        for ind, row in enumerate(infile):
            row = row.rstrip().split(',')
            doc_id = row[0]
            leak_uuid_dict[doc_id] = set(row[1].split(' '))
            if ind % 10000 == 0:
                print("Leakage file : ", ind)
        print(len(leak_uuid_dict))
        print(leak_uuid_dict.keys())
    leak_set = set()
    for idx, (doc_id, uuid_set) in enumerate(leak_uuid_dict.items()):
        for uuid in uuid_set:
            leak_set.add(doc_id+'_'+uuid)
        if idx % 10000 == 0:
            print("Leakage doc ids : ", idx)
    return leak_set


@memory.cache
def generate_data(leak_fname):
    print('Read data')
    df_clicks_train = pd.read_csv('../clicks_train.csv', engine='c', dtype=str)
    df_clicks_test = pd.read_csv('../clicks_test.csv', engine='c', dtype=str)

    # df_document_categories = pd.read_csv('../documents_categories.csv', engine='c', dtype=str)
    # df_document_entities = pd.read_csv('../documents_entities.csv', engine='c', dtype=str)
    # df_document_meta = pd.read_csv('../documents_meta.csv', engine='c', dtype=str)
    # df_document_topics = pd.read_csv('../documents_topics.csv', engine='c', dtype=str)

    df_events = pd.read_csv('../events.csv', engine='c', dtype=str)
    df_events = pd.concat([
        df_events,
        df_events.geo_location.str.split('>', expand=True)\
                 .rename(columns={0: 'geo_1', 1: 'geo_2', 2: 'geo_3'})
    ], axis=1)
    df_promoted_content = pd.read_csv('../promoted_content.csv', engine='c', dtype=str)

    print("Read leakage file..")
    leak_document_id_uuid_set = get_leak_document_id_uuid_set(leak_fname)

    print('Start joining tables and create Features')
    df_clicks = pd.concat([df_clicks_train, df_clicks_test], axis=0)
    df_clicks = pd.merge(df_clicks, df_events, on='display_id', how='left')
    print('====== create leak feature')
    # temp_uuid = df_clicks['document_id'].apply(lambda x: leak_uuid_dict.get(x, set()))
    df_clicks['leak'] = (df_clicks['document_id'] + '_' + df_clicks['uuid']).isin(leak_document_id_uuid_set)
    # df_clicks['leak'] = df_clicks['document_id'].isin(leak_uuid_dict) * \
    #                     df_clicks['uuid'].isin(
    #                         df_clicks['document_id'].apply(lambda x: leak_uuid_dict.get(x, set()))
    #                     )
    # import ipdb; ipdb.set_trace()
    print('====== create other features')
    df_clicks = pd.merge(df_clicks, df_promoted_content.drop('document_id', axis=1), on='ad_id', how='left')
    df_clicks = df_clicks[df_clicks.platform != '\\N'].fillna(NAN_VALUE)
    df_clicks['timestamp'] = df_clicks['timestamp'].astype(int)
    df_clicks['hour'] = df_clicks['timestamp'] // (3600*1000) % 24
    df_clicks['day'] = df_clicks['timestamp'] // (24*3600*1000)
    df_clicks['weekday'] = df_clicks['timestamp'] // (24*3600*1000) % 7

    print('Start Factorizing')
    for col in df_clicks:
        if col in ['clicked', 'display_id', 'timestamp']:
            continue
        elif col == 'ad_id':
            df_clicks['ad_id_fact'] = (df_clicks['ad_id'].factorize()[0].astype(np.int32) % 2e5).astype(np.int32)
            continue
        elif col == 'document_id':
            df_clicks[col] = (df_clicks[col].factorize()[0].astype(np.int32) % 2e5).astype(np.int32)
            continue
        elif col == 'uuid':
            df_clicks[col] = (df_clicks[col].factorize()[0].astype(np.int32) % 5e6).astype(np.int32)
            continue
        else:
            df_clicks[col] = df_clicks[col].factorize()[0].astype(np.int32)
    feature_size = df_clicks.drop(['clicked', 'display_id', 'timestamp', 'ad_id'], axis=1, inplace=False) \
                            .max() \
                            .to_dict()
    return feature_size, df_clicks


def main():
    data_fname = sys.argv[1]
    leak_fname = '../leak.csv'

    feature_size, df_clicks = generate_data(leak_fname)
    # import ipdb; ipdb.set_trace()
    df_train = df_clicks[df_clicks['clicked'] != NAN_VALUE]
    df_test = df_clicks[df_clicks['clicked'] == NAN_VALUE].drop(['clicked', 'timestamp'], axis=1)

    print('Split data')
    max_time = df_train['timestamp'].astype(int).max()
    df_subtrain = df_train[df_train['timestamp'].astype(int) < (max_time * 0.9)].drop('timestamp', axis=1)
    df_validation = df_train[df_train['timestamp'].astype(int) >= (max_time * 0.9)].drop('timestamp', axis=1)

    subtrain_display_ids = list(df_subtrain['display_id'].unique())
    shuffle(subtrain_display_ids)
    sample_display_ids = set(subtrain_display_ids[:int(len(subtrain_display_ids)*0.1)])

    df_subtrain, df_validation = \
        df_subtrain[~df_subtrain['display_id'].isin(sample_display_ids)], \
        pd.concat([df_subtrain[df_subtrain['display_id'].isin(sample_display_ids)], df_validation], axis=0)
    # msk = np.random.rand(len(df_subtrain)) < 0.9
    # df_subtrain, df_validation = \
    #     df_subtrain[msk], \
    #     pd.concat([df_subtrain[~msk], df_validation], axis=0)

    # import ipdb; ipdb.set_trace()
    print('Store data')
    data = {}
    data['subtrain'] = df_subtrain
    data['validation'] = df_validation
    data['test'] = df_test
    data['feature_size'] = feature_size
    with open(data_fname, 'wb') as fw:
        pkl.dump(data, fw)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
