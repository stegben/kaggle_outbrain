import sys
import gc
import csv
import pickle as pkl
import tempfile
from datetime import datetime
import random
random.seed(1234)
from random import shuffle

import numpy as np
import pandas as pd
import joblib

memory = joblib.Memory(cachedir='../cache', verbose=1)

NAN_VALUE = '-1'


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
    print('Read clicks data')
    df_clicks_train = pd.read_csv('../clicks_train.csv', engine='c', dtype={'display_id': str, 'ad_id': int, 'clicked': str})
    df_clicks_test = pd.read_csv('../clicks_test.csv', engine='c', dtype={'display_id': str, 'ad_id': int})

    print('Read event data')
    df_events = pd.read_csv('../events.csv', engine='c', dtype={'display_id': str, 'ad_id': int, 'uuid': str, 'timestamp': int, 'platform': str, 'geo_location': str})
    # display2uuid = df_events.groupby('uuid', sort=False, as_index=False)['display_id'].filter(lambda x: len(x) < 3).set_index('')
    df_events = pd.concat([
        df_events,
        df_events.geo_location.str.split('>', expand=True)\
                 .rename(columns={0: 'geo_1', 1: 'geo_2', 2: 'geo_3'})
    ], axis=1)
    df_events['geo_1__geo_2'] = df_events['geo_1'] + '__' + df_events['geo_2']
    # df_events.drop('uuid', axis=1, inplace=True)

    print('Read promoted_content')
    df_promoted_content = pd.read_csv('../promoted_content.csv', engine='c', dtype={'document_id': int, 'campaign_id': int, 'advertiser_id': int, 'ad_id': int})

    print('Read document topics')
    df_document_topics = pd.read_csv('../documents_topics.csv', engine='c', dtype={'document_id': int, 'topic_id': int, 'confidence_level': float})
    df_document_topics.sort_values(['document_id', 'confidence_level'], inplace=True, ascending=False)
    first_topic_dict = \
        df_document_topics.groupby('document_id', sort=False)['topic_id'] \
                          .nth(0) \
                          .to_dict()
    second_topic_dict = \
        df_document_topics.groupby('document_id', sort=False)['topic_id'] \
                          .nth(1) \
                          .to_dict()
    topic_num_dict = \
        df_document_topics.groupby('document_id', sort=False)['topic_id'] \
                          .agg(len) \
                          .to_dict()
    del df_document_topics

    df_document_entities = pd.read_csv('../documents_entities.csv', engine='c', dtype={'document_id': int, 'entity_id': str, 'confidence_level': float})
    df_document_entities.sort_values(['document_id', 'confidence_level'], inplace=True, ascending=False)
    df_document_entities['entity_id'] = df_document_entities.entity_id.factorize()[0].astype(int)
    first_entity_dict = \
        df_document_entities.groupby('document_id', sort=False)['entity_id'] \
                          .nth(0) \
                          .to_dict()
    entity_num_dict = \
        df_document_entities.groupby('document_id', sort=False)['entity_id'] \
                          .agg(len) \
                          .to_dict()
    del df_document_entities

    print('Read document category')
    df_document_categories = \
        pd.read_csv('../documents_categories.csv',
                    engine='c',
                    dtype={'document_id': int, 'category_id': int, 'confidence_level': float})
    df_document_categories.sort_values(['document_id', 'confidence_level'], inplace=True, ascending=False)
    first_category_dict = \
        df_document_categories.groupby('document_id', sort=False)['category_id'] \
                              .nth(0) \
                              .to_dict()
    second_category_dict = \
        df_document_categories.groupby('document_id', sort=False)['category_id'] \
                              .nth(1) \
                              .to_dict()
    del df_document_categories

    # too many entitis, useless
    # df_document_entities = pd.read_csv('../documents_entities.csv', engine='c', dtype=str)
    print('Read document meta')
    df_document_meta = pd.read_csv('../documents_meta.csv', engine='c', dtype={'document_id': int, 'source_id': str, 'publisher_id': str}, usecols=['document_id', 'source_id', 'publisher_id'])
    df_document_meta.fillna(0, inplace=True)
    df_document_meta = df_document_meta.astype(int)
    df_document_meta['source_id'] = df_document_meta['source_id'] + 1
    df_document_meta['publisher_id'] = df_document_meta['publisher_id'] + 1
    doc2source = df_document_meta.set_index('document_id')['source_id'].to_dict()
    doc2publisher = df_document_meta.set_index('document_id')['publisher_id'].to_dict()
    del df_document_meta


    print('Start joining tables and create Features')
    df_clicks = pd.concat([df_clicks_train, df_clicks_test], axis=0)
    df_clicks = pd.merge(df_clicks, df_events, on='display_id', how='left')
    # df_clicks['uuid'] = df_clicks.display_id.map(display2uuid).fillna(-1)
    df_clicks = pd.merge(df_clicks, df_promoted_content, on='ad_id', how='left', suffixes=('_orig', '_ad'))
    del df_events
    del df_promoted_content
    df_clicks['document_id_orig_source'] = df_clicks.document_id_orig.map(doc2source).fillna(-1) + 2 # max: 14406
    df_clicks['document_id_ad_source'] = df_clicks.document_id_ad.map(doc2source).fillna(-1) + 2 # max: 14407
    # df_clicks['source_doc_id_ad__orig'] = df_clicks.document_id_ad_source * 20000 + df_clicks.document_id_orig_source
    df_clicks['document_id_orig_publisher'] = df_clicks.document_id_orig.map(doc2publisher).fillna(-1) + 2 # max: 1266
    df_clicks['document_id_ad_publisher'] = df_clicks.document_id_ad.map(doc2publisher).fillna(-1) + 2 # max:1255
    df_clicks['publisher_doc_id_ad__orig'] = df_clicks.document_id_ad_publisher * 2000 + df_clicks.document_id_orig_publisher
    # df_clicks.drop(['document_id_orig_publisher', 'document_id_ad_publisher'], axis=1, inplace=True)

    df_clicks['document_id_ad_first_category'] = df_clicks.document_id_ad.map(first_category_dict).fillna(-1) + 2 # max: 2102
    df_clicks['document_id_orig_first_category'] = df_clicks.document_id_orig.map(first_category_dict).fillna(-1) + 2 # max: 2102
    df_clicks['document_id_ad_second_category'] = df_clicks.document_id_ad.map(second_category_dict).fillna(-1) + 2 # max: 2102
    df_clicks['document_id_orig_second_category'] = df_clicks.document_id_orig.map(second_category_dict).fillna(-1) + 2 # max: 2102
    df_clicks['first_category_doc_id_ad__orig'] = df_clicks.document_id_ad_first_category * 3000 + df_clicks.document_id_orig_first_category
    df_clicks['second_category_doc_id_ad__orig'] = df_clicks.document_id_ad_second_category * 3000 + df_clicks.document_id_orig_second_category
    # df_clicks.drop(['document_id_orig_first_category', 'document_id_ad_first_category'], axis=1, inplace=True)
    df_clicks['ad_doc_id__first_second_category'] = df_clicks.document_id_ad_first_category * 3000 + df_clicks.document_id_ad_second_category
    df_clicks['orig_doc_id__first_second_category'] = df_clicks.document_id_orig_first_category * 3000 + df_clicks.document_id_orig_second_category
    # df_clicks.drop(['document_id_orig_second_category', 'document_id_ad_second_category'], axis=1, inplace=True)

    df_clicks['document_id_orig_first_topic'] = df_clicks.document_id_orig.map(first_topic_dict).fillna(-1) + 2
    df_clicks['document_id_orig_second_topic'] = df_clicks.document_id_orig.map(second_topic_dict).fillna(-1) + 2
    df_clicks['document_id_ad_first_topic'] = df_clicks.document_id_ad.map(first_topic_dict).fillna(-1) + 2
    df_clicks['document_id_ad_second_topic'] = df_clicks.document_id_ad.map(second_topic_dict).fillna(-1) + 2
    df_clicks['first_topic_doc_id_ad__orig'] = df_clicks.document_id_ad_first_topic * 3000 + df_clicks.document_id_orig_first_topic
    df_clicks['second_topic_doc_id_ad__orig'] = df_clicks.document_id_ad_second_topic * 3000 + df_clicks.document_id_orig_second_topic
    df_clicks['ad_doc_id__first_second_topic'] = df_clicks.document_id_ad_first_topic * 3000 + df_clicks.document_id_ad_second_topic
    df_clicks['orig_doc_id__first_second_topic'] = df_clicks.document_id_orig_first_topic * 3000 + df_clicks.document_id_orig_second_topic
    # df_clicks.drop(['document_id_orig_first_topic', 'document_id_ad_first_topic'], axis=1, inplace=True)
    # df_clicks.drop(['document_id_orig_second_topic', 'document_id_ad_second_topic'], axis=1, inplace=True)
    df_clicks['document_id_orig_topic_num'] = df_clicks.document_id_orig.map(topic_num_dict).fillna(-1) + 2
    df_clicks['document_id_ad_topic_num'] = df_clicks.document_id_ad.map(topic_num_dict).fillna(-1) + 2

    # df_clicks['document_id_orig_first_entity'] = df_clicks.document_id_orig.map(first_entity_dict).fillna(-1) + 2
    # df_clicks['document_id_ad_first_entity'] = df_clicks.document_id_ad.map(first_entity_dict).fillna(-1) + 2
    # df_clicks['first_entity_doc_id_ad__orig'] = (df_clicks.document_id_ad_first_entity * 3000 + df_clicks.document_id_orig_first_entity) % 1e6
    # df_clicks.drop(['document_id_orig_first_entity', 'document_id_ad_first_entity'], axis=1, inplace=True)
    # df_clicks.drop(['document_id_orig_second_entity', 'document_id_ad_second_entity'], axis=1, inplace=True)
    df_clicks['document_id_orig_entity_num'] = df_clicks.document_id_orig.map(entity_num_dict).fillna(-1) + 2
    # df_clicks['document_id_ad_entity_num'] = df_clicks.document_id_ad.map(entity_num_dict).fillna(-1) + 2

    df_clicks['ad_count'] = df_clicks.display_id.map(df_clicks.groupby('display_id', sort=False)['ad_id'].count().to_dict()).factorize()[0].astype(np.int32)
    df_clicks['ad_id_appear_time'] = pd.cut(df_clicks.ad_id.map(df_clicks.groupby('ad_id', sort=False)['display_id'].count().to_dict()).apply(np.log), bins=200).factorize()[0].astype(np.int32)
    df_clicks['document_id_ad_appear_time'] = pd.cut(df_clicks.document_id_ad.map(df_clicks.groupby('document_id_ad', sort=False)['display_id'].count().to_dict()).apply(np.log), bins=150).factorize()[0].astype(np.int32)
    df_clicks['document_id_orig_appear_time'] = pd.cut(df_clicks.document_id_orig.map(df_clicks.groupby('document_id_orig', sort=False)['display_id'].count().to_dict()).apply(np.log), bins=150).factorize()[0].astype(np.int32)

    print("Read leakage file...")
    leak_document_id_uuid_set = get_leak_document_id_uuid_set(leak_fname)
    print('====== create leak feature')
    df_clicks['leak'] = (df_clicks['document_id_ad'].astype(str) + '_' + df_clicks['uuid'].astype(str)).isin(leak_document_id_uuid_set)
    del leak_document_id_uuid_set
    del first_category_dict
    del second_category_dict
    del first_topic_dict
    del second_topic_dict
    del topic_num_dict
    del first_entity_dict
    del entity_num_dict
    gc.collect()
    print('====== create other features')
    df_clicks.fillna(NAN_VALUE, inplace=True)
    print('========= process time feature')
    df_clicks['timestamp'] = df_clicks['timestamp'].astype(int)
    df_clicks['timeslot'] = df_clicks['timestamp'] // (3600*1000)
    df_clicks['ad_doc_id_timeslot'] = (df_clicks.document_id_ad * 1000 + df_clicks.timeslot).astype(int)
    df_clicks['hour'] = (df_clicks['timestamp'] // (3600*1000) % 24).astype(np.uint8)
    df_clicks['day'] = (df_clicks['timestamp'] // (24*3600*1000)).astype(np.uint8)
    df_clicks['weekday'] = (df_clicks['timestamp'] // (24*3600*1000) % 7).astype(np.uint8)
    df_clicks['weekday__hour'] = df_clicks.hour * 10 + df_clicks.weekday
    with open('../doc_view_times.pkl', 'rb') as f:
        doc_views_map = pkl.load(f)
    with open('../doc_hour_view_times.pkl', 'rb') as f:
        doc_hour_views_map = pkl.load(f)
    # import ipdb; ipdb.set_trace()
    df_clicks['doc_views_map'] = pd.cut(df_clicks.document_id_ad.map(doc_views_map).apply(np.log), bins=100).cat.add_categories(['a']).fillna('a')
    df_clicks['doc_orig_views_map'] = pd.cut(df_clicks.document_id_orig.map(doc_views_map).apply(np.log), bins=100).cat.add_categories(['a']).fillna('a')
    df_clicks['doc_hour_views_map'] = pd.cut(df_clicks.ad_doc_id_timeslot.map(doc_hour_views_map).apply(np.log), bins=100).cat.add_categories(['a']).fillna('a')
    df_clicks.drop(['timeslot', 'ad_doc_id_timeslot'], axis=1, inplace=True)
    # import ipdb; ipdb.set_trace()
    print('Start Factorizing')
    for col in df_clicks:
        if col in ['clicked', 'display_id', 'timestamp']:
            continue
        elif col == 'ad_id':
            # df_clicks['ad_id_fact'] = (df_clicks['ad_id'].factorize()[0].astype(np.int32) % 2e5).astype(np.int32)
            df_clicks['ad_id_fact'] = df_clicks['ad_id'].factorize()[0].astype(np.int32)
            continue
        elif col == 'document_id':
            # df_clicks[col] = (df_clicks[col].factorize()[0].astype(np.int32) % 2e5).astype(np.int32)
            df_clicks[col] = df_clicks[col].factorize()[0].astype(np.int32)
            continue
        elif col == 'uuid':
            # df_clicks[col] = (df_clicks[col].factorize()[0].astype(np.int32) % 2e6).astype(np.int32)
            df_clicks[col] = df_clicks[col].factorize()[0].astype(np.int32)
            continue
        elif col in ['hour', 'day', 'weekday', 'publish_hour', 'publish_day', 'publish_weekday', 'leak']:
            df_clicks[col] = df_clicks[col].astype(np.uint8)
        else:
            df_clicks[col] = df_clicks[col].factorize()[0].astype(np.int32)
    feature_size = {}
    for col in df_clicks:
        if col in ['clicked', 'display_id', 'timestamp', 'ad_id']:
            continue
        feature_size[col] = df_clicks[col].max()
    # feature_size = df_clicks.drop(['clicked', 'display_id', 'timestamp', 'ad_id'], axis=1, inplace=False) \
    #                         .max() \
    #                         .to_dict()
    return feature_size, df_clicks


def main():
    data_fname = sys.argv[1]
    leak_fname = '../leak.csv'

    """
    feature_size, df_clicks_origin = generate_data(leak_fname)
    # perform some feature selection or engineering here
    # df_clicks = df_clicks[['display_id', 'ad_id', 'campaign_id', 'clicked', 'timestamp', 'leak', 'hour', 'weekday', 'day', 'geo_location', 'source_id', 'cat_1', 'topic_1', 'topic_2', 'platform']]
    df_clicks = df_clicks_origin[['display_id', 'ad_id', 'clicked', 'timestamp']]
    df_clicks['ad_count'] = df_clicks_origin.display_id.map(df_clicks_origin.groupby('display_id', sort=False)['ad_id'].count().to_dict())
    # df_clicks['ad_order'] = df_clicks_origin.groupby('display_id', sort=False)['ad_id'].transform(lambda x: x.index)

    # df_clicks['uuid__ad_id'] = (df_clicks['uuid'].astype(str) + '_' + df_clicks['ad_id_fact'].astype(str)).factorize()[0].astype(np.int32) % 5e7
    # df_clicks['ad_id__document_id'] = (df_clicks['ad_id_fact'].astype(str) + '_' + df_clicks['document_id'].astype(str)).factorize()[0].astype(np.int32) % 3e7
    # df_clicks['ad_id__cat_1'] = (df_clicks['ad_id_fact'].astype(str) + '_' + df_clicks['cat_1'].astype(str)).factorize()[0].astype(np.int32) % 1e6
    # df_clicks['ad_id__document_id'] = (df_clicks['ad_id_fact'].astype(str) + '_' + df_clicks['document_id'].astype(str)).factorize()[0].astype(np.int32)
    # df_clicks['ad_id__document_id'] = (df_clicks['ad_id_fact'].astype(str) + '_' + df_clicks['document_id'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['campaign_id__ad_count'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks['ad_count'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['campaign_id__geo_location'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks_origin['geo_location'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['campaign_id__leak'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks_origin['leak'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['campaign_id__document_id'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks_origin['document_id'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['campaign_id__source_id'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks_origin['source_id'].astype(str)).factorize()[0].astype(np.int32)
    # df_clicks['campaign_id__uuid'] = (df_clicks_origin['campaign_id'].astype(str) + '_' + df_clicks_origin['uuid'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['ad_id__weekday__hour'] = (df_clicks_origin['ad_id_fact'].astype(str) + '_' + df_clicks_origin['weekday'].astype(str) + '_' + df_clicks_origin['hour'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['ad_id__day__hour'] = (df_clicks_origin['ad_id_fact'].astype(str) + '_' + df_clicks_origin['day'].astype(str) + '_' + df_clicks_origin['hour'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['ad_id__platform'] = (df_clicks_origin['ad_id_fact'].astype(str) + '_' + df_clicks_origin['platform'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['ad_id__source_id'] = (df_clicks_origin['ad_id_fact'].astype(str) + '_' + df_clicks_origin['source_id'].astype(str)).factorize()[0].astype(np.int32)
    df_clicks['ad_id__leak'] = (df_clicks_origin['ad_id_fact'].astype(str) + '_' + df_clicks_origin['leak'].astype(str)).factorize()[0].astype(np.int32)

    # df_clicks.drop('ad_count', axis=1, inplace=True)
    feature_size = df_clicks.drop(['clicked', 'display_id', 'timestamp', 'ad_id'], axis=1, inplace=False) \
                            .max() \
                            .to_dict()
    """
    feature_size, df_clicks = generate_data(leak_fname)
    # import ipdb; ipdb.set_trace()
    df_clicks = df_clicks[[
        'display_id', 'ad_id', 'clicked', 'timestamp',
        'campaign_id',
        # 'advertiser_id'
        'leak',
        # 'ad_id_fact',
        # 'document_id_ad_source',
        # 'document_id_orig_source',
        'document_id_ad',
        'document_id_orig',
        # 'document_id_ad_appear_time',
        'ad_id_appear_time',
        'ad_count',
        'day',
        'weekday__hour',
        # 'publisher_doc_id_ad__orig',
        # 'first_category_doc_id_ad__orig',
        # 'first_topic_doc_id_ad__orig',
        # 'document_id_ad_topic_num',
        'orig_doc_id__first_second_category',
        'ad_doc_id__first_second_category',
        'ad_doc_id__first_second_topic',
        'orig_doc_id__first_second_topic',
        'platform',
        'geo_location',
        # 'uuid',
        'doc_hour_views_map',
        # 'doc_views_map',
        # 'doc_orig_views_map',
    ]]
    # df_clicks_train_ad_clicked = df_clicks[df_clicks.clicked != NAN_VALUE][['clicked', 'campaign_id']].astype(int)
    # mean_clicked = df_clicks_train_ad_clicked.clicked.mean()
    # df_agg_ad = df_clicks_train_ad_clicked.groupby('ad_id', sort=False).clicked.agg(['count', 'sum', 'mean'])
    # df_agg_ad['likelihood'] = (df_agg_ad['sum'] + 12*mean_clicked) / (12 + df_agg_ad['count'])
    # adid2likelyhood = df_agg_ad['likelihood'].to_dict()
    # import ipdb; ipdb.set_trace()
    # df_clicks = df_clicks.merge(df_clicks[df_clicks.clicked != NAN_VALUE].astype({'clicked': int}).groupby(['day', 'campaign_id'], sort=False, as_index=False)['clicked'].mean(), how='left', on=['day', 'campaign_id'], suffixes=['', '_day']).fillna({'clicked_day': 0.0})
    # import ipdb; ipdb.set_trace()
    # df_clicks['other_ad'] = df_clicks.groupby('display_id', sort=False)['ad_id'].transform(lambda x: list(x))
    # df_clicks['clicked_history'] = df_clicks.groupby('campaign_id', sort=False)['clicked'].rolling(window=10).sum().fillna('0000000000')
    # df_clicks['source_same'] = (df_clicks.document_id_ad_source == df_clicks.document_id_orig_source).astype(int)
    # df_clicks['category_orig_first_ad_second_same'] = (df_clicks.document_id_orig_first_category == df_clicks.document_id_ad_second_category).astype(int)
    # df_clicks['ad_id__platform'] = (df_clicks['ad_id_fact'] * 5 + df_clicks['platform']).factorize()[0].astype(np.int32)
    # df_clicks['campaign_id__source_id'] = (df_clicks['campaign_id'] * 20000 + df_clicks['document_id_orig_source']).factorize()[0].astype(np.int32)
    # df_clicks['likelihood'] = df_clicks.ad_id.map(adid2likelyhood).fillna(0.0)
    df_clicks['_index'] = np.arange(df_clicks.shape[0])
    # df_clicks['day__week__hour'] = (df_clicks.day * 200 + df_clicks.weekday__hour).factorize()[0].astype(np.int32)
    # df_clicks['ad_order'] = (df_clicks['_index'] - df_clicks.display_id.map(df_clicks.groupby('display_id', sort=False)['_index'].first().to_dict())).astype(int)
    df_clicks['platform__ad_count'] = ((df_clicks.platform + 1) * 100 + df_clicks.ad_count).factorize()[0].astype(np.int32)
    df_clicks.drop(['_index', 'platform', 'ad_count'], axis=1, inplace=True)
    feature_size = df_clicks.drop(['clicked', 'display_id', 'timestamp', 'ad_id'], axis=1, inplace=False) \
                            .max() \
                            .to_dict()
    # import ipdb; ipdb.set_trace()
    gc.collect()

    # df_clicks.drop(['uuid', 'geo_3', 'topic_5', 'topic_4', 'topic_3', 'publisher_id', 'publish_day', 'publish_hour', 'publish_weekday', 'advertiser_id'], axis=1, inplace=True)
    # import ipdb; ipdb.set_trace()
    df_train = df_clicks[df_clicks['clicked'] != NAN_VALUE]
    df_test = df_clicks[df_clicks['clicked'] == NAN_VALUE].drop(['clicked', 'timestamp'], axis=1)

    print('Split data')
    df_display_time = df_train.groupby('display_id', as_index=False, sort=False)['timestamp'].mean()
    max_time = df_display_time['timestamp'].max()
    subtrain_display_ids = df_display_time[df_display_time['timestamp'] < (max_time * 0.95)].display_id.tolist()
    validation_display_ids = df_display_time[df_display_time['timestamp'] >= (max_time * 0.95)].display_id.tolist()
    shuffle(subtrain_display_ids)
    sample_display_ids = set(validation_display_ids + subtrain_display_ids[:int(len(subtrain_display_ids)*0.08)])
    df_subtrain, df_validation = \
        df_train[~df_train['display_id'].isin(sample_display_ids)].drop('timestamp', axis=1), \
        df_train[df_train['display_id'].isin(sample_display_ids)].drop('timestamp', axis=1)
    # df_subtrain = df_train[df_train['timestamp'].astype(int) < (max_time * 0.95)].drop('timestamp', axis=1)
    # df_validation = df_train[df_train['timestamp'].astype(int) >= (max_time * 0.92)].drop('timestamp', axis=1)

    # del df_train
    # subtrain_display_ids = list(df_subtrain['display_id'].unique())
    # shuffle(subtrain_display_ids)
    # sample_display_ids = set(subtrain_display_ids[:int(len(subtrain_display_ids)*0.1)])

    # df_subtrain, df_validation = \
    #     df_subtrain[~df_subtrain['display_id'].isin(sample_display_ids)], \
    #     pd.concat([df_subtrain[df_subtrain['display_id'].isin(sample_display_ids)], df_validation], axis=0)
    # # msk = np.random.rand(len(df_subtrain)) < 0.9
    # # df_subtrain, df_validation = \
    # #     df_subtrain[msk], \
    # #     pd.concat([df_subtrain[~msk], df_validation], axis=0)

    # import ipdb; ipdb.set_trace()
    print('Store data')
    data = {}
    data['subtrain'] = df_subtrain
    data['validation'] = df_validation
    data['test'] = df_test
    data['feature_size'] = feature_size
    with open(data_fname, 'wb') as fw:
        pkl.dump(data, fw, protocol=4)


if __name__ == '__main__':
    main()
