import sys
import pickle as pkl
import tempfile

import numpy as np
import pandas as pd



NAN_VALUE = '0000'



def main():
    data_fname = sys.argv[1]
    print('Read data')
    df_clicks_train = pd.read_csv('../clicks_train.csv', engine='c', dtype=str)
    df_clicks_test = pd.read_csv('../clicks_test.csv', engine='c', dtype=str)

    df_document_categories = pd.read_csv('../documents_categories.csv', engine='c', dtype=str)
    df_document_entities = pd.read_csv('../documents_entities.csv', engine='c', dtype=str)
    df_document_meta = pd.read_csv('../documents_meta.csv', engine='c', dtype=str)
    df_document_topics = pd.read_csv('../documents_topics.csv', engine='c', dtype=str)

    df_events = pd.read_csv('../events.csv', engine='c', dtype=str)
    df_promoted_content = pd.read_csv('../promoted_content.csv', engine='c', dtype=str)

    print('Start joining tables')
    df_train = pd.merge(df_clicks_train, df_events, on='display_id', how='left')
    df_train = pd.merge(df_train, df_promoted_content.drop('document_id', axis=1), on='ad_id', how='left')
    df_train = df_train[df_train.platform != '\\N'].fillna(NAN_VALUE)
    df_test = pd.merge(df_clicks_test, df_events, on='display_id', how='left')
    df_test = pd.merge(df_test, df_promoted_content.drop('document_id', axis=1), on='ad_id', how='left').fillna(NAN_VALUE)

    print('Split data')
    max_time = df_train['timestamp'].astype(int).max()
    df_subtrain = df_train[df_train['timestamp'].astype(int) < (max_time * 0.8)].drop('timestamp', axis=1)
    df_validation = df_train[df_train['timestamp'].astype(int) >= (max_time * 0.8)].drop('timestamp', axis=1)

    print('Store data')
    data = {}
    data['subtrain'] = df_subtrain
    data['validation'] = df_validation
    data['test'] = df_test
    with open(data_fname, 'wb') as fw:
        pkl.dump(data, fw)


if __name__ == '__main__':
    main()
