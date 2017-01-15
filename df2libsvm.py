import os
import gc
import sys
import pickle as pkl

import numpy as np
import pandas as pd


def df2svm(df, fname, feature_size, is_train=True):
    df_new = pd.DataFrame()
    drop_column = ['display_id', 'ad_id']
    if is_train:
        drop_column.append('clicked')

    feature_start_idx = 0
    print(df.columns)
    for idx, col in enumerate(df.drop(drop_column, axis=1)):
        print(idx, col)
        df_new[str(idx + 1)] = (df[col].astype(np.int32) + feature_start_idx).astype(str) + ':' + str(1)
        feature_start_idx += feature_size[col] + 1
        del df[col]
    if is_train:
        df_new.insert(0, '0', df['clicked'])
        drop_column.append('clicked')
    else:
        df_new.insert(0, '0', 0)
    # with open(fname, 'w') as fw:
    #     for row in df_new.iterrows():
    #         fw.write(' '.join([str(r) for r in row]))
    # columns = list(df_new.columns)
    # columns.remove('0')
    # columns = ['0'] + columns
    df_new.to_csv(fname, sep=' ', header=False, index=False)
    # df_new.sort_index(axis=1).to_csv(fname, sep=' ', header=False, index=False)
    if is_train:
        df[['display_id', 'ad_id', 'clicked']].to_csv(fname+'.id', index=False)
    else:
        df[['display_id', 'ad_id']].to_csv(fname+'.id', index=False)
    del df_new


def main():
    print('Read data...')
    data_fname = sys.argv[1]
    libsvm_data_fname = sys.argv[2]
    with open(data_fname, 'rb') as f:
        data = pkl.load(f)
    df_subtrain = data['subtrain']
    df_validation = data['validation']
    df_test = data['test']
    feature_size = data['feature_size']

    print('Store svm data')
    subtrain_fname = '../subtrain_' + libsvm_data_fname
    validation_fname = '../validation_' + libsvm_data_fname
    test_fname = '../test_' + libsvm_data_fname
    if not os.path.exists(subtrain_fname):
        df2svm(df_subtrain, subtrain_fname, feature_size, is_train=True)
        del df_subtrain
        gc.collect()
    if not os.path.exists(validation_fname):
        df2svm(df_validation, validation_fname, feature_size, is_train=True)
        del df_validation
        gc.collect()
    if not os.path.exists(test_fname):
        df2svm(df_test, test_fname, feature_size, is_train=False)


if __name__ == '__main__':
    main()
