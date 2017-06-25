import os
import sys
import gc
import pickle as pkl

import pandas as pd


def df2ffm(df, fname, feature_size, is_train=True):
    df_new = pd.DataFrame()
    drop_column = ['display_id', 'ad_id']
    if is_train:
        drop_column.append('clicked')

    feature_start_idx = 0
    for idx, col in enumerate(df.drop(drop_column, axis=1)):
        if col in ['clicked_day', 'likelihood']:
            print(idx, col)
            df_new[str(idx + 1)] = str(idx+1) + ':' + str(int(feature_start_idx)) + ':' + df[col].astype(str)
            feature_start_idx += 1
            continue
        print(idx, col)
        df_new[str(idx + 1)] = str(idx+1) + ':' + (df[col] + feature_start_idx).astype(int).astype(str) + ':' + str(1)
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
    # df_new.sort_index(axis=1).to_csv(fname, sep=' ', header=False, index=False)
    df_new.to_csv(fname, sep=' ', header=False, index=False)
    if is_train:
        df[['display_id', 'ad_id', 'clicked']].to_csv(fname+'.id', index=False)
    else:
        df[['display_id', 'ad_id']].to_csv(fname+'.id', index=False)
    del df_new


def main():
    print('Read data...')
    data_fname = sys.argv[1]
    libffm_data_fname = sys.argv[2]
    with open(data_fname, 'rb') as f:
        data = pkl.load(f)
    df_subtrain = data['subtrain']
    df_validation = data['validation']
    df_test = data['test']
    feature_size = data['feature_size']

    print('Store ffm data')
    subtrain_fname = '../subtrain_' + libffm_data_fname
    validation_fname = '../validation_' + libffm_data_fname
    if not os.path.exists(subtrain_fname):
        df2ffm(df_subtrain, subtrain_fname, feature_size, is_train=True)
        del df_subtrain
        gc.collect()
    if not os.path.exists(validation_fname):
        df2ffm(df_validation, validation_fname, feature_size, is_train=True)
        del df_validation
        gc.collect()
    df2ffm(df_test, '../test_' + libffm_data_fname, feature_size, is_train=False)


if __name__ == '__main__':
    main()
