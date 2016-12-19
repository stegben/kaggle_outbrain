import sys
import pickle as pkl

import pandas as pd


def df2ffm(df, fname, is_train=True):
    df_new = pd.DataFrame()
    drop_column = ['display_id', 'ad_id']

    for idx, col in enumerate(df.drop(drop_column, axis=1)):
        print(idx, col)
        df_new[str(idx + 1)] = str(idx+1) + ':' + df[col].astype(str) + ':' + str(1)
    if is_train:
        df_new['0'] = df['clicked']
        drop_column.append('clicked')
    else:
        df_new['0'] = 0
    # with open(fname, 'w') as fw:
    #     for row in df_new.iterrows():
    #         fw.write(' '.join([str(r) for r in row]))
    df_new.sort_index(axis=1).to_csv(fname, sep=' ', header=False, index=False)
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

    print('Store ffm data')
    # df2ffm(df_subtrain, '../subtrain_' + libffm_data_fname, is_train=True)
    # del df_subtrain
    # df2ffm(df_validation, '../validation_' + libffm_data_fname, is_train=True)
    # del df_validation
    df2ffm(df_test, '../test_' + libffm_data_fname, is_train=False)


if __name__ == '__main__':
    main()
