import sys
import pickle as pkl
# import tempfile

import xgboost as xgb
import pandas as pd

from utils import df2mapk


DROP_FEATURE = ['uuid', 'document_id', 'ad_id_fact']


def main():
    data_fname_base = sys.argv[1]
    subtrain_fname = '../subtrain_' + data_fname_base
    subtrain_id_fname = subtrain_fname + '.id'
    validation_fname = '../validation_' + data_fname_base
    validation_id_fname = validation_fname + '.id'
    test_fname = '../test_' + data_fname_base
    test_id_fname = test_fname + '.id'

    print('read id files...')
    df_subtrain = pd.read_csv(subtrain_id_fname, dtype=str)
    # df_subtrain.drop(DROP_FEATURE, axis=1, inplace=True)
    subtrain_group_size = df_subtrain.groupby('display_id', sort=False)['ad_id'].count().tolist()
    df_subtrain['clicked'] = 0
    df_subtrain['pred'] = 0.

    df_validation = pd.read_csv(validation_id_fname, dtype=str)
    # df_validation.drop(DROP_FEATURE, axis=1, inplace=True)
    validation_group_size = df_validation.groupby('display_id', sort=False)['ad_id'].count().tolist()
    df_validation['clicked'] = 0
    df_validation['pred'] = 0.

    df_test = pd.read_csv(test_id_fname, dtype=str)
    # df_test.drop(DROP_FEATURE, axis=1, inplace=True)
    test_group_size = df_test.groupby('display_id', sort=False)['ad_id'].count().tolist()
    df_test['pred'] = 0.

    # import ipdb; ipdb.set_trace()

    print('create xgb data')
    dsubtrain = xgb.DMatrix(subtrain_fname)
    dsubtrain.set_group(subtrain_group_size)
    dvalidation = xgb.DMatrix(validation_fname)
    dvalidation.set_group(validation_group_size)
    dtest = xgb.DMatrix(test_fname)
    dtest.set_group(test_group_size)

    print('Train XGB')
    param = {'max_depth':5, 'eta':0.01, 'subsample': 0.2, 'colsample_bytree': 0.2, 'silent': 1}
    # param['booster'] = 'gbtree'
    param['nthread'] = 8
    param['eval_metric'] = 'map@12'
    param['objective'] = 'rank:pairwise'
    # param['objective'] = 'binary:logistic'
    param['seed'] = 1234

    watchlist  = [(dvalidation, 'eval'), (dsubtrain,'train')]
    num_round = 10000
    bst = xgb.train(param, dsubtrain, num_round, watchlist, early_stopping_rounds=100)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
